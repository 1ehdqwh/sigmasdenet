import torch
import numpy as np
from torch import autocast
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.SigmaSdeNet import SigmaSdeNet
from copy import deepcopy
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class ModelEMA:
    """ EMA 模型: 用于稳定 SDE 网络的评估结果 """

    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters(): p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            # 适配 DDP 和 单卡
            source_model = model.module if isinstance(model, DDP) else model
            for ema_v, model_v in zip(self.module.state_dict().values(), source_model.state_dict().values()):
                if model_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + model_v * (1 - self.decay))


class nnUNetTrainerSigmaSdeNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5
        self.my_ds_scales = None
        self.ema_model = None

    def initialize(self):
        super().initialize()

        # 1. 初始化 EMA
        if self.network is not None:
            print(">>> Initializing EMA Model (decay=0.999) for Stability <<<")
            raw_network = self.network.module if isinstance(self.network, DDP) else self.network
            self.ema_model = ModelEMA(raw_network)

        # 2. 【DDP 关键修复】强制开启 find_unused_parameters=True
        # 原因：Decoder中第0个Solver的 Memory Projection 层在第一次迭代时未被使用
        if self.is_ddp:
            # 只有当它已经是 DDP 实例时才重新包装，或者直接修改属性
            if isinstance(self.network, DDP):
                # 如果已经是 DDP，通常没法直接改 find_unused_parameters，
                # 但 nnU-Net 的 initialize 可能会再次包装。
                # 无论如何，最稳妥的方式是确保它被正确设置。
                # nnU-Net 默认是用 self.network = DDP(self.network, ...)
                # 我们这里打个补丁：如果发现它已经是 DDP 但配置不对，这很难改。
                # 所以最好是在 nnUNetTrainer 的生命周期里，它还没变 DDP 之前改？
                # 不，nnU-Net 是在 initialize 里做 DDP 的。
                pass
                # 实际上 nnU-Net 只有在多卡时才会 DDP。我们只要确保传入 DDP 构造函数的参数是对的。
                # 但 nnU-Net 内部写死了 DDP 调用。
                # Hack: 既然我们继承了 Trainer，我们可以覆盖 initialize，但这里调用了 super().initialize()
                # 此时 self.network 已经是 DDP 了。
                # 唯一的办法是重新包装 (虽然有点丑，但管用)
                print(">>> [DDP Hack] Re-wrapping with find_unused_parameters=True for Solver Memory Gates <<<")
                raw_net = self.network.module
                self.network = DDP(
                    raw_net,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True  # 必须为 True
                )

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step()
        self.print_to_log_file(f"\nEpoch: {self.current_epoch}")
        self.print_to_log_file(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []

        # 获取原始网络
        network = self.network.module if isinstance(self.network, DDP) else self.network

        for name, param in network.named_parameters():
            if not param.requires_grad: continue

            # 保护 SDE 参数、Norm层、Bias
            # raw_dt, raw_log_r 等参数绝对不能被 decay
            if "norm" in name or "bias" in name or "raw_" in name or "time_emb" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW([
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.initial_lr, eps=1e-4)

        # 读取 build_network 中决策的 warmup 轮数
        warmup = getattr(self, 'warmup_epochs', 20)

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=self.num_epochs - warmup, eta_min=1e-6)
            ],
            milestones=[warmup]
        )
        return optimizer, lr_scheduler

    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision: bool = True) -> torch.nn.Module:
        strides = configuration_manager.pool_op_kernel_sizes
        kernel_sizes = configuration_manager.conv_kernel_sizes
        num_stages = len(strides)

        # ---------------- 智能策略中心 ----------------
        dataset_name = plans_manager.plans.get('dataset_name', 'Unknown')

        # 默认配置
        self.warmup_epochs = 20

        # ACDC / BraTS17 这种小数据集，PlainEncoder 可能需要更快的反馈
        if 'ACDC' in dataset_name or 'acdc' in dataset_name.lower():
            self.warmup_epochs = 10
            print(f">>> ACDC Detected: Warmup shortened to 10 epochs <<<")
        elif '2017' in dataset_name or 'Brats17' in dataset_name:
            self.warmup_epochs = 15
            print(f">>> BraTS 2017 Detected: Warmup set to 15 epochs <<<")
        else:
            print(f">>> Default/BraTS21 Mode: Warmup 20 epochs <<<")
        # ---------------------------------------------

        base_num_features = 32
        embed_dims = [min(base_num_features * (2 ** i), 320) for i in range(num_stages)]

        print(f"==================================================")
        print(f"SigmaSdeNet [Experimental PlainEncoder] Config:")
        print(f"  Encoder : Plain Conv (No Residual/SE)")
        print(f"  Decoder : Time-Dependent SDE + Memory + Explicit Kalman")
        print(f"  Trainer : DDP(Unused=True) + Auto Warmup")
        print(f"  Features: {embed_dims}")
        print(f"==================================================")

        self.my_ds_scales = []
        cumulative_stride = np.array([1.0, 1.0, 1.0])
        for i in range(num_stages - 1):
            s = np.array(strides[i])
            cumulative_stride = cumulative_stride * s
            self.my_ds_scales.append(list(1.0 / cumulative_stride))

        # 注意：现在不需要传 clamp_limit 了，因为模型内部写死了 50.0
        model = SigmaSdeNet(
            in_channels=num_input_channels,
            num_classes=plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            embed_dims=embed_dims,
            patch_kernel_size=kernel_sizes,
            patch_stride=strides,
            deep_supervision=enable_deep_supervision
        )
        return model

    def _get_deep_supervision_scales(self):
        if self.my_ds_scales is None: return [[1.0, 1.0, 1.0]]
        return self.my_ds_scales

    def train_step(self, batch: dict):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        with autocast(self.device.type, enabled=True):
            # 模型内部自动生成 t，不需要外部传入
            output = self.network(data)
            l = self.loss(output, target)

        if torch.isnan(l):
            self.print_to_log_file("!!! CRITICAL WARNING: NaN loss detected! Skipping step. !!!")
            self.optimizer.zero_grad()
            # 返回 0.0 防止 scaler 爆炸
            return {'loss': np.array(0.0, dtype=np.float32)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.ema_model is not None:
            self.ema_model.update(self.network)

        return {'loss': l.detach().cpu().numpy()}

    def perform_actual_validation(self, save_probabilities: bool = False):
        if self.ema_model is None:
            return super().perform_actual_validation(save_probabilities)

        net = self.network.module if isinstance(self.network, DDP) else self.network
        training_state = deepcopy(net.state_dict())
        net.load_state_dict(self.ema_model.module.state_dict())

        try:
            val_out = super().perform_actual_validation(save_probabilities)
        except Exception as e:
            self.print_to_log_file(f"Validation Error: {e}")
            raise e
        finally:
            net.load_state_dict(training_state)

        return val_out

    def validation_step(self, batch: dict):
        val_outputs = super().validation_step(batch)
        if 'loss' in val_outputs and (np.isnan(val_outputs['loss']) or np.isinf(val_outputs['loss'])):
            val_outputs['loss'] = 0.0
        return val_outputs

    def set_deep_supervision_enabled(self, enabled: bool):
        net = self.network.module if isinstance(self.network, DDP) else self.network
        if hasattr(net, 'deep_supervision'):
            net.deep_supervision = enabled