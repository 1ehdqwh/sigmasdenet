import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math


# ============================================================================
# 1. 基础组件: PlainConv
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv3d(nn.Module):
    """3D depthwise convolution followed by pointwise channel mixing."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class PlainStage(nn.Module):
    def __init__(self, in_ch, out_ch, stride, k_size):
        super().__init__()
        if isinstance(k_size, (list, tuple)):
            padding = tuple([(k - 1) // 2 for k in k_size])
        else:
            padding = (k_size - 1) // 2

        self.conv_ops = nn.Sequential(
            ConvBlock(in_ch, out_ch, k_size, stride, padding),
            ConvBlock(out_ch, out_ch, 3, 1, 1)
        )

    def forward(self, x):
        return self.conv_ops(x)


class PlainEncoder(nn.Module):
    def __init__(self, in_channels, stages_ch, strides, kernel_sizes):
        super().__init__()
        self.stages = nn.ModuleList()
        current_ch = in_channels
        for i, (out_ch, stride, k_size) in enumerate(zip(stages_ch, strides, kernel_sizes)):
            stage = PlainStage(current_ch, out_ch, stride, k_size)
            self.stages.append(stage)
            current_ch = out_ch

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


# ============================================================================
# 2. Time Embedding
# ============================================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, t):
        args = t * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t)], dim=-1)
        return emb


# ============================================================================
# 3. Sigma-SDE Solver (核心修改区)
# ============================================================================
class Sigma_SDE_Solver(nn.Module):
    def __init__(self, conv_op, skip_features, hidden_features, memory_features,
                 nonlin, ukf_dim=8, dt_emb_dim=16,
                 use_separable_convs=True,
                 sigma_scales=(0.0, 1.0, -1.0)):
        super().__init__()
        if len(sigma_scales) < 1:
            raise ValueError("sigma_scales must contain at least one value")
        self.ukf_dim = ukf_dim
        self.hidden_features = hidden_features
        self.anchor_dim = min(16, hidden_features)
        self.dt_emb_dim = dt_emb_dim
        self.sigma_scales = tuple(float(i) for i in sigma_scales)

        # Projections
        self.proj_down = conv_op(hidden_features, ukf_dim, 1, bias=False)
        self.proj_up = conv_op(ukf_dim, hidden_features, 1, bias=False)
        self.mem_proj = conv_op(memory_features, self.anchor_dim, 1, bias=False)

        # Gate
        self.mem_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(hidden_features, 1), nn.Sigmoid()
        )

        # Condition Fusion
        self.g_net = nn.Sequential(
            conv_op(skip_features + self.anchor_dim, hidden_features, 1, bias=False),
            nonlin()
        )

        # Drift
        drift_conv = DepthwiseSeparableConv3d if use_separable_convs else conv_op
        self.f_net = nn.Sequential(
            drift_conv(hidden_features, hidden_features, 3, padding=1, bias=False),
            nonlin(),
            drift_conv(hidden_features, hidden_features, 3, padding=1, bias=False)
        )

        # Diffusion MLP
        self.diffusion_mlp = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(hidden_features, hidden_features // 2), nn.LeakyReLU(),
            nn.Linear(hidden_features // 2, hidden_features)
        )

        # Correction / Observation Matrix (H)
        self.h_net = conv_op(hidden_features, skip_features, 3, padding=1, bias=False)

        # Neural dt
        self.time_mlp = nn.Sequential(
            nn.Linear(ukf_dim + dt_emb_dim, ukf_dim),
            nn.LeakyReLU(),
            nn.Linear(ukf_dim, 1)
        )

        self.time_emb = TimeEmbedding(dt_emb_dim)

        # 观测噪声 R
        self.raw_log_r = nn.Parameter(torch.full((1, skip_features, 1, 1, 1), -3.0))

    @property
    def R_diag(self):
        return F.softplus(torch.clamp(self.raw_log_r, max=5.0)) + 1e-4

    def soft_clamp(self, x, limit=50.0):
        return limit * torch.tanh(x / limit)

    def forward(self, mean, cov_diag, skip, memory, t):
        B, C, D, H, W = mean.shape
        r = self.ukf_dim

        # --- 1. Memory Handling ---
        if memory is not None:
            if memory.shape[2:] != (D, H, W):
                memory = F.interpolate(memory, size=(D, H, W), mode='trilinear', align_corners=False)
            mem_feat = self.mem_proj(memory)
        else:
            mem_feat = torch.zeros(B, self.anchor_dim, D, H, W, device=mean.device, dtype=mean.dtype)
        mem_feat = self.soft_clamp(mem_feat, 50.0)

        # --- 2. Condition ---
        if skip.shape[2:] != (D, H, W):
            skip = F.interpolate(skip, size=(D, H, W), mode='trilinear', align_corners=False)
        skip = self.soft_clamp(skip, 50.0)
        cond = torch.cat([skip, mem_feat], dim=1)

        # --- 3. Subspace Sigma Points ---
        z_mean = self.proj_down(mean)
        W_sq = self.proj_down.weight.view(r, C).pow(2)
        z_cov = torch.einsum('rc,bcdhw->brdhw', W_sq, cov_diag)
        z_std = torch.sqrt(z_cov + 1e-5)

        sigma_scales = z_std.new_tensor(self.sigma_scales).view(-1, 1, 1, 1, 1, 1)
        sigmas_z = z_mean.unsqueeze(0) + sigma_scales * z_std.unsqueeze(0)
        num_sigmas = len(self.sigma_scales)
        sigmas_z_reshaped = sigmas_z.view(num_sigmas * B, r, D, H, W)
        sigmas_h = self.proj_up(sigmas_z_reshaped)
        cond_rep = cond.repeat(num_sigmas, 1, 1, 1, 1)

        # --- 4. Drift Evolution ---
        fused = self.soft_clamp(sigmas_h, 50) + self.g_net(cond_rep)
        drift = -sigmas_h + self.f_net(fused)
        drift = self.soft_clamp(drift, 20.0)

        # --- 5. Neural dt ---
        t_emb = self.time_emb(t)
        t_emb_rep = t_emb.repeat(num_sigmas, 1)
        sigma_flat = sigmas_z_reshaped.flatten(2).mean(-1)
        dt_net_input = torch.cat([sigma_flat, t_emb_rep], dim=-1)
        dt = F.softplus(self.time_mlp(dt_net_input)).view(num_sigmas * B, 1, 1, 1, 1)
        dt = torch.clamp(dt, min=0.001, max=0.5)

        # --- 6. Diffusion ---
        diffusion_scale = F.softplus(self.diffusion_mlp(sigmas_h)).view(num_sigmas * B, C, 1, 1, 1)
        noise = torch.randn_like(diffusion_scale)
        diffusion_term = diffusion_scale * noise * torch.sqrt(dt)

        # --- 7. Evolve ---
        evolved = sigmas_h + drift * dt + diffusion_term
        evolved = self.soft_clamp(evolved, 50.0)
        evolved = evolved.view(num_sigmas, B, C, D, H, W)

        # --- 8. Reconstruct ---
        post_mean = evolved.mean(dim=0)
        diff = evolved - post_mean.unsqueeze(0)
        post_cov = diff.pow(2).mean(dim=0)
        post_cov = torch.clamp(post_cov, max=20.0)

        # ================================================================
        # --- 9. Kalman Update (UKF 纯正路线重构) ---
        # ================================================================

        # (1) 均值预测及残差计算
        prop_z = self.h_net(evolved.flatten(0, 1)).view(num_sigmas, B, -1, D, H, W)
        pred_zm = prop_z.mean(dim=0)
        innov = self.soft_clamp(skip - pred_zm, 50.0)

        # (2) 直接使用 Sigma 点在观测空间的样本方差近似 HPH^T
        # 彻底抛弃 3x3 卷积核的 1x1 坍缩，回归 UKF 无迹变换的本质
        diff_z = prop_z - pred_zm.unsqueeze(0)
        S_sample = diff_z.pow(2).mean(dim=0)

        # S = 样本方差 + 观测噪声 R
        S = torch.clamp(S_sample + self.R_diag, min=1e-3)
        inv_S = 1.0 / S  # 显式求逆

        # (3) 计算归一化残差及均值修正
        norm_innovation = innov * inv_S

        # 投影回隐空间 (使用 h_net 转置)
        padding = tuple([k // 2 for k in self.h_net.kernel_size])
        correction_dir = F.conv_transpose3d(norm_innovation, self.h_net.weight, padding=padding)

        # 修正均值: K * innov ≈ post_cov * (H^T * inv_S * innov)
        correction = self.soft_clamp(post_cov * correction_dir, 50.0)
        corrected_mean = self.soft_clamp(post_mean + correction, 50.0)

        # (4) 更新协方差 P_new = (I - KH) P = P - P * (H^T * S^-1 * H) * P
        # 放弃 1x1 近似，改用严格的 3x3 转置卷积将不确定度下降率投影回隐空间
        H_sq = self.h_net.weight.pow(2)  # shape: (C_out, C_in, 3, 3, 3)

        # 使用平方权重计算 H^T * S^-1 * H
        # conv_transpose3d 的 weight_shape 正好匹配 forward 时 Conv3d 的 weight_shape
        H_T_invS_H = F.conv_transpose3d(inv_S, H_sq, padding=padding)

        # 更新后验协方差 (利用对角假设近似缩放比例)
        gain_ratio = torch.clamp(H_T_invS_H * post_cov, max=0.9)  # 限制最大缩减比例
        corrected_cov = post_cov * (1.0 - gain_ratio)
        corrected_cov = torch.clamp(corrected_cov, min=1e-5)

        # --- 10. Memory Update ---
        gate = self.mem_gate(corrected_mean).view(B, 1, 1, 1, 1)
        if memory is None:
            new_memory = corrected_mean
        else:
            new_memory = gate * memory + (1 - gate) * corrected_mean

        return corrected_mean, corrected_cov, new_memory


# ============================================================================
# 4. Decoder
# ============================================================================
class Sigma_SDE_Decoder(nn.Module):
    def __init__(self, encoder_channels, encoder_strides, num_classes,
                 deep_supervision, memory_features,
                 use_separable_convs=True,
                 sigma_scales=(0.0, 1.0, -1.0)):
        super().__init__()
        self.hidden_features = encoder_channels[0]
        self.deep_supervision = deep_supervision

        self.init_proj = nn.Conv3d(encoder_channels[-1], self.hidden_features, 1)
        self.init_log_cov = nn.Parameter(torch.full((1, self.hidden_features, 1, 1, 1), -1.0))

        self.upsamples = nn.ModuleList()
        self.solvers = nn.ModuleList()

        n = len(encoder_channels)
        for s in range(n - 1):
            skip_ch = encoder_channels[n - 2 - s]
            stride = encoder_strides[n - 1 - s]
            if isinstance(stride, (list, tuple)):
                scale = tuple(float(i) for i in stride)
            else:
                scale = float(stride)

            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False),
                    ConvBlock(self.hidden_features, self.hidden_features, 3, 1, 1)
                )
            )

            self.solvers.append(
                Sigma_SDE_Solver(
                    conv_op=nn.Conv3d,
                    skip_features=skip_ch,
                    hidden_features=self.hidden_features,
                    memory_features=self.hidden_features,
                    nonlin=nn.LeakyReLU,
                    ukf_dim=8,
                    use_separable_convs=use_separable_convs,
                    sigma_scales=sigma_scales,
                )
            )

    def forward(self, skips, t):
        skips = skips[::-1]
        x = self.init_proj(skips[0])
        cov = F.softplus(self.init_log_cov).expand_as(x)
        out = []
        mem = None

        for i in range(len(self.solvers)):
            skip = skips[i + 1]
            xu = self.upsamples[i](x)
            cov_up = F.interpolate(cov, size=xu.shape[2:], mode='nearest')

            if self.training:
                x_s, cov_s, mem = checkpoint(self.solvers[i], xu, cov_up, skip, mem, t, use_reentrant=False)
            else:
                x_s, cov_s, mem = self.solvers[i](xu, cov_up, skip, mem, t)

            x = xu + x_s
            cov = cov_s
            out.append(x)

        return out[::-1]


# ============================================================================
# 5. 主网络
# ============================================================================
class SigmaSdeNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3,
                 embed_dims=[32, 64, 128, 256, 320, 320],
                 patch_stride=None, patch_kernel_size=None,
                 deep_supervision=True, use_separable_convs=True,
                 sigma_scales=(0.0, 1.0, -1.0),
                 **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_separable_convs = use_separable_convs
        if patch_stride is None:
            patch_stride = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        if patch_kernel_size is None:
            patch_kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

        self.encoder = PlainEncoder(in_channels, embed_dims, patch_stride, patch_kernel_size)

        self.decoder = Sigma_SDE_Decoder(
            encoder_channels=embed_dims,
            encoder_strides=patch_stride,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            memory_features=32,
            use_separable_convs=use_separable_convs,
            sigma_scales=sigma_scales,
        )

        self.seg_heads = nn.ModuleList([
            nn.Conv3d(self.decoder.hidden_features, num_classes, 1)
            for _ in range(len(embed_dims) - 1)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips = self.encoder(x)

        # 自动生成 t
        B = x.shape[0]
        device = x.device
        if self.training:
            t = torch.rand((B, 1), device=device, dtype=x.dtype)
        else:
            t = torch.full((B, 1), 0.5, device=device, dtype=x.dtype)

        decoder_feats = self.decoder(skips, t)

        outputs = []
        for i, feat in enumerate(decoder_feats):
            logits = self.seg_heads[i](feat)
            logits = 50.0 * torch.tanh(logits / 50.0)
            if i == 0:
                if logits.shape[2:] != x.shape[2:]:
                    logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
            outputs.append(logits)

        if self.deep_supervision:
            return outputs
        else:
            return outputs[0]

    def measure_complexity(self, input_shape=(1, 1, 32, 256, 224)):
        """
        自动计算模型参数量和 GFLOPs。
        注意：thop 无法穿透 torch.utils.checkpoint，所以必须在 eval 模式下运行。
        """
        # 切换到 eval 模式以禁用 checkpoint (否则 thop 统计不到 solver 内部)
        training_state = self.training
        self.eval()

        try:
            from thop import profile, clever_format

            device = next(self.parameters()).device
            dummy_input = torch.randn(input_shape).to(device)

            # 运行 thop 分析
            macs, params = profile(self, inputs=(dummy_input,), verbose=False)

            # 格式化输出
            macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
            gflops = macs / 1e9 * 2  # 粗略估计：1 MAC ≈ 2 FLOPs

            print(f"\n{'=' * 50}")
            print(f"Model Complexity Analysis:")
            print(f"  Input Shape : {input_shape}")
            print(f"  Params      : {params_fmt}")
            print(f"  MACs (thop) : {macs_fmt}")
            print(f"  GFLOPs      : {gflops:.3f} G")
            print(f"{'=' * 50}\n")

        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
        finally:
            # 恢复之前的训练状态
            self.train(training_state)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 使用 BraTS 的典型 Patch Size 进行测试
    model = SigmaSdeNet(in_channels=1, num_classes=3).cuda()

    # 打印参数量和计算量
    model.measure_complexity(input_shape=(1, 1, 32, 256, 224))

    # 简单的 Forward 测试
    x = torch.randn(1, 1, 32, 256, 224).cuda()
    y = model(x)
    print(f"Output type: {type(y)}")
    if isinstance(y, list):
        print(f"Deep Supervision Outputs: {len(y)}")
        print(f"High-Res Output Shape: {y[0].shape}")
    else:
        print(f"Output Shape: {y.shape}")
