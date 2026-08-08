# SigmaSDENet

Official PyTorch implementation of **SigmaSDENet: Unscented Kalman Filter Guided Stochastic Differential Equations for 3-D Medical Image Segmentation**.

SigmaSDENet reformulates 3-D medical image segmentation decoding as a **stochastic prediction--correction process**. Instead of directly fusing encoder skip features, the proposed decoder treats them as observations of an evolving latent state and performs uncertainty-aware correction using a UKF-inspired mechanism.

## Overview

SigmaSDENet consists of:

- A plain hierarchical 3-D convolutional encoder;
- A neural memory stochastic differential equation (**nmSDE**) decoder;
- Three-point subspace sigma sampling;
- Diagonal uncertainty propagation and moment reconstruction;
- UKF-inspired uncertainty-aware **Active Fusion**;
- Progressive memory propagation across decoder stages.

The overall decoding process can be summarized as:

**Stochastic State Prediction → Moment Reconstruction → Observation-based Correction**
