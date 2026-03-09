# ☀️ Swin-iFold: Solar Power Forecasting via Spatiotemporal Folding

[![SOTA](https://img.shields.io/badge/SOTA-Solar--AL-gold)](https://github.com/your-username/Swin-iFold)


**Swin-iFold** is a specialized architecture for high-dimensional time-series forecasting. It reimagines 1D solar irradiance data as a 2D spatiotemporal grid ("folding"), allowing the **Swin Transformer** to capture cyclical solar patterns through hierarchical windowed attention.



## 🏆 The $8 \times 12$ Breakthrough
After exhaustive testing of multiple topologies ($4 \times 24$, $6 \times 16$, and $16 \times 6$), the **$8 \times 12$ configuration** emerged as the "Golden Ratio" for the Solar AL dataset. 

While higher resolutions (like $16 \times 6$) overfitted to training noise, the $8 \times 12$ structure (representing exactly **2 hours per row**) provided the perfect balance between local temporal smoothing and global trend awareness.

### 📊 Benchmark Comparison
| Model / Topology | Test MSE | Generalization Gap | Status |
| :--- | :--- | :--- | :--- |
| iTransformer | 0.203 | - | ❌ Defeated |
| TimeMixer | 0.189 | - | ❌ Defeated |
| TimeMixer++ | 0.171 | - | ❌ Defeated |
| **Swin-iFold ($8 \times 12$)** | **0.1636** |  | **👑 Champion** |


## 🧠 Core Architecture
The model utilizes a sequence of 96 historical points (16 hours) and folds them into a 2D matrix.

1. **Folding Strategy**: The 1D sequence is reshaped into $H \times W = 8 \times 12$. Each row captures a 2-hour window of solar activity.
2. **Swin Blocks**: Utilizing $(4, 4)$ windows to capture 40 minutes of horizontal variance and 8 hours of vertical (daily arc) progression.
3. **DAE Block**: A Denoising Disruption Block forces the model to learn robust features by reconstructing signal from injected Gaussian and drift noise.
4. **RevIN**: Reversible Instance Normalization handles the non-stationary nature of solar data.



---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch 2.x
* 4x NVIDIA RTX 4090 (Recommended for DDP)
* `timm` library

### Training
To reproduce the SOTA results using the $8 \times 12$ champion topology:

```bash
torchrun --nproc_per_node=4 train_v8_8x12.py
