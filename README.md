# SDRL Gravity Super-Resolution

Implementation of **"Super-Resolution Reconstruction of Gravity Data Using Semi-Supervised Dual Regression Learning"**  
*Remote Sens. 2026, 18, 453* — Jia et al.

A semi-supervised deep learning framework that enhances marine gravity resolution by fusing sparse high-resolution shipborne data with wide-coverage low-resolution satellite altimetry.

---

## Architecture Overview

```
Satellite LR (50×50 @ 1′)  ──►  PrimaryNet (LR→HR)  ──►  SR Output (200×200 @ 15″)
                                       │                          │
                                       ▼                          ▼
                               DualNet (HR→LR)  ◄────────────────┘
                                       │
                                       ▼
                              Cycle-consistency loss
```

- **PrimaryNet**: Encoder-decoder with LiteMLA attention + RCAB blocks (×4 super-resolution)  
- **DualNet**: Learned degradation network (HR→LR) for cycle-consistent self-supervision  
- **Loss**: Composite L1 + SSIM with learnable α weighting (Eq. 3 & 5 from paper)

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/sdrl-gravity.git
cd sdrl-gravity

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install
pip install -e ".[dev]"
```

### 2. Add your data

```
data/
  raw/
    grav_33.1.nc          # SSV33.1 satellite altimetry (download from SIO)
    shipborne/
      A_faa.grd           # Shipborne survey files
      B_faa.grd
      CS_faa.grd
      ... (all .grd files)
```

**Satellite data**: https://topex.ucsd.edu/pub/  
**Shipborne data**: https://www.marine-geo.org

### 3. Preprocess

```bash
python scripts/preprocess.py --sat data/raw/grav_33.1.nc \
                              --ship data/raw/shipborne \
                              --out data/processed
```

### 4. Train

```bash
# Full SDRL (semi-supervised, recommended)
python scripts/train.py --config configs/sdrl.yaml

# Supervised baseline only
python scripts/train.py --config configs/supervised.yaml
```

### 5. Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth \
                            --data data/processed/paired
```

---

## Project Structure

```
sdrl-gravity/
├── sdrl/                    # Core library
│   ├── __init__.py
│   ├── model.py             # PrimaryNet + DualNet architectures
│   ├── loss.py              # CompositeLoss + SDRLLoss (Eq. 3 & 5)
│   ├── dataset.py           # Paired + Unpaired datasets + DataLoader
│   └── preprocess.py        # Memory-safe per-file preprocessing
├── scripts/
│   ├── preprocess.py        # CLI preprocessing entry point
│   ├── train.py             # CLI training entry point
│   └── evaluate.py          # CLI evaluation + metrics
├── configs/
│   ├── sdrl.yaml            # Semi-supervised SDRL config
│   └── supervised.yaml      # Supervised baseline config
├── tests/
│   ├── test_model.py        # Model shape + forward pass tests
│   ├── test_loss.py         # Loss function tests
│   └── test_dataset.py      # Dataset loading tests
├── notebooks/
│   └── explore.ipynb        # Data exploration notebook
├── data/                    # (git-ignored)
├── outputs/                 # (git-ignored)
├── pyproject.toml           # Package + dependency definition
├── .github/workflows/
│   └── ci.yml               # GitHub Actions: test on push
└── README.md
```

---

## Configuration

Edit `configs/sdrl.yaml` to change training settings:

```yaml
model:
  base_ch: 32          # network width (32=default, 64=larger)

train:
  epochs: 500
  batch_size: 8
  lr: 1e-4
  lam: 1.0             # cycle-consistency weight λ
  mu: 0.5              # dual regression weight µ
  sigma: 0.5           # dual consistency weight σ

data:
  patch_lr: 50
  patch_hr: 200
  stride_lr: 25
  ssim_lo: 0.2
  ssim_hi: 0.9
  rmse_max: 0.4
```

---

## Development with Cursor

Open the repo folder in Cursor. The `.cursorrules` file configures Claude to understand the project context. Suggested workflow:

- Ask Claude to **explain** any function: *"explain the LiteMLA attention mechanism in model.py"*
- Ask Claude to **debug**: *"why is the SSIM loss not decreasing after epoch 50?"*
- Ask Claude to **extend**: *"add a inference script that runs on a new .grd file"*

---

## Results (from paper)

| Method | PSNR (dB) | SSIM | MRE |
|---|---|---|---|
| LR input | 18.97 | 0.69 | 0.69 |
| Supervised | 16.96 | 0.85 | 0.34 |
| **SDRL (ours)** | **18.56** | **0.89** | **0.07** |
| SDRL (50% labels) | 17.88 | 0.87 | 0.10 |

---

## Citation

```bibtex
@article{jia2026sdrl,
  title   = {Super-Resolution Reconstruction of Gravity Data Using
             Semi-Supervised Dual Regression Learning},
  author  = {Jia, Bode and Sun, Jian and Geng, Xiangfeng and
             Wan, Xiaolei and Liu, Huaishan},
  journal = {Remote Sensing},
  volume  = {18},
  number  = {3},
  pages   = {453},
  year    = {2026},
  doi     = {10.3390/rs18030453}
}
```
