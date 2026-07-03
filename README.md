# NZ-SDRL: Marine Gravity Super-Resolution for the New Zealand EEZ

A regional adaptation of the **Semi-Supervised Dual Regression Learning (SDRL)** framework for 4× super-resolution of marine gravity fields over the New Zealand Exclusive Economic Zone (NZ EEZ), with focus on the **Hikurangi Margin** and **Chatham Rise** priority sub-regions.

This implementation is based on:
> Jia, B., Sun, J., Geng, X., Wan, X., & Liu, H. (2026). *Super-Resolution Reconstruction of Gravity Data Using Semi-Supervised Dual Regression Learning.* Remote Sensing, 18(3), 453. https://doi.org/10.3390/rs18030453

Developed as part of thesis research **IT9115** at Whitecliffe, New Zealand, supervised by Dr. Nabeel Shaukat and Dr. Shahbaz Pervez.

---

## What This Does

Enhances marine gravity spatial resolution from approximately **1 arcminute (~1.85 km)** satellite altimetry to approximately **15 arcseconds (~230 m)** — a 4× improvement — by learning a mapping from low-resolution (LR) SIO V33.1 satellite gravity to high-resolution (HR) GNS Science shipborne survey measurements.

```
SIO V33.1 satellite gravity          GNS shipborne gravity
  LR input  50×50 pixels      →      SR output  200×200 pixels
  ~1 arcmin per pixel                ~15 arcsec per pixel
  (~1.85 km at mid-latitude)         (~230 m at mid-latitude)
```

### Architecture

```
LR patch (50×50 · 1ch)
        │
        ▼
┌─────────────────────────────────────────────┐
│              PrimaryNet  (36M params)        │  ← LR → SR
│  Conv + MeanShift → LiteMLA → Downsample    │
│       ↓              skip connections ──────┐│
│  LiteMLA → Downsample (50×50 · 80ch)       ││
│       ↓                                     ││
│  RCAB + Upsample → RCAB + Upsample ←───────┘│
│       ↓                                      │
│  PixelShuffle 4× → SR output (200×200 · 1ch)│
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│              DualNet  (2M params)            │  ← SR/HR → LR
│  Conv + LeakyReLU → Downsample × 3          │
│  Learnable degradation (not fixed kernel)    │
└─────────────────────────────────────────────┘
        │
        ▼
  Cycle-consistency loss  +  Dual regression loss
```

**PrimaryNet**: Encoder-decoder with multi-scale linear attention (LiteMLA, O(N) complexity) and Residual Channel Attention Blocks (RCAB) with PixelShuffle upsampling.

**DualNet**: Learnable HR→LR degradation model that captures satellite altimetry biases (orbital aliasing, along-track filtering) rather than assuming a fixed downsampling kernel.

**Loss function** (Eq. 5, Jia et al. 2026):
```
L = λ · L(D(P(x)), x)                          # cycle-consistency (unpaired)
  + I_M(x) · [ L(P(x), y) + μ · L(D(y), x)    # supervised (paired only)
             + σ · L(D(P(x)), D(y)) ]           # dual consistency
```
where L(a,b) = (1−α)·‖a−b‖₁ + α·(1−SSIM(a,b)), α=0.84 learnable.
An additional Sobel gradient loss (grad_w=0.4) penalises first-order derivative deviations.

---

## NZ EEZ Results

Trained on 800 paired LR-HR patches from the **Hikurangi Margin / Chatham Rise** region (172–179.9°E, 45–37°S), 150 epochs on a Kaggle P100 GPU.

| Metric | Value | Notes |
|---|---|---|
| **SSIM** | **0.9998** | Full test set (1,440 patches) |
| **PSNR** | **45.90 dB** | Full test set |
| **MAE** | **0.0053** | Normalized [0,1] units |
| **MRE** | **0.0098** | Per-patch mean relative error |
| SSIM — hard patches (top 10% by variance) | 0.9997 | HR std ≥ 0.0127 |
| PSNR — hard patches | 44.57 dB | |
| SSIM — easy patches (bottom 50%) | 0.9998 | HR std ≤ 0.0062 |
| PSNR — easy patches | 46.53 dB | |
| **PSNR gap (hard vs easy)** | **1.96 dB** | Near-consistent across terrain complexity |

> **Note:** Evaluation uses a random 90/10 in-distribution hold-out (not a geographically independent test region). The 1.96 dB terrain-stratified gap is the primary generalisation-relevant metric.

---

## Data Sources

| Dataset | Variable | Resolution | Source |
|---|---|---|---|
| SIO V33.1 satellite gravity | Free-air anomaly (mGal) | 1 arcmin | [topex.ucsd.edu](https://topex.ucsd.edu/pub/) |
| GNS freeair.asc | Free-air anomaly (mGal) | ~0.017° | GNS Science NZ |
| GEBCO 2026 | Bathymetry (m) | 15 arcsec | [gebco.net](https://www.gebco.net) |
| EGM2008 | Geoid (mGal) | ~5 arcmin | [icgem.gfz-potsdam.de](http://icgem.gfz-potsdam.de) |
| NZ Bathymetry 2016 | Bathymetry (m) | ~50 m | NIWA / LINZ |

**Study bounds:** `[172, -45, 179.9, -37]` (lon_min, lat_min, lon_max, lat_max)
**Training:** 800 paired LR-HR patches + 500 unpaired LR patches
**Patch sizes:** LR 50×50 px · HR 200×200 px · Stride 30 px · Min HR coverage 90%

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/thilinasrimal/sdrl.git
cd sdrl
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Organize data

```
data/
  raw/
    grav_33.1.nc          # SIO V33.1 satellite gravity
    freeair.asc           # GNS Science NZ shipborne free-air gravity
    gebco_2026.nc         # GEBCO 2026 bathymetry
    EGM2008.tif           # EGM2008 geoid reference
    nzbathy_2016.tif      # NZ Bathymetry 2016 (NZTM2000 CRS — see note below)
```

> ⚠️ **NZBathy CRS note:** `nzbathy_2016.tif` uses NZTM2000 projected coordinates (metres, EPSG:2193), not geographic degrees. The preprocessing pipeline currently reprojects this to WGS84 geographic before masking. Verify your file's CRS with `rasterio.open(path).crs` before running.

### 3. Preprocess

```bash
python scripts/preprocess.py \
    --sat   data/raw/grav_33.1.nc \
    --ship  data/raw/freeair.asc \
    --gebco data/raw/gebco_2026.nc \
    --egm   data/raw/EGM2008.tif \
    --out   data/processed \
    --bounds 172 -45 179.9 -37
```

This produces:
- `data/processed/nz_eez_features.npz` — common-grid multi-source feature stack including `gravity_raw` (raw SIO mGal field, not EGM-residual)
- `data/processed/paired/paired.npz` — 800 aligned LR-HR patch pairs with `global_min`/`global_max` normalization parameters
- `data/processed/unpaired/lr_patches.npy` — 500 unpaired LR patches for semi-supervised training

> ⚠️ **Load `gravity_raw`, not `features[0]`:** The feature stack channel 0 is an EGM2008-residual (already normalized). For SR training, always load `features['gravity_raw']` as the LR input to ensure physical consistency with the raw shipborne HR reference.

### 4. Train

```bash
# Full SDRL — semi-supervised (recommended)
python scripts/train.py --config configs/sdrl.yaml

# Supervised baseline only
python scripts/train.py --config configs/supervised.yaml
```

The best checkpoint is saved to `outputs/checkpoints/best_model_nz.pth` based on peak validation SSIM.

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model_nz.pth \
    --data data/processed/paired
```

Produces per-patch SSIM/PSNR/MAE/MRE, terrain-stratified metrics (top 10% vs bottom 50% by HR variance), and a PSNR-vs-SSIM joint distribution plot.

### 6. Export full-region SR grid

```bash
python scripts/export_sr_grid.py \
    --checkpoint outputs/checkpoints/best_model_nz.pth \
    --features   data/processed/nz_eez_features.npz \
    --paired     data/processed/paired/paired.npz \
    --out        outputs/nz_eez_gravity_4x_mgal.npy \
    --patch 50 --stride 30
```

Outputs a `(15364 × 15172)` float32 array in real mGal units covering the full study region.

---

## Configuration

```yaml
# configs/sdrl.yaml
model:
  base_ch: 48              # network width — 36M params at 48, 9M at 32

train:
  n_epochs: 150
  batch_size: 4            # 2 paired + 2 unpaired per batch
  lr: 5.0e-5
  lr_min: 1.0e-7
  warmup_epochs: 10
  grad_clip: 0.5
  val_split: 0.1
  lam: 0.2                 # cycle-consistency weight λ
  mu: 0.3                  # dual regression weight μ
  sigma: 0.3               # dual consistency weight σ
  grad_w: 0.4              # Sobel gradient loss weight
  alpha: 0.84              # SSIM/L1 balance in composite loss

data:
  patch_lr: 50
  patch_hr: 200
  stride: 30
  min_hr_coverage: 0.9     # minimum shipborne coverage fraction per patch
  max_patches: 800
```

---

## Project Structure

```
sdrl/
├── sdrl/
│   ├── model.py            # PrimaryNet (36M) + DualNet (2M) architectures
│   ├── loss.py             # CompositeLoss · SDRLLoss (Eq. 3 & 5 from paper)
│   ├── dataset.py          # PairedDataset · UnpairedDataset · DataLoader
│   └── preprocess.py       # Memory-safe per-patch preprocessing (no global HR alloc)
├── scripts/
│   ├── preprocess.py       # CLI: multi-source data alignment and patch extraction
│   ├── train.py            # CLI: SDRL / supervised training with cyclic LR
│   ├── evaluate.py         # CLI: patch metrics + terrain-stratified evaluation
│   └── export_sr_grid.py   # CLI: full-region sliding-window SR export (mGal output)
├── configs/
│   ├── sdrl.yaml           # Semi-supervised SDRL config (recommended)
│   └── supervised.yaml     # Supervised baseline config
├── tests/
│   ├── test_model.py       # Model shape + forward pass
│   ├── test_loss.py        # Loss function unit tests
│   └── test_dataset.py     # Dataset loading + alignment correlation check
├── notebooks/
│   └── explore.ipynb       # Data exploration + patch alignment diagnostics
├── data/                   # (git-ignored)
├── outputs/                # (git-ignored)
├── pyproject.toml
└── README.md
```

---

## Known Issues and Pipeline Notes

### 1. ESRI ASCII Grid north-south convention
The `freeair.asc` file stores rows **north-to-south** (row 0 = northernmost latitude). The lat coordinate array built from `yllcorner` is ascending (south-to-north). Without `np.flipud(grid)` applied unconditionally after loading, every coordinate-based query retrieves data from the mirror-image latitude. This is handled automatically in `sdrl/preprocess.py` — do not add a conditional flip.

### 2. Use `gravity_raw`, not `features['features'][0]`
Channel 0 of the multi-source feature stack is the EGM2008-residual after normalization (~0.18 std in [0,1] space). For SR training, always load `features['gravity_raw']` (raw SIO free-air in mGal) to ensure the LR and HR inputs represent the same physical quantity. Pairing a residual-scale LR with an absolute-scale HR is a training-invisible failure that produces flat, near-zero-variance patches and corrupts all subsequent metrics.

### 3. Per-patch HR interpolation (no global HR array)
The preprocessing pipeline evaluates the HR RectBivariateSpline interpolator **per patch**, never over a global pre-allocated 15k×15k array. Attempting to build the full-resolution HR grid upfront requires approximately 15 GB RAM and crashes the Kaggle P100 kernel silently. The current implementation keeps peak HR-specific memory at approximately 0.3 MB per patch step.

### 4. NZBathy 2016 CRS mismatch
The NZ Bathymetry 2016 file uses NZTM2000 projected coordinates (metres, EPSG:2193). Geographic masking with degree-valued study bounds returns an empty subset without prior CRS transformation. See preprocessing note above.

---

## Patch Alignment Diagnostic

Before training, verify data alignment with the built-in diagnostic:

```python
import numpy as np
from scipy.ndimage import zoom

data = np.load('data/processed/paired/paired.npz')
lr, hr = data['lr'], data['hr']

corrs = [np.corrcoef(zoom(lr[i,0], 4, order=3).ravel(), hr[i,0].ravel())[0,1]
         for i in range(len(lr))]
corrs = np.array(corrs)

print(f"Mean correlation:   {corrs.mean():.4f}  (expect > 0.7)")
print(f"% positive:         {(corrs > 0).mean()*100:.1f}%  (expect > 95%)")
print(f"% strong (> 0.5):   {(corrs > 0.5).mean()*100:.1f}%  (expect > 90%)")
```

**Expected healthy output:**
```
Mean correlation:   0.935   (expect > 0.7)
% positive:         99.4%   (expect > 95%)
% strong (> 0.5):   98.9%   (expect > 90%)
```

If correlation mean is near zero or the distribution is bimodal (≈50% near +1, 50% near -1), stop and diagnose before training — a training loss curve will appear normal even with completely misaligned data.

---

## Comparison with Reference Paper

| Configuration | SSIM | PSNR (dB) | MRE |
|---|---|---|---|
| Jia et al. — supervised baseline | 0.8470 | 16.96 | 0.3430 |
| Jia et al. — SDRL (100% labels) | 0.8852 | 18.56 | 0.0659 |
| Jia et al. — SDRL (50% labels) | 0.8696 | 17.88 | 0.0978 |
| Jia et al. — SDRL, unseen test region | 0.9244 | 17.15 | 0.2709 |
| **This repo — NZ EEZ, in-distribution** | **0.9998** | **45.90** | **0.0098** |

> The higher PSNR/SSIM in this repo primarily reflects in-distribution evaluation (random hold-out from the same regional dataset) rather than the geographically independent test used by Jia et al. A spatial hold-out evaluation is planned.

---

## Planned Extensions

- [ ] **Radial power spectrum comparison** — FFT-based validation that SR recovers genuine high-frequency content beyond the LR Nyquist limit (vs. bicubic baseline)
- [ ] **Spatial generalization test** — train on Hikurangi sub-region, evaluate on Chatham Rise (and vice versa)
- [ ] **NZBathy channel** — fix NZTM2000 → WGS84 reprojection to enable 2-channel and 4-channel NZGravNet variants
- [ ] **Physics-informed loss** — discrete Laplacian residual penalty enforcing the harmonic constraint on the free-air anomaly field
- [ ] **Uncertainty quantification** — Monte Carlo dropout for per-pixel prediction confidence maps
- [ ] **Model card** — structured documentation of training provenance, spatial validity, and recommended use conditions

---

## Citation

If you use this code, please cite both the original SDRL paper and this regional implementation:

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

@mastersthesis{srimal2026nzgravsr,
  title   = {Deep Learning-Based Super-Resolution of Marine Gravity Fields
             over the New Zealand Exclusive Economic Zone},
  author  = {Srimal, Thilina},
  school  = {Whitecliffe, New Zealand},
  year    = {2026},
  note    = {IT9115 Master of Information Technology,
             supervisors: Dr. Nabeel Shaukat and Dr. Shahbaz Pervez}
}
```

---

## Licence

MIT — see `LICENSE`.

---

## Acknowledgements

SIO V33.1 satellite gravity data: Scripps Institution of Oceanography, University of California San Diego. Shipborne gravity compilation: GNS Science New Zealand. Bathymetry: GEBCO Compilation Group. Geoid: International Centre for Global Earth Models (ICGEM). Training infrastructure: Kaggle (P100 GPU).
