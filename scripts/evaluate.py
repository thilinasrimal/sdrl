"""
scripts/evaluate.py
-------------------
Evaluate a trained checkpoint and produce visualizations.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth \
                                --data data/processed/paired \
                                --n_samples 8 --save_dir outputs/figures
"""
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SDRL model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data",       type=Path, default=Path("data/processed/paired"))
    p.add_argument("--n_samples",  type=int,  default=8,
                   help="Number of samples to visualize")
    p.add_argument("--save_dir",   type=Path, default=Path("outputs/figures"))
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import matplotlib.pyplot as plt
    from sdrl.model   import build_models
    from sdrl.loss    import compute_metrics
    from sdrl.dataset import PairedGravityDataset
    from torch.utils.data import DataLoader

    # device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # load checkpoint
    ck = torch.load(args.checkpoint, map_location=device)
    cfg = ck.get("cfg", {})
    base_ch = cfg.get("model", {}).get("base_ch", 32)
    P, D = build_models(base_ch=base_ch, device=device)
    P.load_state_dict(ck["primary_state"])
    P.eval()
    print(f"Loaded checkpoint: epoch {ck['epoch']}, "
          f"val SSIM {ck['val_ssim']:.4f}")

    # data
    ds = PairedGravityDataset(str(args.data), augment=False)
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=False)
    lr_batch, hr_batch = next(iter(loader))

    # inference
    with torch.no_grad():
        sr_batch = P(lr_batch.to(device)).cpu()

    # ── metrics ─────────────────────────────────────────────────────────
    all_metrics = []
    for i in range(len(lr_batch)):
        m = compute_metrics(sr_batch[i:i+1], hr_batch[i:i+1])
        all_metrics.append(m)

    avg = {k: float(np.mean([m[k] for m in all_metrics]))
           for k in all_metrics[0]}

    print(f"\n── Evaluation on {len(lr_batch)} samples ──")
    print(f"  PSNR : {avg['psnr']:.2f} dB")
    print(f"  SNR  : {avg['snr']:.2f} dB")
    print(f"  SSIM : {avg['ssim']:.4f}")
    print(f"  MSE  : {avg['mse']:.6f}")
    print(f"  MAE  : {avg['mae']:.6f}")
    print(f"  MRE  : {avg['mre']:.4f}")

    # ── visualize ────────────────────────────────────────────────────────
    n = min(args.n_samples, 4)
    fig, axes = plt.subplots(n, 3, figsize=(11, n * 3.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    kw = dict(cmap="RdBu_r", vmin=0, vmax=1)
    for i in range(n):
        m = all_metrics[i]
        axes[i, 0].imshow(lr_batch[i, 0].numpy(),  **kw)
        axes[i, 0].set_title("LR input (satellite)", fontsize=9)
        axes[i, 1].imshow(sr_batch[i, 0].numpy(),  **kw)
        axes[i, 1].set_title(
            f"SR output  SSIM={m['ssim']:.3f}  PSNR={m['psnr']:.1f}dB",
            fontsize=9)
        axes[i, 2].imshow(hr_batch[i, 0].numpy(),  **kw)
        axes[i, 2].set_title("HR ground truth (shipborne)", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        f"SDRL Gravity SR  |  avg SSIM={avg['ssim']:.4f}  "
        f"PSNR={avg['psnr']:.2f}dB  MRE={avg['mre']:.4f}",
        fontsize=11)
    plt.tight_layout()
    out_path = args.save_dir / "eval_samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
