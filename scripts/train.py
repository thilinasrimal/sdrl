"""
scripts/train.py
----------------
CLI entry point for SDRL training.

Usage:
    python scripts/train.py --config configs/sdrl.yaml
    python scripts/train.py --config configs/sdrl.yaml --resume
    python scripts/train.py --config configs/supervised.yaml
"""
import argparse
import os
import time
import yaml
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train SDRL gravity super-resolution")
    p.add_argument("--config", type=Path, default=Path("configs/sdrl.yaml"))
    p.add_argument("--resume", action="store_true",
                   help="Resume from best checkpoint if available")
    p.add_argument("--device", default=None,
                   help="Device override, e.g. 'cuda', 'cpu', 'mps'")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(args.config.read_text())

    import torch
    from sdrl.model   import build_models
    from sdrl.loss    import SDRLLoss, CompositeLoss, compute_metrics
    from sdrl.dataset import SDRLDataLoader, PairedGravityDataset
    from torch.utils.data import DataLoader, random_split
    from tqdm import tqdm

    # ── device ──────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"           # Apple Silicon
    else:
        device = "cpu"
    print(f"Device: {device}")

    tc  = cfg["train"]
    dc  = cfg["data"]
    oc  = cfg["output"]
    mc  = cfg["model"]
    mode = tc["mode"]           # "sdrl" | "supervised"

    torch.manual_seed(tc["seed"])
    ckpt_dir = Path(oc["checkpoint_dir"])
    log_dir  = Path(oc["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.config.stem}_log.csv"

    # ── models ──────────────────────────────────────────────────────────
    P, D = build_models(base_ch=mc["base_ch"], device=device)

    if mode == "sdrl":
        criterion = SDRLLoss(
            lam=tc["lam"], mu=tc["mu"], sigma=tc["sigma"]).to(device)
        optimizer = torch.optim.Adam([
            {"params": P.parameters(),           "lr": tc["lr_P"]},
            {"params": D.parameters(),           "lr": tc["lr_D"]},
            {"params": criterion.L.parameters(), "lr": tc.get("lr_alpha", 1e-3)},
        ])
    else:  # supervised
        criterion = SDRLLoss(lam=0, mu=0, sigma=0).to(device)
        optimizer = torch.optim.Adam([
            {"params": P.parameters(),           "lr": tc["lr_P"]},
            {"params": criterion.L.parameters(), "lr": tc.get("lr_alpha", 1e-3)},
        ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["epochs"], eta_min=1e-6)

    # ── data ────────────────────────────────────────────────────────────
    if mode == "sdrl":
        dl = SDRLDataLoader(
            paired_dir  = dc["paired_dir"],
            unpair_dir  = dc["unpair_dir"],
            batch_size  = tc["batch_size"],
            num_workers = tc.get("num_workers", 2),
            augment     = True,
            val_split   = tc.get("val_split", 0.1),
            seed        = tc["seed"],
        )
        steps_per_epoch = dl.steps_per_epoch()
        train_gen       = dl.train_batches()
        val_loader      = dl.val_loader
    else:
        # supervised: only paired data, split train/val
        from sdrl.dataset import PairedGravityDataset
        full = PairedGravityDataset(dc["paired_dir"], augment=True)
        n_val   = max(1, int(len(full) * tc.get("val_split", 0.1)))
        n_train = len(full) - n_val
        g = torch.Generator().manual_seed(tc["seed"])
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
        half = tc["batch_size"] // 2
        train_loader = DataLoader(train_ds, batch_size=half * 2,
                                  shuffle=True, num_workers=tc.get("num_workers", 2),
                                  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=half * 2, shuffle=False)
        steps_per_epoch = len(train_loader)

    # ── resume ──────────────────────────────────────────────────────────
    best_ckpt   = ckpt_dir / "best_model.pth"
    start_epoch = 0
    best_ssim   = -1.0

    if args.resume and best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device)
        P.load_state_dict(ck["primary_state"])
        D.load_state_dict(ck["dual_state"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_ssim   = ck["val_ssim"]
        print(f"Resumed from epoch {ck['epoch']}, best SSIM={best_ssim:.4f}")

    # ── logging ─────────────────────────────────────────────────────────
    if not log_path.exists():
        log_path.write_text(
            "epoch,train_loss,train_cycle,train_recon,"
            "val_psnr,val_snr,val_ssim,val_mse,val_mae,val_mre,lr\n")

    print(f"\nTraining [{mode}] for {tc['epochs']} epochs from {start_epoch}")
    print("=" * 65)

    for epoch in range(start_epoch, tc["epochs"]):
        P.train(); D.train()
        t0     = time.time()
        accum  = {"total": 0., "cycle": 0., "recon": 0.}
        n_steps = 0

        it = tqdm(range(steps_per_epoch),
                  desc=f"Ep {epoch:03d}", leave=False)

        for _ in it:
            optimizer.zero_grad()

            if mode == "sdrl":
                batch = next(train_gen)
                x_p = batch["lr_paired"].to(device)
                y_p = batch["hr_paired"].to(device)
                x_u = batch["lr_unpaired"].to(device)
                lp  = criterion(P, D, x_p, y_p)
                lu  = criterion(P, D, x_u, None)
                loss = 0.5 * (lp["total"] + lu["total"])
                accum["cycle"] += lp["cycle"].item()
                accum["recon"] += lp["recon"].item()
            else:
                x_p, y_p = next(iter(train_loader))
                x_p = x_p.to(device); y_p = y_p.to(device)
                lp   = criterion(P, D, x_p, y_p)
                loss = lp["total"]
                accum["cycle"] += 0.
                accum["recon"] += lp["recon"].item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            optimizer.step()
            accum["total"] += loss.item()
            n_steps += 1
            it.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ── validation ───────────────────────────────────────────────
        P.eval(); D.eval()
        vm_list = []
        with torch.no_grad():
            for lr_v, hr_v in val_loader:
                vm_list.append(
                    compute_metrics(P(lr_v.to(device)), hr_v.to(device)))
        vm  = {k: float(np.mean([m[k] for m in vm_list])) for k in vm_list[0]}
        cur_lr = optimizer.param_groups[0]["lr"]

        avg = {k: v / n_steps for k, v in accum.items()}
        elapsed = time.time() - t0
        print(f"Ep {epoch:03d} | loss {avg['total']:.4f} "
              f"cycle {avg['cycle']:.4f} recon {avg['recon']:.4f} | "
              f"SSIM {vm['ssim']:.4f} PSNR {vm['psnr']:.2f} "
              f"MRE {vm['mre']:.4f} | {elapsed:.0f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg['total']:.6f},{avg['cycle']:.6f},"
                    f"{avg['recon']:.6f},"
                    f"{vm['psnr']:.4f},{vm['snr']:.4f},{vm['ssim']:.4f},"
                    f"{vm['mse']:.6f},{vm['mae']:.6f},{vm['mre']:.6f},"
                    f"{cur_lr:.2e}\n")

        # ── checkpointing ────────────────────────────────────────────
        state = dict(epoch=epoch,
                     primary_state=P.state_dict(),
                     dual_state=D.state_dict(),
                     optimizer=optimizer.state_dict(),
                     val_ssim=vm["ssim"], cfg=cfg)

        if (epoch + 1) % oc.get("save_every", 10) == 0:
            torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")

        if vm["ssim"] > best_ssim:
            best_ssim = vm["ssim"]
            torch.save(state, best_ckpt)
            print(f"  ★ best SSIM {best_ssim:.4f} → best_model.pth")

    print(f"\nDone. Best SSIM: {best_ssim:.4f}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
