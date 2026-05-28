"""
loss.py
-------
Loss functions for SDRL gravity super-resolution.
Based on: Remote Sens. 2026, 18, 453  (Equations 3 and 5)

Upload to: /content/drive/MyDrive/sdrl_gravity/loss.py
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim as pt_ssim


# ════════════════════════════════════════════════════════════════════════════
# Composite loss  L(a, b)  — Eq. (3)
# ════════════════════════════════════════════════════════════════════════════
class CompositeLoss(nn.Module):
    """
    L(a, b) = (1 - α)·‖a - b‖₁  +  α·(1 - SSIM(a, b))

    α is a *learnable* scalar initialised at 0.9 (paper Section 3.2).
    Both a and b should be in range [0, 1] (already normalised patches).
    """
    def __init__(self, alpha_init: float = 0.9):
        super().__init__()
        # store as logit so gradient can push it without going out of [0,1]
        self._alpha_logit = nn.Parameter(
            torch.tensor(self._to_logit(alpha_init)))

    @staticmethod
    def _to_logit(p):
        return float(torch.log(torch.tensor(p / (1.0 - p))))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_logit)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        α = self.alpha
        l1   = torch.mean(torch.abs(a - b))
        ssim_val = pt_ssim(a, b, data_range=1.0, size_average=True)
        return (1.0 - α) * l1 + α * (1.0 - ssim_val)


# ════════════════════════════════════════════════════════════════════════════
# SDRL semi-supervised loss  L_semi  — Eq. (5)
# ════════════════════════════════════════════════════════════════════════════
class SDRLLoss(nn.Module):
    """
    L_semi = λ·L(D(P(x)), x)
           + I_M(x)·[ L(P(x), y)  +  µ·L(D(y), x)  +  σ·L(D(P(x)), D(y)) ]

    Parameters
    ----------
    lam   : weight for unsupervised cycle-consistency loss
    mu    : weight for dual regression  D(y) ≈ x
    sigma : weight for dual consistency D(P(x)) ≈ D(y)
    """
    def __init__(self,
                 lam:   float = 1.0,
                 mu:    float = 0.5,
                 sigma: float = 0.5,
                 alpha_init: float = 0.9):
        super().__init__()
        self.lam   = lam
        self.mu    = mu
        self.sigma = sigma
        self.L     = CompositeLoss(alpha_init)

    def forward(self,
                P_model,
                D_model,
                x: torch.Tensor,               # LR satellite  (B,1,50,50)
                y: torch.Tensor | None = None,  # HR shipborne  (B,1,200,200)
                ) -> dict:
        """
        Returns dict with keys: total, cycle, recon, dual_reg, dual_cons.
        Pass y=None for fully unsupervised (unpaired) samples.
        """
        y_hat = P_model(x)          # predicted HR
        x_hat = D_model(y_hat)      # cycle-reconstructed LR

        # ── cycle loss (always computed) ─────────────────────────────────
        cycle_loss = self.L(x_hat, x)

        recon_loss = dual_reg_loss = dual_cons_loss = \
            torch.zeros(1, device=x.device, requires_grad=False)

        if y is not None:
            x_tld = D_model(y)           # D applied to ground-truth HR

            # reconstruction  P(x) ≈ y
            recon_loss     = self.L(y_hat, y)
            # dual regression  D(y) ≈ x
            dual_reg_loss  = self.L(x_tld, x)
            # dual consistency  D(P(x)) ≈ D(y)
            dual_cons_loss = self.L(x_hat, x_tld.detach())

        total = (self.lam * cycle_loss
                 + recon_loss
                 + self.mu    * dual_reg_loss
                 + self.sigma * dual_cons_loss)

        return {
            'total':     total,
            'cycle':     cycle_loss.detach(),
            'recon':     recon_loss.detach(),
            'dual_reg':  dual_reg_loss.detach(),
            'dual_cons': dual_cons_loss.detach(),
            'alpha':     self.L.alpha.detach(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Evaluation metrics  (no gradients)
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_metrics(pred: torch.Tensor,
                    gt:   torch.Tensor,
                    data_range: float = 1.0) -> dict:
    """
    Returns PSNR, SSIM, MSE, MAE, MRE for a batch.
    All tensors: (B, 1, H, W) in [0, 1].
    """
    mse  = torch.mean((pred - gt) ** 2).item()
    mae  = torch.mean(torch.abs(pred - gt)).item()
    mre  = torch.mean(torch.abs(pred - gt) / (torch.abs(gt) + 1e-6)).item()
    psnr = 10 * torch.log10(torch.tensor(data_range ** 2 / (mse + 1e-10))).item()
    # signal power
    sig_power = torch.mean(gt ** 2).item()
    snr  = 10 * torch.log10(torch.tensor(sig_power / (mse + 1e-10))).item()
    ssim_v = pt_ssim(pred, gt, data_range=data_range,
                     size_average=True).item()
    return dict(psnr=psnr, snr=snr, ssim=ssim_v,
                mse=mse, mae=mae, mre=mre)


# ════════════════════════════════════════════════════════════════════════════
# Quick test
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sdrl.model import build_models

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P, D   = build_models(device=device)
    criterion = SDRLLoss().to(device)

    x = torch.rand(2, 1,  50,  50, device=device)
    y = torch.rand(2, 1, 200, 200, device=device)

    # paired forward
    losses = criterion(P, D, x, y)
    losses['total'].backward()
    print("Paired loss:", {k: f"{v.item():.4f}"
                           for k, v in losses.items() if k != 'alpha'})
    print(f"α = {losses['alpha'].item():.4f}")

    # unpaired forward
    P.zero_grad(); D.zero_grad()
    losses2 = criterion(P, D, x, None)
    losses2['total'].backward()
    print("Unpaired loss:", {k: f"{v.item():.4f}"
                             for k, v in losses2.items() if k != 'alpha'})
    print("Loss module test passed ✓")
