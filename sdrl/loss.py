"""
loss.py  (v3 — multi-channel input support)
------------------------------------------
Loss functions for SDRL gravity super-resolution.
Based on: Remote Sens. 2026, 18, 453  (Equations 3 and 5)

Fixes vs v2:
  - cycle_loss and dual_reg_loss now compare against x[:, :1] (the
    gravity channel only) instead of the full x. This matters when x
    has more than 1 channel (e.g. gravity + GEBCO + NZ Bathymetry +
    EGM2008): DualNet (model.py) always outputs exactly 1 channel — it
    was never meant to reconstruct the auxiliary channels — so
    comparing its output against the full multi-channel x raised a
    shape mismatch (and would have been conceptually wrong even if the
    shapes had happened to match). dual_cons_loss is unaffected since
    both sides were already 1-channel.
  - When x has exactly 1 channel, x[:, :1] is a no-op slice, so this
    change is fully backward compatible with the single-channel
    pipeline.

Fixes carried over from v1->v2:
  - alpha is FIXED at 0.84 (not learnable) — eliminates collapse to 0
  - MRE-aware loss: adds gradient penalty to push model off mean prediction
  - recon loss weighted 3x higher than cycle to prioritise supervised signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim as pt_ssim


# ════════════════════════════════════════════════════════════════════════════
# Composite loss  L(a, b)  — Eq. (3)  with FIXED alpha
# ════════════════════════════════════════════════════════════════════════════
class CompositeLoss(nn.Module):
    """
    L(a, b) = (1-alpha)*L1 + alpha*(1-SSIM)

    alpha is FIXED (not learnable) to prevent collapse.
    Default alpha=0.84 gives strong SSIM guidance.
    """
    def __init__(self, alpha: float = 0.84):
        super().__init__()
        # Fixed — not a Parameter, not learnable, cannot collapse
        self.register_buffer('_alpha', torch.tensor(alpha))

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape == b.shape, (
            f"CompositeLoss received mismatched shapes: {tuple(a.shape)} vs "
            f"{tuple(b.shape)}. If one side comes from DualNet (always "
            f"1-channel) and the other is a multi-channel LR tensor, make "
            f"sure you're slicing to the gravity channel, e.g. x[:, :1], "
            f"before calling this loss."
        )
        α    = self._alpha
        l1   = torch.mean(torch.abs(a - b))
        ssim_val = pt_ssim(
            a.clamp(0, 1), b.clamp(0, 1),
            data_range=1.0, size_average=True)
        return (1.0 - α) * l1 + α * (1.0 - ssim_val)


# ════════════════════════════════════════════════════════════════════════════
# Gradient / detail loss — penalises over-smoothing
# ════════════════════════════════════════════════════════════════════════════
def _sobel_grad(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude."""
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                       dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = kx.transpose(-2,-1)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 loss on Sobel gradients — forces sharp edges."""
    return torch.mean(torch.abs(_sobel_grad(pred) - _sobel_grad(gt)))


# ════════════════════════════════════════════════════════════════════════════
# SDRL semi-supervised loss  L_semi  — Eq. (5)  with fixes
# ════════════════════════════════════════════════════════════════════════════
class SDRLLoss(nn.Module):
    """
    L_semi = lam*L(D(P(x)), x_grav)                      ← cycle (unsupervised)
           + I_M(x)*[
               3*L(P(x), y)                              ← recon (main signal)
             + mu*L(D(y), x_grav)                         ← dual regression
             + sigma*L(D(P(x)), D(y))                    ← dual consistency
             + grad_w*gradient_loss(P(x), y)             ← anti-blur
           ]

    where x_grav = x[:, :1] — DualNet always outputs a single channel
    (it maps HR gravity -> LR gravity), so every loss term that compares
    against a DualNet output must compare against the gravity channel of
    x specifically, not the full multi-channel LR tensor. This is a
    no-op when x already has exactly 1 channel.

    Key changes vs v1:
      - alpha fixed at 0.84 (no collapse)
      - recon weighted 3x
      - gradient loss added to fight blank/blurry outputs
    Key change vs v2:
      - cycle/dual_reg losses correctly sliced to the gravity channel
        for multi-channel inputs
    """
    def __init__(self,
                 lam:    float = 0.5,
                 mu:     float = 0.3,
                 sigma:  float = 0.3,
                 grad_w: float = 0.5,
                 alpha:  float = 0.84):
        super().__init__()
        self.lam    = lam
        self.mu     = mu
        self.sigma  = sigma
        self.grad_w = grad_w
        self.L      = CompositeLoss(alpha)

    def forward(self,
                P_model,
                D_model,
                x: torch.Tensor,
                y: torch.Tensor | None = None) -> dict:

        # DualNet's target/reference channel — always channel 0 (gravity),
        # regardless of how many channels x has. No-op slice if x is
        # already single-channel.
        x_grav = x[:, :1]

        y_hat = P_model(x)           # predicted HR  (B,1,200,200)
        x_hat = D_model(y_hat)       # cycle LR      (B,1,50,50)

        # clamp outputs to valid range to avoid SSIM errors
        y_hat_c = y_hat.clamp(0, 1)

        # cycle loss — always computed (unsupervised)
        cycle_loss = self.L(x_hat, x_grav)

        recon_loss = dual_reg_loss = dual_cons_loss = grad_loss = \
            torch.zeros(1, device=x.device)

        if y is not None:
            x_tld = D_model(y)           # D(HR ground truth), (B,1,50,50)

            recon_loss     = self.L(y_hat_c, y)
            dual_reg_loss  = self.L(x_tld, x_grav)
            dual_cons_loss = self.L(x_hat, x_tld.detach())
            grad_loss      = gradient_loss(y_hat_c, y)

        total = (self.lam  * cycle_loss
                 + 3.0     * recon_loss          # 3x weight on main signal
                 + self.mu * dual_reg_loss
                 + self.sigma * dual_cons_loss
                 + self.grad_w * grad_loss)

        return {
            'total':     total,
            'cycle':     cycle_loss.detach(),
            'recon':     recon_loss.detach(),
            'dual_reg':  dual_reg_loss.detach(),
            'grad':      grad_loss.detach(),
            'alpha':     self.L.alpha.detach(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_metrics(pred: torch.Tensor,
                    gt:   torch.Tensor,
                    data_range: float = 1.0) -> dict:
    assert pred.shape == gt.shape, (
        f"compute_metrics received mismatched shapes: {tuple(pred.shape)} "
        f"vs {tuple(gt.shape)}."
    )
    pred_c = pred.clamp(0, 1)
    gt_c   = gt.clamp(0, 1)
    mse    = torch.mean((pred_c - gt_c) ** 2).item()
    mae    = torch.mean(torch.abs(pred_c - gt_c)).item()
    mre    = torch.mean(
        torch.abs(pred_c - gt_c) / (torch.abs(gt_c) + 1e-6)).item()
    psnr   = 10 * torch.log10(
        torch.tensor(data_range**2 / (mse + 1e-10))).item()
    sig_power = torch.mean(gt_c**2).item()
    snr    = 10 * torch.log10(
        torch.tensor(sig_power / (mse + 1e-10))).item()
    ssim_v = pt_ssim(pred_c, gt_c,
                     data_range=data_range, size_average=True).item()
    return dict(psnr=psnr, snr=snr, ssim=ssim_v,
                mse=mse, mae=mae, mre=mre)
