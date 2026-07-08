"""
model.py  (v4 — shortcut slices gravity channel instead of averaging)
-------------------------------------------------------
SDRL network architecture for gravity super-resolution.
Based on: Remote Sens. 2026, 18, 453  (Section 3.4)

Fixes vs v3:
  - shortcut_proj now SLICES channel 0 (gravity) instead of averaging all
    input channels together. The shortcut path exists specifically to
    give the decoder a strong, learnable-but-simple prior: "gravity,
    bilinearly upsampled." Averaging gravity together with GEBCO
    bathymetry, NZ bathymetry, and EGM2008 — three fields with
    completely different physical meaning and spatial statistics —
    diluted that prior and likely made the residual path meaningfully
    less useful than in the original 1-channel model, forcing the
    decoder to compensate for a weaker starting point. Slicing
    reproduces exactly the single-channel model's shortcut behavior
    regardless of how many auxiliary channels are present. This is a
    pure indexing operation (not a learned layer), so there's no new
    randomly-initialized weight on this path at all — one less thing
    to train from scratch.
  - shortcut_proj is now a lightweight callable (a channel-slicing
    module) rather than a Conv2d, for in_channels > 1. Still
    nn.Identity() when in_channels == 1, so single-channel behavior is
    completely unchanged.

Fixes vs v2->v3 (multi-channel input support, superseded above):
  - PrimaryNet accepted `in_channels` (default 1, backward compatible)
  - stem's first conv uses in_channels instead of hardcoded 1
  - build_models() accepts and forwards in_channels
  - DualNet is UNCHANGED and stays fixed at 1 input/output channel by
    design: it maps HR gravity -> LR gravity for the cycle-consistency
    term and was never meant to touch the auxiliary channels.

Fixes carried over from v1->v2:
  - Shortcut scaled by learnable 0.1 weight so decoder is forced to
    actually learn rather than collapsing to pure bilinear upsampling
  - BatchNorm added to stem to normalise input distribution
  - Decoder n_rcab increased from 4→6 for stronger feature learning
  - PixelShuffle init with ICNR to reduce checkerboard artefacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ════════════════════════════════════════════════════════════════════════════
# ICNR initialisation for PixelShuffle (reduces checkerboard)
# ════════════════════════════════════════════════════════════════════════════
def icnr_init(tensor: torch.Tensor, scale: int = 4):
    """ICNR init: each sub-pixel gets the same initialisation."""
    out_ch, in_ch, kH, kW = tensor.shape
    sub = out_ch // (scale * scale)
    kernel = torch.zeros(sub, in_ch, kH, kW)
    nn.init.kaiming_normal_(kernel)
    kernel = kernel.repeat(scale * scale, 1, 1, 1)
    with torch.no_grad():
        tensor.copy_(kernel)


# ════════════════════════════════════════════════════════════════════════════
# Lite Multi-scale Linear Attention  (LiteMLA)
# ════════════════════════════════════════════════════════════════════════════
class LiteMLA(nn.Module):
    def __init__(self, channels: int, heads: int = 4,
                 kernel_sizes=(3, 5, 7)):
        super().__init__()
        # ensure heads divides channels
        while channels % heads != 0 and heads > 1:
            heads //= 2
        self.heads = heads

        self.qkv  = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels,
                      k, padding=k // 2, groups=channels, bias=False)
            for k in kernel_sizes
        ])
        self.key_mix = nn.Conv2d(channels * len(kernel_sizes),
                                 channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=1)

        K_ms = torch.cat([dw(K) for dw in self.dw_convs], dim=1)
        K    = self.key_mix(K_ms)

        Q = F.relu(Q) + 1e-6
        K = F.relu(K) + 1e-6

        N = H * W
        d = max(1, C // self.heads)
        Q = Q.reshape(B, self.heads, d, N)
        K = K.reshape(B, self.heads, d, N)
        V = V.reshape(B, self.heads, d, N)

        KV  = torch.einsum('bhdN,bhvN->bhdv', K, V)
        out = torch.einsum('bhdN,bhdv->bhvN', Q, KV)
        K_sum = K.sum(dim=-1, keepdim=True)
        Z     = (Q * K_sum).sum(dim=2, keepdim=True) + 1e-6
        out   = out / Z

        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ════════════════════════════════════════════════════════════════════════════
# Residual Channel Attention Block  (RCAB)
# ════════════════════════════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))


class RCAB(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            ChannelAttention(channels, reduction)
        )

    def forward(self, x):
        return x + self.body(x)


# ════════════════════════════════════════════════════════════════════════════
# Encoder block
# ════════════════════════════════════════════════════════════════════════════
class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attn: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn     = LiteMLA(out_ch) if use_attn else nn.Identity()
        self.downsamp = nn.Conv2d(out_ch, out_ch, 3, stride=2,
                                  padding=1, bias=False)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.attn(feat)
        skip = feat
        down = self.downsamp(feat)
        return down, skip


# ════════════════════════════════════════════════════════════════════════════
# Decoder block — F.interpolate for exact skip alignment
# ════════════════════════════════════════════════════════════════════════════
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 n_rcab: int = 6):
        super().__init__()
        self.fuse = nn.Conv2d(in_ch + skip_ch, in_ch, 1, bias=False)
        self.rcab = nn.Sequential(*[RCAB(in_ch) for _ in range(n_rcab)])
        self.up   = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.rcab(x)
        return self.up(x)


# ════════════════════════════════════════════════════════════════════════════
# Channel-slicing shortcut projection
# ════════════════════════════════════════════════════════════════════════════
class GravityChannelSlice(nn.Module):
    """
    Extracts channel 0 (gravity) from a multi-channel input. Used in place
    of a learned projection for the shortcut/residual path, so that path
    reproduces exactly what the original single-channel model's shortcut
    did (a bilinear upsample of the raw gravity field), regardless of how
    many auxiliary channels are present in the full input.
    """
    def forward(self, x):
        return x[:, :1]


# ════════════════════════════════════════════════════════════════════════════
# Primary Network  P : LR → HR   (×4 super-resolution)
# ════════════════════════════════════════════════════════════════════════════
class PrimaryNet(nn.Module):
    """
    Input : (B, in_channels, 50,  50)
    Output: (B, 1, 200, 200)

    `in_channels` defaults to 1 for full backward compatibility. When
    in_channels > 1, the auxiliary channels are used by the encoder /
    bottleneck / decoder as before (so the model can still learn to
    combine them), but the shortcut residual path now uses ONLY the
    gravity channel (channel 0) — see GravityChannelSlice — reproducing
    the original single-channel model's strong "bilinear-upsampled
    gravity" prior rather than diluting it by averaging across
    physically unrelated auxiliary channels.

    Key fix (v2): shortcut is scaled by learnable weight init=0.1
    so the decoder MUST learn meaningful residuals rather than
    collapsing to pure bilinear upsampling (blank output bug).
    """
    def __init__(self, base_ch: int = 32, in_channels: int = 1):
        super().__init__()
        c = base_ch
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc1 = EncoderBlock(c,     c * 2)
        self.enc2 = EncoderBlock(c * 2, c * 4)
        self.enc3 = EncoderBlock(c * 4, c * 8, use_attn=False)

        self.bottleneck = nn.Sequential(
            LiteMLA(c * 8),
            RCAB(c * 8),
            RCAB(c * 8),
            RCAB(c * 8),
        )

        # dec(in_ch, skip_ch, out_ch)
        self.dec3 = DecoderBlock(c * 8, c * 8, c * 4, n_rcab=6)
        self.dec2 = DecoderBlock(c * 4, c * 4, c * 2, n_rcab=6)
        self.dec1 = DecoderBlock(c * 2, c * 2, c,     n_rcab=4)

        # ×4 upsample with ICNR init — always produces 1 output channel
        ps_conv = nn.Conv2d(c, 1 * 16, 3, padding=1, bias=False)
        icnr_init(ps_conv.weight, scale=4)
        self.final_up = nn.Sequential(
            RCAB(c),
            ps_conv,
            nn.PixelShuffle(4),
        )

        # Shortcut: extract gravity channel (channel 0) BEFORE bilinear
        # upsampling. Identity when in_channels==1 (no-op slice — a
        # single-channel input's "channel 0" is the whole input), so
        # single-channel behavior is completely unchanged from v2/v3.
        self.shortcut_proj = (
            GravityChannelSlice() if in_channels != 1 else nn.Identity()
        )

        self.shortcut       = nn.Upsample(scale_factor=4, mode='bilinear',
                                          align_corners=False)
        # Shortcut scaled by small learnable weight (init=0.1)
        # This FORCES the decoder to contribute rather than being ignored
        self.shortcut_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        s  = self.stem(x)
        d1, sk1 = self.enc1(s)
        d2, sk2 = self.enc2(d1)
        d3, sk3 = self.enc3(d2)
        bn = self.bottleneck(d3)
        u3 = self.dec3(bn, sk3)
        u2 = self.dec2(u3, sk2)
        u1 = self.dec1(u2, sk1)
        hr = self.final_up(u1)
        # scaled shortcut — model must learn to produce non-trivial residuals
        shortcut_1ch = self.shortcut_proj(x)
        return hr + self.shortcut_scale * self.shortcut(shortcut_1ch)


# ════════════════════════════════════════════════════════════════════════════
# Dual Network  D : HR → LR   (×4 downscaling)
# ════════════════════════════════════════════════════════════════════════════
class DualNet(nn.Module):
    """
    Input : (B, 1, 200, 200)
    Output: (B, 1, 50,  50)

    UNCHANGED, and deliberately stays fixed at 1 in/out channel. This
    network's role is the cycle-consistency term: map HR gravity back
    down to LR gravity. It is not meant to reconstruct the auxiliary
    channels — the loss function compares its output only against the
    gravity channel of the LR input (x[:, :1]). See loss.py.
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.enc = nn.Sequential(
            nn.Conv2d(1,     c,     3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            LiteMLA(c),
            nn.Conv2d(c,     c * 2, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.body = nn.Sequential(
            RCAB(c * 4),
            RCAB(c * 4),
            LiteMLA(c * 4),
        )
        self.degrade = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, 1,     3, padding=1, bias=False),
        )
        self.shortcut = nn.AvgPool2d(4, stride=4)

    def forward(self, y):
        feat = self.enc(y)
        feat = self.body(feat)
        return self.degrade(feat) + self.shortcut(y)


# ════════════════════════════════════════════════════════════════════════════
# Builder
# ════════════════════════════════════════════════════════════════════════════
def build_models(base_ch: int = 32, in_channels: int = 1, device: str = 'cuda'):
    """
    in_channels: number of input channels for PrimaryNet (e.g. 4 for
    gravity + GEBCO + NZ Bathymetry + EGM2008). Defaults to 1 for full
    backward compatibility. DualNet is always 1-channel by design.
    """
    P = PrimaryNet(base_ch, in_channels=in_channels).to(device)
    D = DualNet(base_ch).to(device)
    n_P = sum(p.numel() for p in P.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"PrimaryNet : {n_P:,} params  (base_ch={base_ch}, in_channels={in_channels})")
    print(f"DualNet    : {n_D:,} params  (in_channels=1, fixed)")
    return P, D


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # single-channel backward-compatibility check
    P, D = build_models(base_ch=32, in_channels=1, device=device)
    lr = torch.randn(2, 1, 50,  50, device=device)
    hr = torch.randn(2, 1, 200, 200, device=device)
    y_hat = P(lr)
    x_hat = D(y_hat)
    print(f"[in_channels=1] P(lr)    → {y_hat.shape}  expected (2,1,200,200)")
    print(f"[in_channels=1] D(P(lr)) → {x_hat.shape}  expected (2,1,50,50)")
    print(f"shortcut_scale = {P.shortcut_scale.item():.3f}")
    assert isinstance(P.shortcut_proj, torch.nn.Identity), \
        "in_channels=1 should use Identity for shortcut_proj"

    # multi-channel check — verify shortcut actually uses channel 0 only
    P4, D4 = build_models(base_ch=32, in_channels=4, device=device)
    lr4 = torch.randn(2, 4, 50, 50, device=device)
    y_hat4 = P4(lr4)
    x_hat4 = D4(y_hat4)
    print(f"[in_channels=4] P(lr)    → {y_hat4.shape}  expected (2,1,200,200)")
    print(f"[in_channels=4] D(P(lr)) → {x_hat4.shape}  expected (2,1,50,50)")
    assert isinstance(P4.shortcut_proj, GravityChannelSlice), \
        "in_channels=4 should use GravityChannelSlice for shortcut_proj"
    sliced = P4.shortcut_proj(lr4)
    assert sliced.shape[1] == 1 and torch.equal(sliced, lr4[:, :1]), \
        "shortcut_proj should extract exactly channel 0"
    print("shortcut_proj correctly slices channel 0 (verified)")

    print("✓ Model check passed")
