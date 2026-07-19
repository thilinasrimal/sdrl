"""
model.py  (v5 — ablation-ready)
-------------------------------------------------------
SDRL / NZGravNet network architecture for gravity super-resolution.
Based on: Remote Sens. 2026, 18, 453 (Jia et al.), adapted for NZ EEZ.

New in this version (ablation support):
  - PrimaryNet(disable_shortcut=True) implements Test B1: removes the
    residual shortcut path entirely (returns hr directly), to test
    whether the decoder is generating genuine new high-frequency content
    or the model's apparent quality is coming mostly from the bilinear-
    upsampled gravity prior added at the end.
  - PrimaryNet(rcab_depth_scale=0.5) implements Test C2: halves the
    number of RCAB blocks in the bottleneck and each decoder stage
    (rounded up, minimum 1), to test whether the current depth is
    actually earning its computational cost.
  - base_ch already existed and directly implements Test C1 (width
    ablation) — no change needed, just pass a smaller base_ch (e.g. 16
    or 24 instead of 48) when building the model.
  - Channel ablations (A1/A2/A3) do NOT require changes here — they are
    implemented by zeroing input channels before the batch reaches the
    model (see the training cell), since in_channels stays fixed at 4
    for a fair architecture comparison.

All fixes from v4 are preserved unchanged:
  - in_channels support in PrimaryNet / build_models
  - shortcut path restricted to the gravity channel only (not an
    average across all 4 channels), via GravityChannelSlice
  - DualNet unchanged, fixed at 1 channel by design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ════════════════════════════════════════════════════════════════════════════
# ICNR initialisation for PixelShuffle (reduces checkerboard)
# ════════════════════════════════════════════════════════════════════════════
def icnr_init(tensor: torch.Tensor, scale: int = 4):
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
        n_rcab = max(1, n_rcab)  # never allow zero blocks
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
    """Extracts channel 0 (gravity) from a multi-channel input, so the
    shortcut/residual path reproduces the single-channel model's
    behaviour (bilinear-upsampled gravity) regardless of auxiliary
    channel count."""
    def forward(self, x):
        return x[:, :1]


# ════════════════════════════════════════════════════════════════════════════
# Primary Network  P : LR → HR   (×4 super-resolution)
# ════════════════════════════════════════════════════════════════════════════
class PrimaryNet(nn.Module):
    """
    Input : (B, in_channels, 50,  50)
    Output: (B, 1, 200, 200)

    disable_shortcut (Test B1): if True, the residual shortcut path is
    removed entirely and forward() returns the decoder's output (hr)
    directly, with no added bilinear-upsampled gravity term. Used to
    test whether the network's internal layers are generating genuine
    new high-frequency content, or whether apparent quality is coming
    mostly from the shortcut's low-frequency prior.

    rcab_depth_scale (Test C2): scales the number of RCAB blocks in the
    bottleneck and each decoder stage relative to the default depth
    (bottleneck 3, dec3 6, dec2 6, dec1 4), rounded up with a minimum of
    1 block per stage. rcab_depth_scale=0.5 approximately halves depth.
    rcab_depth_scale=1.0 (default) reproduces the original architecture
    exactly.
    """
    def __init__(self, base_ch: int = 32, in_channels: int = 1,
                 disable_shortcut: bool = False,
                 rcab_depth_scale: float = 1.0):
        super().__init__()
        c = base_ch
        self.in_channels = in_channels
        self.disable_shortcut = disable_shortcut

        def scaled(n):
            return max(1, math.ceil(n * rcab_depth_scale))

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
            *[RCAB(c * 8) for _ in range(scaled(3))],
        )

        self.dec3 = DecoderBlock(c * 8, c * 8, c * 4, n_rcab=scaled(6))
        self.dec2 = DecoderBlock(c * 4, c * 4, c * 2, n_rcab=scaled(6))
        self.dec1 = DecoderBlock(c * 2, c * 2, c,     n_rcab=scaled(4))

        ps_conv = nn.Conv2d(c, 1 * 16, 3, padding=1, bias=False)
        icnr_init(ps_conv.weight, scale=4)
        self.final_up = nn.Sequential(
            RCAB(c),
            ps_conv,
            nn.PixelShuffle(4),
        )

        if not disable_shortcut:
            self.shortcut_proj = (
                GravityChannelSlice() if in_channels != 1 else nn.Identity()
            )
            self.shortcut = nn.Upsample(scale_factor=4, mode='bilinear',
                                        align_corners=False)
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

        if self.disable_shortcut:
            # Test B1: no residual shortcut — output is whatever the
            # decoder produces on its own, with no added low-frequency
            # gravity prior.
            return hr

        shortcut_1ch = self.shortcut_proj(x)
        return hr + self.shortcut_scale * self.shortcut(shortcut_1ch)


# ════════════════════════════════════════════════════════════════════════════
# Dual Network  D : HR → LR   (×4 downscaling)
# ════════════════════════════════════════════════════════════════════════════
class DualNet(nn.Module):
    """
    Input : (B, 1, 200, 200)
    Output: (B, 1, 50,  50)

    Unchanged across all ablations — always 1-channel, always the same
    depth. Test B2 (no dual-regression loss) is implemented by setting
    mu=0 in SDRLLoss, not by modifying this network.
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
def build_models(base_ch: int = 32, in_channels: int = 1, device: str = 'cuda',
                  disable_shortcut: bool = False, rcab_depth_scale: float = 1.0):
    """
    base_ch          : Test C1 (width) — pass e.g. 16 or 24 for a
                        lightweight variant instead of the default 48.
    disable_shortcut : Test B1 — True removes the residual shortcut path.
    rcab_depth_scale : Test C2 — 0.5 approximately halves RCAB depth.
    """
    P = PrimaryNet(base_ch, in_channels=in_channels,
                   disable_shortcut=disable_shortcut,
                   rcab_depth_scale=rcab_depth_scale).to(device)
    D = DualNet(base_ch).to(device)
    n_P = sum(p.numel() for p in P.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"PrimaryNet : {n_P:,} params  (base_ch={base_ch}, in_channels={in_channels}, "
          f"disable_shortcut={disable_shortcut}, rcab_depth_scale={rcab_depth_scale})")
    print(f"DualNet    : {n_D:,} params  (in_channels=1, fixed)")
    return P, D
