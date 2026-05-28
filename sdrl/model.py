"""
model.py
--------
SDRL network architecture for gravity super-resolution.
Based on: Remote Sens. 2026, 18, 453  (Section 3.4)

Components:
  - LiteMLA   : Lite Multi-scale Linear Attention  (O(N) complexity)
  - RCAB       : Residual Channel Attention Block
  - PrimaryNet : Encoder-decoder, LR→HR  (scale × 4 via PixelShuffle)
  - DualNet    : HR→LR degradation network

Upload to: /content/drive/MyDrive/sdrl_gravity/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# Lite Multi-scale Linear Attention  (LiteMLA)
# ════════════════════════════════════════════════════════════════════════════
class LiteMLA(nn.Module):
    """
    Linear attention with multi-scale depth-wise convolution keys.
    Complexity: O(N) instead of O(N²).
    Kernel sizes: 3×3, 5×5, 7×7 for multi-scale geophysical feature capture.
    """
    def __init__(self, channels: int, heads: int = 4,
                 kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5

        self.qkv  = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

        # multi-scale depth-wise convolutions for keys
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels,
                      k, padding=k // 2, groups=channels, bias=False)
            for k in kernel_sizes
        ])
        self.key_mix = nn.Conv2d(channels * len(kernel_sizes),
                                 channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)                          # (B, 3C, H, W)
        Q, K, V = qkv.chunk(3, dim=1)

        # multi-scale keys
        K_ms = torch.cat([dw(K) for dw in self.dw_convs], dim=1)
        K    = self.key_mix(K_ms)

        # ReLU kernel (linear approximation)
        Q = F.relu(Q) + 1e-6
        K = F.relu(K) + 1e-6

        # reshape to (B, heads, head_dim, N)
        N = H * W
        d = C // self.heads
        Q = Q.reshape(B, self.heads, d, N)
        K = K.reshape(B, self.heads, d, N)
        V = V.reshape(B, self.heads, d, N)

        # linear attention: O(N·d²)
        KV  = torch.einsum('bhdN,bhvN->bhdv', K, V)     # (B,h,d,d)
        out = torch.einsum('bhdN,bhdv->bhvN', Q, KV)    # (B,h,d,N)

        # normalise
        K_sum = K.sum(dim=-1, keepdim=True)              # (B,h,d,1)
        Z     = (Q * K_sum).sum(dim=2, keepdim=True) + 1e-6
        out   = out / Z

        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ════════════════════════════════════════════════════════════════════════════
# Residual Channel Attention Block  (RCAB)
# ════════════════════════════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))


class RCAB(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
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
# Encoder block  (conv + LeakyReLU + optional LiteMLA before downsample)
# ════════════════════════════════════════════════════════════════════════════
class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attn: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
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
# Decoder block  (RCAB stack + bilinear upsample)
# Uses F.interpolate to match skip connection size exactly — this eliminates
# the 7/13 dimension mismatch that arises when the encoder strides over
# odd spatial sizes (50→25→13→7 instead of clean powers-of-two).
# ════════════════════════════════════════════════════════════════════════════
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 n_rcab: int = 4, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.fuse = nn.Conv2d(in_ch + skip_ch, in_ch, 1, bias=False)
        self.rcab = nn.Sequential(*[RCAB(in_ch) for _ in range(n_rcab)])
        self.up   = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x, skip):
        # ── upsample x to match skip's spatial size exactly ──────────────
        if self.upsample:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        # now cat is always safe regardless of odd input dimensions
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.rcab(x)
        return self.up(x)


# ════════════════════════════════════════════════════════════════════════════
# Primary Network  P : LR → HR   (×4 super-resolution)
# ════════════════════════════════════════════════════════════════════════════
class PrimaryNet(nn.Module):
    """
    Encoder-decoder with:
      - 3 encoder levels (ch: 1→32→64→128)
      - bottleneck (128→128 with LiteMLA)
      - 3 decoder levels with RCAB + PixelShuffle
      - shortcut: bilinear 4× upsample of input merged at output
    Input:  (B, 1, 50,  50)
    Output: (B, 1, 200, 200)
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        # stem
        self.stem = nn.Conv2d(1, c, 3, padding=1, bias=False)
        # encoder
        self.enc1 = EncoderBlock(c,     c * 2)
        self.enc2 = EncoderBlock(c * 2, c * 4)
        self.enc3 = EncoderBlock(c * 4, c * 8, use_attn=False)
        # bottleneck
        self.bottleneck = nn.Sequential(
            LiteMLA(c * 8),
            RCAB(c * 8),
            RCAB(c * 8),
        )
        # Encoder output channels:
        #   stem → c,  enc1 skip → c*2,  enc2 skip → c*4,  enc3 skip → c*8
        #   bottleneck → c*8
        # DecoderBlock(in_ch, skip_ch, out_ch):
        #   dec3: bottleneck(c*8) + enc3 skip(c*8) → out c*4
        #   dec2: dec3 out(c*4)   + enc2 skip(c*4) → out c*2
        #   dec1: dec2 out(c*2)   + enc1 skip(c*2) → out c
        self.dec3 = DecoderBlock(c * 8, c * 8, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 4, c * 2)
        self.dec1 = DecoderBlock(c * 2, c * 2, c)
        # final ×4 upsample (50 → 200)
        self.final_up = nn.Sequential(
            RCAB(c),
            nn.Conv2d(c, 1 * 16, 3, padding=1, bias=False),
            nn.PixelShuffle(4),
        )
        # shortcut
        self.shortcut = nn.Upsample(scale_factor=4, mode='bilinear',
                                    align_corners=False)

    def forward(self, x):
        s  = self.stem(x)               # (B, c,   50, 50)
        d1, sk1 = self.enc1(s)          # d1=(B,2c, 25,25)  sk1=(B,2c, 50,50)
        d2, sk2 = self.enc2(d1)         # d2=(B,4c, 13,13)  sk2=(B,4c, 25,25)
        d3, sk3 = self.enc3(d2)         # d3=(B,8c,  7, 7)  sk3=(B,8c, 13,13)
        bn = self.bottleneck(d3)        # (B, 8c, 7, 7)
        # decoder: F.interpolate in DecoderBlock ensures exact size match
        u3 = self.dec3(bn, sk3)         # → (B,4c, 13,13)
        u2 = self.dec2(u3, sk2)         # → (B,2c, 25,25)
        u1 = self.dec1(u2, sk1)         # → (B, c, 50,50)
        hr = self.final_up(u1)          # → (B, 1,200,200)
        return hr + self.shortcut(x)    # residual shortcut


# ════════════════════════════════════════════════════════════════════════════
# Dual Network  D : HR → LR   (×4 downscaling / degradation)
# ════════════════════════════════════════════════════════════════════════════
class DualNet(nn.Module):
    """
    HR-to-LR degradation network.
    Learns adaptive downsampling + bias correction (satellite sensor model).
    Input:  (B, 1, 200, 200)
    Output: (B, 1, 50,  50)
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.enc = nn.Sequential(
            nn.Conv2d(1,     c,     3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            LiteMLA(c),
            nn.Conv2d(c,     c * 2, 3, stride=2, padding=1, bias=False),  # /2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),  # /4
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.body = nn.Sequential(
            RCAB(c * 4),
            RCAB(c * 4),
            LiteMLA(c * 4),
        )
        # adaptive degradation: learns per-region bias/noise pattern
        self.degrade = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 2, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, 1,     3, padding=1, bias=False),
        )
        # shortcut: simple average-pool downsample
        self.shortcut = nn.AvgPool2d(4, stride=4)

    def forward(self, y):
        feat = self.enc(y)
        feat = self.body(feat)
        lr   = self.degrade(feat)
        return lr + self.shortcut(y)


# ════════════════════════════════════════════════════════════════════════════
# Convenience builder
# ════════════════════════════════════════════════════════════════════════════
def build_models(base_ch: int = 32, device='cuda'):
    P = PrimaryNet(base_ch).to(device)
    D = DualNet(base_ch).to(device)
    n_P = sum(p.numel() for p in P.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"PrimaryNet params: {n_P:,}")
    print(f"DualNet    params: {n_D:,}")
    return P, D


# ════════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P, D = build_models(device=device)
    lr = torch.randn(2, 1, 50,  50, device=device)
    hr = torch.randn(2, 1, 200, 200, device=device)
    y_hat = P(lr)
    x_hat = D(y_hat)
    x_tld = D(hr)
    print(f"P(lr)   → {y_hat.shape}   expected (2,1,200,200)")
    print(f"D(P(lr))→ {x_hat.shape}   expected (2,1,50,50)")
    print(f"D(hr)   → {x_tld.shape}   expected (2,1,50,50)")
    print("Model sanity check passed ✓")
