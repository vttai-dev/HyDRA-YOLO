"""CSDN-style YOLO26 variants used with DynamicConv (Dyn) baseline.

Lightweight, pickleable modules for overnight/smoke search.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SPDConv",
    "CARAFE",
    "DySampleLite",
    "ECA",
    "CoordAtt",
    "EMA",
    "SimAM",
    "ASFFLevel",
    "BiFPNAdd",
    "C3k2_DynDCN",
    "C3k2_ODConv",
]


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPDConv(nn.Module):
    """Space-to-depth then 3x3 conv (preserves detail vs stride-2)."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, 3, 1)

    def forward(self, x):
        # (B,C,H,W) -> (B,4C,H/2,W/2)
        x = torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]],
            dim=1,
        )
        return self.conv(x)


class CARAFE(nn.Module):
    """Simplified CARAFE upsampling (content-aware kernels)."""

    def __init__(self, c1, scale=2, k_up=5, k_enc=3, cm=64):
        super().__init__()
        self.scale = int(scale)
        self.k = k_up
        self.compress = Conv(c1, cm, 1)
        self.encoder = nn.Conv2d(cm, (self.scale * k_up) ** 2, k_enc, 1, k_enc // 2)

    def forward(self, x):
        b, c, h, w = x.shape
        s, k = self.scale, self.k
        mask = self.encoder(self.compress(x))  # B, (s*k)^2, H, W
        mask = F.pixel_shuffle(mask, s)  # B, k*k, sH, sW
        mask = mask.view(b, 1, k * k, h * s, w * s)
        mask = torch.softmax(mask, dim=2)
        x_up = F.interpolate(x, scale_factor=s, mode="nearest")
        pad = k // 2
        x_unf = F.unfold(F.pad(x_up, [pad] * 4), kernel_size=k).view(b, c, k * k, h * s, w * s)
        return (x_unf * mask).sum(dim=2)


class DySampleLite(nn.Module):
    """Lightweight dynamic upsample (learned offset on nearest grid). MPS-safe."""

    def __init__(self, c1, scale=2):
        super().__init__()
        self.scale = scale
        self.offset = nn.Conv2d(c1, 2 * scale * scale, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale
        # predict offsets at low-res then pixel-shuffle to hi-res grid
        off = self.offset(x)  # B, 2*s*s, H, W
        off = F.pixel_shuffle(off, s)  # B, 2, sH, sW
        # base sampling grid
        ys = torch.linspace(-1, 1, h * s, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w * s, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        # small residual offsets
        off = off.permute(0, 2, 3, 1) * 0.25
        grid = (grid + off).clamp(-1, 1)
        x_up = F.interpolate(x, scale_factor=s, mode="nearest")
        return F.grid_sample(x_up, grid, mode="bilinear", align_corners=True, padding_mode="zeros")


class ECA(nn.Module):
    def __init__(self, c1, c2=None, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True).squeeze(-1).transpose(-1, -2)  # B,1,C
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1).sigmoid()
        return x * y


class CoordAtt(nn.Module):
    def __init__(self, c1, c2=None, reduction=32):
        super().__init__()
        mip = max(8, c1 // reduction)
        self.conv1 = Conv(c1, mip, 1)
        self.conv_h = nn.Conv2d(mip, c1, 1)
        self.conv_w = nn.Conv2d(mip, c1, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = self.conv1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return x * a_h * a_w


class EMA(nn.Module):
    """Efficient multi-scale attention (channel+spatial gates)."""

    def __init__(self, c1, c2=None, factor=8):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid = max(8, c1 // factor)
        self.conv1 = Conv(c1, mid, 1)
        self.conv_h = nn.Conv2d(mid, c1, 1)
        self.conv_w = nn.Conv2d(mid, c1, 1)
        self.conv3 = Conv(c1, c1, 3, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.conv1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return self.conv3(x * a_h * a_w)


class SimAM(nn.Module):
    def __init__(self, c1=None, c2=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e)


class ASFFLevel(nn.Module):
    """ASFF producing one fused scale from [P3,P4,P5]. level=0/1/2."""

    def __init__(self, c1, level=0, cm=16):
        super().__init__()
        self.level = int(level)
        self.cm = cm
        self._built = False

    def _build(self, xs):
        cs = [x.shape[1] for x in xs]
        c_out = cs[self.level]
        dev = xs[0].device
        self.proj = nn.ModuleList([Conv(c, self.cm, 1) for c in cs]).to(dev)
        self.weight = nn.Conv2d(self.cm * 3, 3, 1).to(dev)
        self.expand = Conv(self.cm, c_out, 1).to(dev)
        self._built = True

    def forward(self, xs):
        assert isinstance(xs, (list, tuple)) and len(xs) == 3
        if not self._built:
            self._build(xs)
        target = xs[self.level].shape[-2:]
        feats = []
        for i, x in enumerate(xs):
            t = self.proj[i](x)
            if t.shape[-2:] != target:
                t = F.interpolate(t, size=target, mode="nearest")
            feats.append(t)
        w = torch.softmax(self.weight(torch.cat(feats, 1)), dim=1)
        fused = sum(feats[i] * w[:, i : i + 1] for i in range(3))
        return self.expand(fused)


class BiFPNAdd(nn.Module):
    """Weighted residual add for two same-resolution features (BiFPN-style)."""

    def __init__(self, c1, c2=None):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4
        self.conv = Conv(c1, c1, 3, 1) if c2 is None or c2 == c1 else Conv(c1, c2, 3, 1)

    def forward(self, xs):
        a, b = xs
        if a.shape[1] != b.shape[1]:
            # channel mismatch: project b
            if not hasattr(self, "proj_b"):
                self.proj_b = Conv(b.shape[1], a.shape[1], 1).to(a.device)
            b = self.proj_b(b)
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="nearest")
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return self.conv(w[0] * a + w[1] * b)


# --- Dyn + DCN / ODConv bottlenecks (reuse DynamicConvolution helpers) ---
from ultralytics.nn.modules.addModules.DynamicConvolution import (  # noqa: E402
    Bottleneck_DynamicConv,
    C2f,
    C3,
    DynamicConv,
    PSABlock,
    _make_c3k,
    _make_c3k2,
)


class DeformConvLite(nn.Module):
    """DCNv2-style via torchvision if available, else depthwise 3x3 fallback."""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.offset = nn.Conv2d(c1, 2 * k * k, 3, s, 1)
        try:
            from torchvision.ops import DeformConv2d

            self.dcn = DeformConv2d(c1, c2, k, stride=s, padding=k // 2, bias=False)
            self.use_dcn = True
        except Exception:
            self.dcn = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
            self.use_dcn = False
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.use_dcn:
            off = self.offset(x)
            return self.act(self.bn(self.dcn(x, off)))
        return self.act(self.bn(self.dcn(x)))


class Bottleneck_DynDCN(nn.Module):
    """cv1 Conv + cv2 Dyn, then residual DCN refine (lite DCNv3 stand-in)."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DynamicConv(c_, c2, 3, stride=1, groups=g)
        self.dcn = DeformConvLite(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.dcn(y)
        return x + y if self.add else y


class ODConv(nn.Module):
    """Simplified omni-dimensional dynamic conv (per-sample expert mix of 3x3 kernels)."""

    def __init__(self, c1, c2, k=3, s=1, n_experts=4):
        super().__init__()
        self.n = n_experts
        self.s = s
        self.p = k // 2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c1, n_experts)
        self.weight = nn.Parameter(torch.randn(n_experts, c2, c1, k, k) * (2.0 / (c1 * k * k)) ** 0.5)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        b = x.shape[0]
        a = torch.softmax(self.fc(self.gap(x).flatten(1)), dim=-1)  # B,n
        # weight: (n, cout, cin, k, k) → per-sample mix
        w = torch.einsum("bn,nockh->bockh", a, self.weight)
        outs = [F.conv2d(x[i : i + 1], w[i], padding=self.p, stride=self.s) for i in range(b)]
        return self.act(self.bn(torch.cat(outs, 0)))


class Bottleneck_ODConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = ODConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


C3k_DynDCN = _make_c3k(Bottleneck_DynDCN, "C3k_DynDCN")
C3k2_DynDCN = _make_c3k2(Bottleneck_DynDCN, C3k_DynDCN, "C3k2_DynDCN")
C3k_ODConv = _make_c3k(Bottleneck_ODConv, "C3k_ODConv")
C3k2_ODConv = _make_c3k2(Bottleneck_ODConv, C3k_ODConv, "C3k2_ODConv")

# pickle needs classes on the declared __module__; factories stamp DynamicConvolution
for _cls in (C3k_DynDCN, C3k2_DynDCN, C3k_ODConv, C3k2_ODConv, Bottleneck_DynDCN, Bottleneck_ODConv):
    _cls.__module__ = __name__
    _cls.__qualname__ = _cls.__name__

__all__ += ["C3k2_DynDCN", "C3k2_ODConv"]
