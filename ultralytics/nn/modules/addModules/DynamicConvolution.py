"""Overnight Dyn variants: StripMix / MixPool / expert-count / shared helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import CondConv2d

__all__ = [
    "C3k2_DynamicConv",
    "DynamicConv",
    "C3k2_StripDyn",
    "C3k2_StripMix",
    "C3k2_MixPool",
    "C3k2_DynE2",
    "C3k2_DynE6",
    "C3k2_DynE8",
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

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DynamicConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=1,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=False,
        num_experts=4,
    ):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(
            in_features, out_features, kernel_size, stride, padding, dilation, groups, bias, num_experts
        )

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = torch.sigmoid(self.routing(pooled))
        return self.cond_conv(x, w)


class StripMixDynamicConv(DynamicConv):
    """Param-free strip mix into Dyn GAP features (same Linear, same AdamW groups)."""

    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        sh = F.adaptive_avg_pool2d(x, (1, x.shape[-1])).mean(-1).flatten(1)
        sw = F.adaptive_avg_pool2d(x, (x.shape[-2], 1)).mean(-2).flatten(1)
        feat = gap + self.beta * ((sh - gap) + (sw - gap))
        w = torch.sigmoid(self.routing(feat))
        return self.cond_conv(x, w)


class MixPoolDynamicConv(DynamicConv):
    """Param-free GAP+GMP mix into Dyn router (same Linear)."""

    def __init__(self, *args, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
        a = self.alpha
        feat = a * gap + (1.0 - a) * gmp
        w = torch.sigmoid(self.routing(feat))
        return self.cond_conv(x, w)


class StripDynamicConv(DynamicConv):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding="", dilation=1, groups=1, bias=False, num_experts=4):
        super().__init__(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias, num_experts)
        self.route_h = nn.Linear(in_features, num_experts)
        self.route_w = nn.Linear(in_features, num_experts)
        nn.init.zeros_(self.route_h.weight)
        nn.init.zeros_(self.route_h.bias)
        nn.init.zeros_(self.route_w.weight)
        nn.init.zeros_(self.route_w.bias)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        sh = F.adaptive_avg_pool2d(x, (1, x.shape[-1])).mean(-1).flatten(1)
        sw = F.adaptive_avg_pool2d(x, (x.shape[-2], 1)).mean(-2).flatten(1)
        w = torch.sigmoid(self.routing(gap) + self.route_h(sh) + self.route_w(sw))
        return self.cond_conv(x, w)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DynamicConv(nn.Module):
    conv_cls = DynamicConv

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = self.conv_cls(c_, c2, 3, stride=1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_StripDyn(Bottleneck_DynamicConv):
    conv_cls = StripDynamicConv


class Bottleneck_StripMix(Bottleneck_DynamicConv):
    conv_cls = StripMixDynamicConv


class Bottleneck_MixPool(Bottleneck_DynamicConv):
    conv_cls = MixPoolDynamicConv


class DynamicConvE2(DynamicConv):
    def __init__(self, *a, **k):
        k["num_experts"] = 2
        super().__init__(*a, **k)


class DynamicConvE6(DynamicConv):
    def __init__(self, *a, **k):
        k["num_experts"] = 6
        super().__init__(*a, **k)


class DynamicConvE8(DynamicConv):
    def __init__(self, *a, **k):
        k["num_experts"] = 8
        super().__init__(*a, **k)


class Bottleneck_DynE2(Bottleneck_DynamicConv):
    conv_cls = DynamicConvE2


class Bottleneck_DynE6(Bottleneck_DynamicConv):
    conv_cls = DynamicConvE6


class Bottleneck_DynE8(Bottleneck_DynamicConv):
    conv_cls = DynamicConvE8


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


def _make_c3k(bottleneck_cls, name: str):
    class _C3k(C3):
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)
            self.m = nn.Sequential(*(bottleneck_cls(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    _C3k.__name__ = name
    _C3k.__qualname__ = name
    _C3k.__module__ = __name__
    return _C3k


def _make_c3k2(bottleneck_cls, c3k_cls, name: str):
    class _C3k2(C2f):
        def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
            super().__init__(c1, c2, n, shortcut, g, e)
            self.m = nn.ModuleList(
                nn.Sequential(
                    bottleneck_cls(self.c, self.c, shortcut, g),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
                if attn
                else c3k_cls(self.c, self.c, 2, shortcut, g)
                if c3k
                else bottleneck_cls(self.c, self.c, shortcut, g)
                for _ in range(n)
            )

    _C3k2.__name__ = name
    _C3k2.__qualname__ = name
    _C3k2.__module__ = __name__
    return _C3k2


C3k_DynamicConv = _make_c3k(Bottleneck_DynamicConv, "C3k_DynamicConv")
C3k2_DynamicConv = _make_c3k2(Bottleneck_DynamicConv, C3k_DynamicConv, "C3k2_DynamicConv")

C3k_StripDyn = _make_c3k(Bottleneck_StripDyn, "C3k_StripDyn")
C3k2_StripDyn = _make_c3k2(Bottleneck_StripDyn, C3k_StripDyn, "C3k2_StripDyn")

C3k_StripMix = _make_c3k(Bottleneck_StripMix, "C3k_StripMix")
C3k2_StripMix = _make_c3k2(Bottleneck_StripMix, C3k_StripMix, "C3k2_StripMix")

C3k_MixPool = _make_c3k(Bottleneck_MixPool, "C3k_MixPool")
C3k2_MixPool = _make_c3k2(Bottleneck_MixPool, C3k_MixPool, "C3k2_MixPool")

C3k_DynE2 = _make_c3k(Bottleneck_DynE2, "C3k_DynE2")
C3k2_DynE2 = _make_c3k2(Bottleneck_DynE2, C3k_DynE2, "C3k2_DynE2")

C3k_DynE6 = _make_c3k(Bottleneck_DynE6, "C3k_DynE6")
C3k2_DynE6 = _make_c3k2(Bottleneck_DynE6, C3k_DynE6, "C3k2_DynE6")

C3k_DynE8 = _make_c3k(Bottleneck_DynE8, "C3k_DynE8")
C3k2_DynE8 = _make_c3k2(Bottleneck_DynE8, C3k_DynE8, "C3k2_DynE8")


# Beta variants as distinct classes for YAML (no extra ctor args in parse_model)
class StripMixDynamicConv05(StripMixDynamicConv):
    def __init__(self, *a, **k):
        super().__init__(*a, beta=0.05, **k)


class StripMixDynamicConv20(StripMixDynamicConv):
    def __init__(self, *a, **k):
        super().__init__(*a, beta=0.2, **k)


class Bottleneck_StripMix05(Bottleneck_DynamicConv):
    conv_cls = StripMixDynamicConv05


class Bottleneck_StripMix20(Bottleneck_DynamicConv):
    conv_cls = StripMixDynamicConv20


C3k_StripMix05 = _make_c3k(Bottleneck_StripMix05, "C3k_StripMix05")
C3k2_StripMix05 = _make_c3k2(Bottleneck_StripMix05, C3k_StripMix05, "C3k2_StripMix05")
C3k_StripMix20 = _make_c3k(Bottleneck_StripMix20, "C3k_StripMix20")
C3k2_StripMix20 = _make_c3k2(Bottleneck_StripMix20, C3k_StripMix20, "C3k2_StripMix20")


class SoftmaxDynamicConv(DynamicConv):
    """Dyn with softmax routing (experts sum to 1) instead of independent sigmoid."""

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = torch.softmax(self.routing(pooled), dim=-1)
        return self.cond_conv(x, w)


class TempDynamicConv05(DynamicConv):
    """Sharper sigmoid routing (logits / 0.5)."""

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = torch.sigmoid(self.routing(pooled) / 0.5)
        return self.cond_conv(x, w)


class TempDynamicConv20(DynamicConv):
    """Softer sigmoid routing (logits / 2.0)."""

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        w = torch.sigmoid(self.routing(pooled) / 2.0)
        return self.cond_conv(x, w)


class MixPoolDynamicConv25(MixPoolDynamicConv):
    def __init__(self, *a, **k):
        super().__init__(*a, alpha=0.25, **k)


class MixPoolDynamicConv75(MixPoolDynamicConv):
    def __init__(self, *a, **k):
        super().__init__(*a, alpha=0.75, **k)


class StripMixDynamicConv01(StripMixDynamicConv):
    def __init__(self, *a, **k):
        super().__init__(*a, beta=0.01, **k)


class LocalMixDynamicConv(DynamicConv):
    """Param-free: blend GAP with 3x3 avg-pool collapsed to vector (local context)."""

    def __init__(self, *a, beta: float = 0.1, **k):
        super().__init__(*a, **k)
        self.beta = float(beta)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1).mean((-2, -1))
        feat = gap + self.beta * (local - gap)
        w = torch.sigmoid(self.routing(feat))
        return self.cond_conv(x, w)


class Bottleneck_SoftmaxDyn(Bottleneck_DynamicConv):
    conv_cls = SoftmaxDynamicConv


class Bottleneck_Temp05(Bottleneck_DynamicConv):
    conv_cls = TempDynamicConv05


class Bottleneck_Temp20(Bottleneck_DynamicConv):
    conv_cls = TempDynamicConv20


class Bottleneck_MixPool25(Bottleneck_DynamicConv):
    conv_cls = MixPoolDynamicConv25


class Bottleneck_MixPool75(Bottleneck_DynamicConv):
    conv_cls = MixPoolDynamicConv75


class Bottleneck_StripMix01(Bottleneck_DynamicConv):
    conv_cls = StripMixDynamicConv01


class Bottleneck_LocalMix(Bottleneck_DynamicConv):
    conv_cls = LocalMixDynamicConv


C3k_SoftmaxDyn = _make_c3k(Bottleneck_SoftmaxDyn, "C3k_SoftmaxDyn")
C3k2_SoftmaxDyn = _make_c3k2(Bottleneck_SoftmaxDyn, C3k_SoftmaxDyn, "C3k2_SoftmaxDyn")
C3k_Temp05 = _make_c3k(Bottleneck_Temp05, "C3k_Temp05")
C3k2_Temp05 = _make_c3k2(Bottleneck_Temp05, C3k_Temp05, "C3k2_Temp05")
C3k_Temp20 = _make_c3k(Bottleneck_Temp20, "C3k_Temp20")
C3k2_Temp20 = _make_c3k2(Bottleneck_Temp20, C3k_Temp20, "C3k2_Temp20")
C3k_MixPool25 = _make_c3k(Bottleneck_MixPool25, "C3k_MixPool25")
C3k2_MixPool25 = _make_c3k2(Bottleneck_MixPool25, C3k_MixPool25, "C3k2_MixPool25")
C3k_MixPool75 = _make_c3k(Bottleneck_MixPool75, "C3k_MixPool75")
C3k2_MixPool75 = _make_c3k2(Bottleneck_MixPool75, C3k_MixPool75, "C3k2_MixPool75")
C3k_StripMix01 = _make_c3k(Bottleneck_StripMix01, "C3k_StripMix01")
C3k2_StripMix01 = _make_c3k2(Bottleneck_StripMix01, C3k_StripMix01, "C3k2_StripMix01")
C3k_LocalMix = _make_c3k(Bottleneck_LocalMix, "C3k_LocalMix")
C3k2_LocalMix = _make_c3k2(Bottleneck_LocalMix, C3k_LocalMix, "C3k2_LocalMix")

__all__ += [
    "C3k2_StripMix05",
    "C3k2_StripMix20",
    "C3k2_SoftmaxDyn",
    "C3k2_Temp05",
    "C3k2_Temp20",
    "C3k2_MixPool25",
    "C3k2_MixPool75",
    "C3k2_StripMix01",
    "C3k2_LocalMix",
]
