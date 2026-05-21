"""Clifford neural-network layers: rotor sandwich, channel-mixing linear, geometric norm/GELU."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import CliffordAlgebra


class RotorLayer(nn.Module):
    """Learnable rotor: applies sandwich R x R~ for rotation equivariance.

    Learns bivector coefficients, builds rotors via exponentiation,
    and applies the sandwich product.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, init_std: float = 0.01):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.num_bivectors = math.comb(algebra.n, 2)
        self.register_buffer("_biv_indices", algebra.bivector_indices())

        self.biv_weights = nn.Parameter(torch.zeros(channels, self.num_bivectors))
        nn.init.normal_(self.biv_weights, std=init_std)

    def _build_bivector(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        biv = torch.zeros(self.channels, self.algebra.dim, device=device, dtype=dtype)
        biv[:, self._biv_indices] = self.biv_weights.to(device=device, dtype=dtype)
        return biv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply R * x * R~ per channel.

        Args:
            x: [B, C, dim] (or [B, S, C, dim] for seq).
        """
        biv = self._build_bivector(x.device, x.dtype)
        while biv.dim() < x.dim():
            biv = biv.unsqueeze(0)

        R = self.algebra.exp_bivector(-0.5 * biv)
        R_rev = self.algebra.reverse(R)
        return self.algebra.sandwich_product(R, x, R_rev)


class CliffordLinear(nn.Module):
    """Channel-mixing linear layer on multivectors.

    out[b, ..., o, k] = sum_i W[o, i] * x[b, ..., i, k]
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.algebra = algebra
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels, algebra.dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("oi,...id->...od", self.weight, x)
        if self.bias is not None:
            out = out + self.bias
        return out


class CliffordLayerNorm(nn.Module):
    """Geometric LayerNorm: normalizes multivector norm, preserves direction.

    x_normed = x / ||x||, then affine transform with scale on all blades and
    bias on grade-0 only. Optionally recovers log-magnitude into grade-0.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, eps: float = 1e-6, recover_scale: bool = True):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.scale_gate = nn.Parameter(torch.zeros(channels)) if recover_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.algebra.norm_sq(x).sqrt()
        x_normed = x / norm.clamp(min=self.eps)
        out = x_normed * self.weight.unsqueeze(-1)

        scalar_mask = self.algebra.grade_mask(0, x.device, x.dtype)
        out = out + self.bias.unsqueeze(-1) * scalar_mask

        if self.scale_gate is not None:
            log_norm = torch.log1p(norm)
            out = out + self.scale_gate.unsqueeze(-1) * log_norm * scalar_mask

        return out


class GeometricGELU(nn.Module):
    """Geometric GELU: x' = x * GELU(||x|| + b) / ||x||.

    Preserves direction while applying nonlinear magnitude scaling.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        super().__init__()
        self.algebra = algebra
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.algebra.norm_sq(x).sqrt()
        scale = F.gelu(norm + self.bias.unsqueeze(-1)) / norm.clamp(min=1e-6)
        return x * scale


class BladeSelector(nn.Module):
    """Per-grade gating: 2 * sigmoid(logit) per grade."""

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        super().__init__()
        self.algebra = algebra
        self.num_grades = algebra.n + 1
        self.logits = nn.Parameter(torch.zeros(channels, self.num_grades))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = 2.0 * torch.sigmoid(self.logits)
        grade_idx = self.algebra._grade_index.to(x.device)
        gates_per_blade = gates[..., grade_idx]
        return x * gates_per_blade


class CliffordAttractorBlock(nn.Module):
    """Single block of the fixed-point map f.

    x → LayerNorm → Rotor(skip) → Linear → GP(x,x) → GeomGELU
      → Linear → BladeSelector → LayerNorm → + skip
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, hidden_channels: Optional[int] = None,
                 use_geometric_activation: bool = True, use_blade_selector: bool = True,
                 init_std: float = 0.01):
        super().__init__()
        self.algebra = algebra
        hidden = hidden_channels or channels

        self.norm1 = CliffordLayerNorm(algebra, channels)
        self.rotor = RotorLayer(algebra, channels, init_std=init_std)
        self.linear1 = CliffordLinear(algebra, channels, hidden)
        self.act = GeometricGELU(algebra, hidden) if use_geometric_activation else nn.Identity()
        self.linear2 = CliffordLinear(algebra, hidden, channels)
        self.gate = BladeSelector(algebra, channels) if use_blade_selector else nn.Identity()
        self.norm2 = CliffordLayerNorm(algebra, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.rotor(h) + x
        h2 = self.linear1(h)
        h2 = self.algebra.geometric_product(h2, h2)
        h2 = self.act(h2)
        h2 = self.linear2(h2)
        h2 = self.gate(h2)
        return self.norm2(h2) + h
