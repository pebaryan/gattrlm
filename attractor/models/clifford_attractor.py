"""
Geometric Clifford/Rotor-based Attractor Model.

Extends the Attractor framework with Clifford algebra layers:
  - Cl(p,q) multivector representations with geometric product operations
  - RotorLayer: Learnable sandwich product R x R~ for rotation/reflection equivariance
  - CliffordLinear: Channel mixing with grade structure preservation
  - Fixed-point attractor dynamics with Anderson acceleration
  - Implicit differentiation (IFT) for memory-efficient gradients

Compatible with the existing AttractorConfig system and training loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================
#  Clifford Algebra Core — Geometric Product Table & Utilities
# ========================================================================


def _blade_grade(bitmask: int) -> int:
    """Grade (number of vector factors) of a basis blade."""
    return bitmask.bit_count()


def _gp_blade(a: int, b: int, p: int, q: int, r: int, n: int) -> Tuple[int, int]:
    """Geometric product of two basis blades in Cl(p,q,r).

    Returns (result_bitmask, sign). Returns (0, 0) if the product vanishes.
    """
    result = a
    sign = 1
    for i in range(n):
        if b & (1 << i):
            swaps = bin(result >> (i + 1)).count("1")
            sign *= -1 if swaps & 1 else 1
            if result & (1 << i):
                result &= ~(1 << i)
                if i < p:
                    pass
                elif i < p + q:
                    sign *= -1
                else:
                    return 0, 0
            else:
                result |= (1 << i)
    return result, sign


def build_gp_table(p: int, q: int, r: int = 0) -> torch.Tensor:
    """Build geometric product Cayley table for Cl(p,q,r).

    Returns T[dim, dim, dim] where T[i,j,k] = coeff of blade k in gp(basis_i, basis_j).
    """
    n = p + q + r
    dim = 1 << n
    table = torch.zeros(dim, dim, dim, dtype=torch.float32)
    for i in range(dim):
        for j in range(dim):
            result, sign = _gp_blade(i, j, p, q, r, n)
            if result != 0 or sign != 0:
                table[i, j, result] = float(sign)
    return table


def _extract_sparse_gp(table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract sparse (i, j, k, sign) triples from a dense GP table.

    The GP table has shape [dim, dim, dim] but each basis blade product
    produces exactly one blade (table[i,j,:] has one non-zero). So the
    table has exactly dim*dim non-zero entries out of dim*dim*dim.

    Returns:
        i_idx: [nnz] — indices into the first operand x
        j_idx: [nnz] — indices into the second operand y
        k_idx: [nnz] — indices into the result
        signs: [nnz] — corresponding GP coefficients
    """
    nnz_mask = table.abs() > 0
    indices = torch.where(nnz_mask)
    i_idx, j_idx, k_idx = indices
    signs = table[i_idx, j_idx, k_idx]
    return i_idx, j_idx, k_idx, signs


def _filter_sparse_op(
    i_idx: torch.Tensor, j_idx: torch.Tensor, k_idx: torch.Tensor,
    signs: torch.Tensor, grade_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter sparse GP triples to keep only outer (wedge) product entries.

    The outer product keeps entries where grade(k) == grade(i) + grade(j),
    i.e., the grade of the result equals the sum of the input grades.
    """
    gi = grade_index[i_idx]
    gj = grade_index[j_idx]
    gk = grade_index[k_idx]
    mask = gk == gi + gj
    return i_idx[mask], j_idx[mask], k_idx[mask], signs[mask]


def _filter_sparse_ip(
    i_idx: torch.Tensor, j_idx: torch.Tensor, k_idx: torch.Tensor,
    signs: torch.Tensor, grade_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter sparse GP triples to keep only inner product entries.

    The Hestenes inner product keeps entries where
    grade(k) == abs(grade(i) - grade(j)).
    """
    gi = grade_index[i_idx]
    gj = grade_index[j_idx]
    gk = grade_index[k_idx]
    mask = gk == (gi - gj).abs()
    return i_idx[mask], j_idx[mask], k_idx[mask], signs[mask]


def _grade_index_map(n: int) -> torch.Tensor:
    """Return [dim] tensor with grade of each basis blade."""
    return torch.tensor([i.bit_count() for i in range(1 << n)], dtype=torch.long)


def _blade_metric_signs_precomputed(p: int, q: int, r: int) -> torch.Tensor:
    """Precompute metric sign for each basis blade in Cl(p,q,r).

    In Cl(p,q): if a blade contains k negative-norm vectors, its metric sign is (-1)^k.
    """
    n = p + q + r
    dim = 1 << n
    signs = torch.ones(dim, dtype=torch.float32)
    for blade in range(dim):
        s = 1.0
        for j in range(n):
            if blade & (1 << j):
                if j >= p + q:
                    s = 0.0
                    break
                if j >= p:
                    s *= -1.0
        signs[blade] = s
    return signs


# ========================================================================
#  CliffordAlgebra — Precomputed tables with batched operations
# ========================================================================


class CliffordAlgebra(nn.Module):
    """Differentiable Clifford algebra kernel for Cl(p,q,r).

    Stores precomputed GP table, grade indices, and metric signs as buffers.
    All operations support arbitrary batch dimensions.
    """

    def __init__(self, p: int, q: int, r: int = 0):
        super().__init__()
        self.p = p
        self.q = q
        self.r = r
        self.n = p + q + r
        self.dim = 1 << self.n

        gp_table = build_gp_table(p, q, r)
        self.register_buffer("_gp_table", gp_table)
        self.register_buffer("_grade_index", _grade_index_map(self.n))
        self.register_buffer("_metric_signs", _blade_metric_signs_precomputed(p, q, r))

        # Sparse GP indices: for efficient geometric product at higher dimensions.
        # Each basis blade product e_i * e_j = sign * e_k (single blade, no sum),
        # so we can replace dense einsum O(dim^3) with sparse index_select + index_add_ O(dim^2).
        i_idx, j_idx, k_idx, signs = _extract_sparse_gp(gp_table)
        self.register_buffer("_gp_i", i_idx)
        self.register_buffer("_gp_j", j_idx)
        self.register_buffer("_gp_k", k_idx)
        self.register_buffer("_gp_sign", signs)

        # Sparse OP/IP indices (for opt-in sparse path).
        grade_idx = self._grade_index
        op_i, op_j, op_k, op_sign = _filter_sparse_op(i_idx, j_idx, k_idx, signs, grade_idx)
        self.register_buffer("_op_i", op_i)
        self.register_buffer("_op_j", op_j)
        self.register_buffer("_op_k", op_k)
        self.register_buffer("_op_sign", op_sign)
        ip_i, ip_j, ip_k, ip_sign = _filter_sparse_ip(i_idx, j_idx, k_idx, signs, grade_idx)
        self.register_buffer("_ip_i", ip_i)
        self.register_buffer("_ip_j", ip_j)
        self.register_buffer("_ip_k", ip_k)
        self.register_buffer("_ip_sign", ip_sign)

    # --- Core operations ---

    def _dense_geometric_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geometric product via dense einsum (efficient for small dim)."""
        gp = self._gp_table.to(device=x.device, dtype=x.dtype)
        return torch.einsum("...i,...j,ijk->...k", x, y, gp)

    def _dense_filtered_table(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        attr_name: str,
        predicate: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Lazily build and cache a dense grade-filtered Cayley table."""
        table = getattr(self, attr_name, None)
        if table is None:
            grade_idx = self._grade_index
            table = torch.zeros(self.dim, self.dim, self.dim, device=grade_idx.device, dtype=self._gp_table.dtype)
            for n in range(len(self._gp_i)):
                ii = int(self._gp_i[n])
                jj = int(self._gp_j[n])
                kk = int(self._gp_k[n])
                if predicate(grade_idx[ii], grade_idx[jj], grade_idx[kk], self._gp_sign[n]):
                    table[ii, jj, kk] = float(self._gp_sign[n])
            self.register_buffer(attr_name, table)
        return table.to(device=device, dtype=dtype)

    # Sparse GP uses index_select + index_add_ which is O(dim^2) vs O(dim^3) for dense.
    # On CPU, einsum (MKL) is faster for dim <= 128. On GPU with @torch.compile,
    # sparse becomes beneficial around dim >= 32. Set higher to always prefer dense.
    _USE_SPARSE_GP = False

    def _sparse_contract(
        self, x: torch.Tensor, y: torch.Tensor,
        i_idx: torch.Tensor, j_idx: torch.Tensor,
        k_idx: torch.Tensor, signs: torch.Tensor
    ) -> torch.Tensor:
        """Generic sparse contraction: result[k] += x[i] * y[j] * sign
        for each (i, j, k, sign) triple.
        """
        i_idx = i_idx.to(device=x.device)
        j_idx = j_idx.to(device=x.device)
        k_idx = k_idx.to(device=x.device)
        signs = signs.to(device=x.device, dtype=x.dtype)
        x_gathered = x.index_select(-1, i_idx)
        y_gathered = y.index_select(-1, j_idx)
        products = x_gathered * y_gathered * signs
        out = products.new_zeros(*products.shape[:-1], self.dim)
        return out.index_add_(-1, k_idx, products)

    def geometric_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geometric product: x * y. Supports arbitrary batch dims.

        Uses dense einsum by default. Sparse (index_select + index_add_)
        can be enabled by setting `alg._USE_SPARSE_GP = True` for GPU
        workloads at dim >= 32.
        """
        if self._USE_SPARSE_GP:
            return self._sparse_contract(
                x, y, self._gp_i, self._gp_j, self._gp_k, self._gp_sign
            )
        return self._dense_geometric_product(x, y)

    def outer_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Outer (wedge) product: grade-raising part of the geometric product.

        Uses precomputed dense OP table (cached buffer) for the default path,
        or sparse contraction when _USE_SPARSE_GP is enabled.
        """
        if self._USE_SPARSE_GP:
            return self._sparse_contract(
                x, y, self._op_i, self._op_j, self._op_k, self._op_sign
            )
        op = self._dense_filtered_table(
            device=x.device,
            dtype=x.dtype,
            attr_name="_op_table",
            predicate=lambda gi, gj, gk, _: gk == gi + gj,
        )
        return torch.einsum("...i,...j,ijk->...k", x, y, op)

    def inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hestenes inner product (grade-lowering).

        Uses precomputed dense IP table (cached buffer) for the default path,
        or sparse contraction when _USE_SPARSE_GP is enabled.
        """
        if self._USE_SPARSE_GP:
            return self._sparse_contract(
                x, y, self._ip_i, self._ip_j, self._ip_k, self._ip_sign
            )
        ip = self._dense_filtered_table(
            device=x.device,
            dtype=x.dtype,
            attr_name="_ip_table",
            predicate=lambda gi, gj, gk, _: gk == (gi - gj).abs(),
        )
        return torch.einsum("...i,...j,ijk->...k", x, y, ip)

    def sandwich_product(
        self, left: torch.Tensor, x: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Sandwich product: left * x * right (typically R * x * R~)."""
        temp = self.geometric_product(left, x)
        return self.geometric_product(temp, right)

    # --- Grade operations ---

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reversion: (~x)_k = (-1)^{k(k-1)/2} * x_k."""
        grades = self._grade_index.to(x.device)
        signs = ((-1.0) ** (grades * (grades - 1) // 2)).to(x.dtype)
        return x * signs

    def grade_involution(self, x: torch.Tensor) -> torch.Tensor:
        """Grade involution: (x^)_k = (-1)^k * x_k."""
        grades = self._grade_index.to(x.device)
        signs = ((-1.0) ** grades).to(x.dtype)
        return x * signs

    def grade_projection(self, x: torch.Tensor, grade: int) -> torch.Tensor:
        """Project to blades of given grade."""
        mask = (self._grade_index == grade).to(x.device, dtype=x.dtype)
        return x * mask

    def scalar_part(self, x: torch.Tensor) -> torch.Tensor:
        """Grade-0 (scalar) component."""
        return self.grade_projection(x, 0)

    # --- Norm ---

    def blade_metric_signs(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Cached metric signs for each blade."""
        return self._metric_signs.to(device=device, dtype=dtype)

    def norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """Euclidean squared norm: sum(x_i^2).

        Uses absolute values of metric signs so that the norm is always
        non-negative. Clamps to 1e-16 minimum to prevent sqrt gradient blowup
        (inf * 0 = NaN).

        For Cl(p,q) with mixed metric signatures, the metric-weighted norm
        can be near zero even for non-zero multivectors due to sign cancellation.
        We use the Euclidean norm for numerical stability in neural layers.
        """
        signs = self.blade_metric_signs(x.device, x.dtype).abs()
        return (x ** 2 * signs).sum(dim=-1, keepdim=True).clamp(min=1e-16)

    # --- Bivector exponential ---

    def exp_bivector(self, biv: torch.Tensor) -> torch.Tensor:
        """Exponential of a bivector in Cl(p,q).

        If B^2 < 0 (elliptic): exp(B) = cos|B| + (B/|B|) sin|B|
        If B^2 > 0 (hyperbolic): exp(B) = cosh|B| + (B/|B|) sinh|B|

        Uses Taylor expansion for small angles to avoid numerical instability:
        - For small theta (< 1e-4): exp(B) ~ 1 + B + B^2/2
        - For larger theta: standard formula with sin(theta)/theta

        Args:
            biv: [..., dim] (only grade-2 should be non-zero).

        Returns:
            Rotor: [..., dim] (scalar + bivector).
        """
        B2 = self.geometric_product(biv, biv)
        scalar_B2 = self.scalar_part(B2)

        # scalar_B2 is [..., dim] with only grade-0 component non-zero
        theta_sq = scalar_B2.abs()
        # Clamp BEFORE sqrt to avoid 0 * inf in gradient when theta_sq = 0
        theta_sq_safe = theta_sq.clamp(min=1e-16)
        theta_scalar = theta_sq_safe.sqrt()
        theta_val = theta_scalar[..., 0:1]  # [..., 1]

        is_elliptic = scalar_B2[..., 0:1] < 0
        one = torch.zeros_like(biv)
        one[..., 0:1] = 1.0

        # Use Taylor expansion for small theta to avoid 1/theta blowup in gradients
        small_angle = 1e-4
        is_small = theta_val < small_angle

        # --- Standard formula (for theta >= small_angle) ---
        theta_clamped = theta_val.clamp(min=small_angle)
        cos_std = torch.where(
            is_elliptic, theta_clamped.cos(), theta_clamped.cosh()
        )
        sin_over_theta_std = torch.where(
            is_elliptic,
            theta_clamped.sin() / theta_clamped,
            theta_clamped.sinh() / theta_clamped,
        )

        # --- Taylor expansion (for theta < small_angle) ---
        # cos(θ) ≈ 1 - θ²/2, cosh(θ) ≈ 1 + θ²/2
        cos_taylor = torch.where(
            is_elliptic,
            1.0 - 0.5 * theta_val.square(),
            1.0 + 0.5 * theta_val.square(),
        )
        # sin(θ)/θ ≈ 1 - θ²/6, sinh(θ)/θ ≈ 1 + θ²/6
        sin_over_theta_taylor = torch.where(
            is_elliptic,
            1.0 - theta_val.square() / 6.0,
            1.0 + theta_val.square() / 6.0,
        )

        cos_val = torch.where(is_small, cos_taylor, cos_std)
        sin_over_theta = torch.where(is_small, sin_over_theta_taylor, sin_over_theta_std)

        return cos_val * one + sin_over_theta * biv

    # --- Index helpers ---

    def bivector_indices(self) -> torch.Tensor:
        """Return indices of grade-2 basis blades in [0, dim)."""
        return torch.where(self._grade_index == 2)[0]

    def grade_mask(self, grade: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (self._grade_index == grade).to(device=device, dtype=dtype)


# ========================================================================
#  Clifford Neural Network Layers
# ========================================================================


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
        """Assemble dense bivector [C, dim] from coefficients."""
        biv = torch.zeros(self.channels, self.algebra.dim, device=device, dtype=dtype)
        biv[:, self._biv_indices] = self.biv_weights.to(device=device, dtype=dtype)
        return biv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply R * x * R~ per channel.

        Args:
            x: [B, C, dim] (or [B, S, C, dim] for seq).

        Returns:
            Same shape as input.
        """
        biv = self._build_bivector(x.device, x.dtype)  # [C, dim]
        # Broadcast over batch and seq dims: expand to [1, 1, C, dim]
        while biv.dim() < x.dim():
            biv = biv.unsqueeze(0)

        R = self.algebra.exp_bivector(-0.5 * biv)  # [1, 1, C, dim] or [1, C, dim]
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
        """Args:
            x: [B, ..., C_in, dim].

        Returns:
            [B, ..., C_out, dim].
        """
        out = torch.einsum("oi,...id->...od", self.weight, x)
        if self.bias is not None:
            out = out + self.bias
        return out


class CliffordLayerNorm(nn.Module):
    """Geometric LayerNorm: normalizes multivector norm, preserves direction.

    x_normed = x / ||x||, then affine transform with scale on all blades
    and bias on grade-0 only. Optionally recovers log-magnitude in grade-0.
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
        """Args:
            x: [B, ..., C, dim].

        Returns:
            Same shape, normalized.
        """
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
        gates = 2.0 * torch.sigmoid(self.logits)  # [C, G]
        grade_idx = self._ensure_grade_index(x.device)
        gates_per_blade = gates[..., grade_idx]  # [C, dim]
        return x * gates_per_blade

    def _ensure_grade_index(self, device: torch.device) -> torch.Tensor:
        return self.algebra._grade_index.to(device)


# ========================================================================
#  Clifford Attractor Block (the fixed-point map f)
# ========================================================================


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
        h2 = self.algebra.geometric_product(h2, h2)  # GP(x,x)
        h2 = self.act(h2)
        h2 = self.linear2(h2)
        h2 = self.gate(h2)
        return self.norm2(h2) + h


# ========================================================================
#  DEQ Fixed-Point Solver with Implicit Differentiation
# ========================================================================


class DEQFixedPoint(torch.autograd.Function):
    """Custom autograd Function for DEQ fixed-point solving.

    Forward: Anderson-accelerated fixed-point iteration.
    Backward: Implicit Function Theorem (IFT) — solve J^T * g = -dl/dx_out
    using conjugate gradient / Neumann series, without storing the solver
    trajectory (constant memory).
    """

    @staticmethod
    def forward(ctx, f: Callable, x0: torch.Tensor, max_iter: int, tol: float,
                anderson_m: int, anderson_beta: float) -> torch.Tensor:
        """Solve x* = f(x*) via Anderson acceleration.

        Args:
            f: callable defining the fixed-point equation.
            x0: [*, dim] initial guess.
            max_iter: maximum iterations.
            tol: convergence tolerance.
            anderson_m: Anderson memory size. 0 = plain fixed-point.
            anderson_beta: Anderson mixing parameter.

        Returns:
            x_star: [*, dim] fixed-point solution (with gradreq).
        """
        # Save f for backward
        ctx.f = f
        ctx.max_iter = max_iter
        ctx.tol = tol

        # Capture f's parameters for IFT gradient computation in backward.
        # f may be a bound method (e.g., self._f), so check __self__ for module params.
        param_source = f
        if hasattr(f, '__self__') and hasattr(f.__self__, 'parameters'):
            param_source = f.__self__
        if hasattr(param_source, 'parameters'):
            ctx.params = [p for p in param_source.parameters() if p.requires_grad]
        else:
            ctx.params = []

        with torch.no_grad():
            x = x0.detach().clone()
            # Anderson history
            X_hist, G_hist = [], []

            for i in range(max_iter):
                fx = f(x)
                gx = fx - x
                residual = gx.norm(dim=-1).mean().item()

                if residual < tol:
                    x = fx
                    break

                X_hist.append(x.detach().clone())
                G_hist.append(gx.detach().clone())

                if anderson_m > 0 and len(X_hist) >= 2:
                    if len(X_hist) > anderson_m:
                        X_hist.pop(0)
                        G_hist.pop(0)
                    x = DEQFixedPoint._anderson_step(X_hist, G_hist, anderson_beta)
                else:
                    x = fx

        # Save final state for IFT backward
        ctx.save_for_backward(x.detach().clone())
        # Return with gradient tracking for backward
        return x.detach().requires_grad_()

    @staticmethod
    def _anderson_step(X_hist, G_hist, beta: float) -> torch.Tensor:
        """Single Anderson extrapolation step."""
        k = len(X_hist)
        G = torch.stack(G_hist, dim=-1)  # [*, dim, k]
        X = torch.stack(X_hist, dim=-1)  # [*, dim, k]
        flat_shape = G.shape[:-2]
        cd = G.shape[-2]

        G_flat = G.reshape(-1, cd, k)  # [B, dim, k]
        G_diff = G_flat[:, :, 1:] - G_flat[:, :, :1]  # [B, dim, k-1]
        G0 = G_flat[:, :, :1]  # [B, dim, 1]

        GtG = torch.bmm(G_diff.transpose(1, 2), G_diff)  # [B, k-1, k-1]
        rhs = torch.bmm(G_diff.transpose(1, 2), -G0)  # [B, k-1, 1]

        # Regularized solve to avoid near-singular GtG
        reg = 1e-4 * torch.eye(k - 1, device=GtG.device, dtype=GtG.dtype).unsqueeze(0)
        try:
            gamma = torch.linalg.solve(GtG + reg, rhs)  # [B, k-1, 1]
        except torch.linalg.LinAlgError:
            # Fallback: use last iterate as the update
            return X_hist[-1] + (G_hist[-1] - G_hist[-2]) if len(G_hist) >= 2 else X_hist[-1]

        gamma = gamma.squeeze(-1)  # [B, k-1]
        gamma_sum = gamma.sum(dim=-1, keepdim=True)
        alpha = torch.cat([1.0 - gamma_sum, gamma], dim=-1)  # [B, k]

        X_flat = X.reshape(-1, cd, k)
        x_new = (X_flat * alpha.unsqueeze(1)).sum(dim=-1)
        g_new = (G_flat * alpha.unsqueeze(1)).sum(dim=-1)
        x_new = x_new + beta * g_new

        return x_new.reshape(*flat_shape, cd)

    @staticmethod
    def backward(ctx, grad_output: Optional[torch.Tensor]):
        """Implicit Function Theorem backward.

        dL/dx_in = -(I - df/dx*)^-T * dL/dx_out

        Uses damped fixed-point iteration for the inverse:
        g = -(I - J^T)^{-1} * g_out  where J = df/dx*
        Solved via: g_{k+1} = J^T * g_k - g_out

        Also computes gradients w.r.t. f's parameters via:
        dL/dθ = g^T * df(x*, θ)/dθ
        """
        if grad_output is None:
            return None, None, None, None, None, None

        f = ctx.f
        params = ctx.params
        x_star, = ctx.saved_tensors
        if not x_star.requires_grad:
            x_star = x_star.detach().requires_grad_()

        with torch.enable_grad():
            fx_star = f(x_star)

            # --- Solve for adjoint g = dL/d(x_star) via IFT ---
            # Damped iteration: g_{k+1} = J^T * g_k - grad_output
            # Normalize gradient to prevent blowup when spectral radius > 1
            g_out = grad_output.detach().clone()
            g_out_norm = g_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            g = -g_out
            for _ in range(10):
                Jtg = torch.autograd.grad(
                    fx_star, x_star, g, create_graph=False, retain_graph=True
                )[0]
                if Jtg is None:
                    break
                g = Jtg - g_out
                # Clip g to prevent numerical blowup
                g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = (g_out_norm * 10.0) / g_norm
                scale = scale.clamp(max=1.0)  # only scale down, never up
                g = g * scale

            # --- Compute parameter gradients via VJP: g^T * df/dθ ---
            if params:
                param_grads = torch.autograd.grad(
                    fx_star, params, grad_outputs=g,
                    retain_graph=False, allow_unused=True
                )
                for p, pg in zip(params, param_grads):
                    if pg is not None:
                        if p.grad is None:
                            p.grad = pg.detach().contiguous()
                        else:
                            p.grad.add_(pg.detach())

        return None, g, None, None, None, None


def _solve_fixed_point(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-4,
    anderson_m: int = 5,
    anderson_beta: float = 1.0,
) -> torch.Tensor:
    """Solve x* = f(x*) with DEQ autograd (constant memory).

    Uses custom autograd Function for implicit differentiation.
    """
    return DEQFixedPoint.apply(f, x0, max_iter, tol, anderson_m, anderson_beta)


# ========================================================================
#  Clifford Attractor Config
# ========================================================================


@dataclass
class CliffordAttractorConfig:
    """Configuration for the Clifford Attractor.

    Attributes:
        p, q, r: Clifford algebra signature Cl(p,q,r).
        channels: Number of multivector channels.
        hidden_channels: Hidden channel count (default = channels * 2).
        num_blocks: Depth of the fixed-point map f.
        num_rotors: Number of rotor heads.
        use_blade_selector: Enable per-grade gating.
        use_geometric_activation: Enable GeometricGELU.
        max_iter: Maximum fixed-point iterations.
        tol: Convergence tolerance.
        anderson_m: Anderson memory (0 = plain fixed-point).
        output_mode: 'scalar' extracts grade-0, 'linear' projects full MV.
        init_std: Rotor weight init std.
    """
    p: int = 3
    q: int = 0
    r: int = 0
    channels: int = 32
    hidden_channels: Optional[int] = None
    num_blocks: int = 4
    num_rotors: int = 4
    use_blade_selector: bool = True
    use_geometric_activation: bool = True
    max_iter: int = 50
    tol: float = 1e-4
    anderson_m: int = 5
    output_mode: str = "scalar"
    init_std: float = 0.01
    max_seq_len: int = 512
    use_sequence_mixer: bool = True


# ========================================================================
#  Clifford Attractor Model
# ========================================================================


class CliffordAttractor(nn.Module):
    """Geometric Clifford/Rotor-based Attractor Model.

    Architecture:
        1. Token → multivector embedding.
        2. Fixed-point solve: X* = f(X*) with batched Anderson acceleration
           and implicit differentiation for gradients.
        3. Multivector → scalar output projection.

    The fixed-point map f is a stack of CliffordAttractorBlocks using
    rotor sandwiches, geometric products, and geometric activations.
    """

    def __init__(self, config: CliffordAttractorConfig, vocab_size: int = 11):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Clifford algebra (shared across all sub-modules)
        self.algebra = CliffordAlgebra(config.p, config.q, config.r)

        # Token → multivector embedding
        self.token_embed = nn.Embedding(vocab_size, config.channels * self.algebra.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.channels * self.algebra.dim)
        self.use_sequence_mixer = config.use_sequence_mixer
        if self.use_sequence_mixer:
            self.sequence_mixer = nn.GRU(
                input_size=config.channels * self.algebra.dim,
                hidden_size=config.channels * self.algebra.dim,
                batch_first=True,
            )
            self.sequence_gate = nn.Parameter(torch.tensor(0.5))
        self.output_gate = nn.Parameter(torch.tensor(-2.0))

        # Fixed-point blocks (stacked to form f)
        hidden = config.hidden_channels or (config.channels * 2)
        self.blocks = nn.ModuleList([
            CliffordAttractorBlock(
                self.algebra, config.channels, hidden,
                use_geometric_activation=config.use_geometric_activation,
                use_blade_selector=config.use_blade_selector,
                init_std=config.init_std,
            )
            for _ in range(config.num_blocks)
        ])

        # Output projection: multivector → logits
        self.output_proj = nn.Linear(config.channels * self.algebra.dim, vocab_size)

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        """The fixed-point function f applied to all positions in parallel.

        Args:
            x: [B, C*dim] flattened multivectors (batch over all positions).

        Returns:
            f(x) with same shape.
        """
        # Reshape to [B, C, dim]
        x_mv = x.view(-1, self.config.channels, self.algebra.dim)
        for block in self.blocks:
            x_mv = block(x_mv)
        return x_mv.reshape(x.shape)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_solver_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass with batched fixed-point solve across all positions.

        Args:
            input_ids: [B, seq_len] token indices.
            return_solver_stats: If True, return solver metadata.

        Returns:
            logits: [B, seq_len, vocab_size].
            Optionally, dict with solver stats.
        """
        B, S = input_ids.shape
        if S > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len={self.config.max_seq_len}. "
                "Increase CliffordAttractorConfig.max_seq_len for longer contexts."
            )

        # Embed tokens → multivectors [B, S, C*dim]
        emb = self.token_embed(input_ids)
        pos = torch.arange(S, device=input_ids.device, dtype=torch.long)
        emb = emb + self.pos_embed(pos).unsqueeze(0)

        if self.use_sequence_mixer:
            mixed, _ = self.sequence_mixer(emb)
            mix_gate = torch.sigmoid(self.sequence_gate)
            emb = mix_gate * mixed + (1.0 - mix_gate) * emb

        # Flatten batch and sequence for batched solve
        x0 = emb.reshape(B * S, -1)  # [B*S, C*dim]

        cfg = self.config
        x_star = _solve_fixed_point(
            self._f, x0,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            anderson_m=cfg.anderson_m,
        )

        # Project to vocab logits [B*S, vocab_size]
        output_mix = torch.sigmoid(self.output_gate)
        x_out = output_mix * x_star + (1.0 - output_mix) * x0
        logits = self.output_proj(x_out)  # [B*S, vocab_size]
        logits = logits.view(B, S, -1)  # [B, S, vocab_size]

        if return_solver_stats:
            return logits, {}
        return logits


# ========================================================================
#  Factory function
# ========================================================================


def create_clifford_attractor(
    p: int = 3,
    q: int = 0,
    channels: int = 32,
    num_blocks: int = 4,
    vocab_size: int = 11,
    **kwargs,
) -> CliffordAttractor:
    """Create a CliffordAttractor with given parameters."""
    config = CliffordAttractorConfig(p=p, q=q, channels=channels, num_blocks=num_blocks, **kwargs)
    return CliffordAttractor(config, vocab_size=vocab_size)
