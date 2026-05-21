"""Cl(p,q,r) geometric algebra kernel: Cayley tables, grade ops, rotor exp."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn


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
        i_idx, j_idx, k_idx, signs: [nnz] tensors.
    """
    nnz_mask = table.abs() > 0
    indices = torch.where(nnz_mask)
    i_idx, j_idx, k_idx = indices
    signs = table[i_idx, j_idx, k_idx]
    return i_idx, j_idx, k_idx, signs


def _filter_sparse_op(
    i_idx: torch.Tensor, j_idx: torch.Tensor, k_idx: torch.Tensor,
    signs: torch.Tensor, grade_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep outer-product triples: grade(k) == grade(i) + grade(j)."""
    gi = grade_index[i_idx]
    gj = grade_index[j_idx]
    gk = grade_index[k_idx]
    mask = gk == gi + gj
    return i_idx[mask], j_idx[mask], k_idx[mask], signs[mask]


def _filter_sparse_ip(
    i_idx: torch.Tensor, j_idx: torch.Tensor, k_idx: torch.Tensor,
    signs: torch.Tensor, grade_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep Hestenes inner-product triples: grade(k) == |grade(i) - grade(j)|."""
    gi = grade_index[i_idx]
    gj = grade_index[j_idx]
    gk = grade_index[k_idx]
    mask = gk == (gi - gj).abs()
    return i_idx[mask], j_idx[mask], k_idx[mask], signs[mask]


def _grade_index_map(n: int) -> torch.Tensor:
    """[dim] tensor with grade of each basis blade."""
    return torch.tensor([i.bit_count() for i in range(1 << n)], dtype=torch.long)


def _blade_metric_signs_precomputed(p: int, q: int, r: int) -> torch.Tensor:
    """Metric sign for each basis blade in Cl(p,q,r).

    If a blade contains k negative-norm vectors, its metric sign is (-1)^k.
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

        # Sparse GP indices: O(dim^2) vs dense einsum O(dim^3).
        i_idx, j_idx, k_idx, signs = _extract_sparse_gp(gp_table)
        self.register_buffer("_gp_i", i_idx)
        self.register_buffer("_gp_j", j_idx)
        self.register_buffer("_gp_k", k_idx)
        self.register_buffer("_gp_sign", signs)

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

    # --- Core products ---

    def _dense_geometric_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gp = self._gp_table.to(device=x.device, dtype=x.dtype)
        return torch.einsum("...i,...j,ijk->...k", x, y, gp)

    def _dense_filtered_table(
        self, *,
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

    # On CPU, einsum (MKL) is faster for dim <= 128. On GPU with @torch.compile,
    # sparse becomes beneficial around dim >= 32. Set on a subclass to override.
    _USE_SPARSE_GP = False

    def _sparse_contract(
        self, x: torch.Tensor, y: torch.Tensor,
        i_idx: torch.Tensor, j_idx: torch.Tensor,
        k_idx: torch.Tensor, signs: torch.Tensor,
    ) -> torch.Tensor:
        """Generic sparse contraction: result[k] += x[i] * y[j] * sign."""
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
        """Geometric product x * y. Supports arbitrary batch dims."""
        if self._USE_SPARSE_GP:
            return self._sparse_contract(
                x, y, self._gp_i, self._gp_j, self._gp_k, self._gp_sign
            )
        return self._dense_geometric_product(x, y)

    def outer_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Outer (wedge) product: grade-raising part of the geometric product."""
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
        """Hestenes inner product (grade-lowering)."""
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

    def sandwich_product(self, left: torch.Tensor, x: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
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
        return self._metric_signs.to(device=device, dtype=dtype)

    def norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """Euclidean squared norm: sum(x_i^2).

        Uses absolute metric signs so the result is always non-negative.
        Clamps to 1e-16 minimum to prevent sqrt gradient blowup (inf * 0 = NaN).
        """
        signs = self.blade_metric_signs(x.device, x.dtype).abs()
        return (x ** 2 * signs).sum(dim=-1, keepdim=True).clamp(min=1e-16)

    # --- Bivector exponential ---

    def exp_bivector(self, biv: torch.Tensor) -> torch.Tensor:
        """Exponential of a bivector in Cl(p,q).

        If B^2 < 0 (elliptic): exp(B) = cos|B| + (B/|B|) sin|B|
        If B^2 > 0 (hyperbolic): exp(B) = cosh|B| + (B/|B|) sinh|B|

        Uses Taylor expansion for small theta to avoid 1/theta gradient blowup.

        Args:
            biv: [..., dim] (only grade-2 should be non-zero).

        Returns:
            Rotor: [..., dim] (scalar + bivector).
        """
        B2 = self.geometric_product(biv, biv)
        scalar_B2 = self.scalar_part(B2)

        theta_sq = scalar_B2.abs()
        theta_sq_safe = theta_sq.clamp(min=1e-16)
        theta_scalar = theta_sq_safe.sqrt()
        theta_val = theta_scalar[..., 0:1]

        is_elliptic = scalar_B2[..., 0:1] < 0
        one = torch.zeros_like(biv)
        one[..., 0:1] = 1.0

        small_angle = 1e-4
        is_small = theta_val < small_angle

        theta_clamped = theta_val.clamp(min=small_angle)
        cos_std = torch.where(is_elliptic, theta_clamped.cos(), theta_clamped.cosh())
        sin_over_theta_std = torch.where(
            is_elliptic,
            theta_clamped.sin() / theta_clamped,
            theta_clamped.sinh() / theta_clamped,
        )

        cos_taylor = torch.where(
            is_elliptic,
            1.0 - 0.5 * theta_val.square(),
            1.0 + 0.5 * theta_val.square(),
        )
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
        """Indices of grade-2 basis blades in [0, dim)."""
        return torch.where(self._grade_index == 2)[0]

    def grade_mask(self, grade: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (self._grade_index == grade).to(device=device, dtype=dtype)
