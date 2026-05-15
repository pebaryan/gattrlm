"""
Conformal Geometric Algebra (CGA) utilities for Cl(4,1).

Provides functions to:
  - Embed 3D points into the CGA null cone
  - Construct sphere/plane/circle/line/point-pair multivectors
  - Compute intersections (meet) of geometric objects
  - Construct CGA rotors (translation, rotation, screw motions)

All operations build on the generic CliffordAlgebra class in clifford_attractor.py.

Cl(4,1) CGA conventions:
  - Basis: e1, e2, e3 (Euclidean), e4 (+ive norm), e5 (-ive norm)
  - Null vectors: e0 = (e5 - e4)/2 (origin), einf = e5 + e4 (infinity)
  - Point embedding: P(x) = e0 + x + 0.5*|x|^2 * einf
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .clifford_attractor import CliffordAlgebra


# ========================================================================
#  Precomputed outer product table
# ========================================================================


def build_outer_product_table(algebra: CliffordAlgebra) -> torch.Tensor:
    """Build the outer (wedge) product Cayley table for Cl(p,q,r).

    For basis blades: a ∧ b = 0 if a & b (overlap), else a ∧ b = gp(a,b).
    The table stores only grade-increasing contributions.

    Returns T[dim, dim, dim] where T[i,j,k] = coeff of blade k in
    outer_product(basis_i, basis_j), or 0 if the outer product vanishes.
    """
    gp = algebra._gp_table
    grades = algebra._grade_index
    dim = gp.shape[0]
    op_table = torch.zeros_like(gp)
    for i in range(dim):
        gi = grades[i].item()
        for j in range(dim):
            gj = grades[j].item()
            g_target = gi + gj
            for k in range(dim):
                val = gp[i, j, k].item()
                if val != 0 and grades[k].item() == g_target:
                    op_table[i, j, k] = float(val)
    return op_table


# ========================================================================
#  Null basis utilities
# ========================================================================


def _ensure_cga(algebra: CliffordAlgebra) -> None:
    """Verify algebra is Cl(4,1)."""
    if algebra.p != 4 or algebra.q != 1 or algebra.r != 0:
        raise ValueError(f"CGA requires Cl(4,1), got Cl({algebra.p},{algebra.q},{algebra.r})")


def cga_basis(algebra: CliffordAlgebra) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the CGA null basis multivectors (e0, einf) for Cl(4,1).

    e0 = (e5 - e4) / 2  (point at origin)
    einf = e5 + e4       (point at infinity)

    Returns:
        e0_mv: [dim] tensor representing e0.
        einf_mv: [dim] tensor representing einf.
    """
    _ensure_cga(algebra)
    device = algebra._grade_index.device
    dtype = torch.float32
    dim = algebra.dim

    idx_e4 = 8   # bit 3
    idx_e5 = 16  # bit 4

    e0 = torch.zeros(dim, device=device, dtype=dtype)
    einf = torch.zeros(dim, device=device, dtype=dtype)

    e0[idx_e4] = -0.5
    e0[idx_e5] = 0.5
    einf[idx_e4] = 1.0
    einf[idx_e5] = 1.0

    return e0, einf


# ========================================================================
#  Point embedding
# ========================================================================


def embed_point(algebra: CliffordAlgebra, x: torch.Tensor) -> torch.Tensor:
    """Embed a 3D point into the CGA null cone.

    P(x) = e0 + x + 0.5 * |x|^2 * einf

    Args:
        algebra: CliffordAlgebra(4, 1).
        x: [..., 3] Euclidean coordinates (x1, x2, x3).

    Returns:
        [..., dim] CGA point (null vector, grade 1).
    """
    _ensure_cga(algebra)
    e0_mv, einf_mv = cga_basis(algebra)
    *batch_dims, _ = x.shape
    dim = algebra.dim
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)

    pt = e0_mv.expand(*batch_dims, dim).contiguous()
    pt = pt + 0.5 * x_sq * einf_mv.expand(*batch_dims, dim)

    # Fill Euclidean components at blades e1(1), e2(2), e3(4)
    pt[..., 1] = pt[..., 1] + x[..., 0]
    pt[..., 2] = pt[..., 2] + x[..., 1]
    pt[..., 4] = pt[..., 4] + x[..., 2]

    return pt


def extract_euclidean(algebra: CliffordAlgebra, pt: torch.Tensor) -> torch.Tensor:
    """Extract the Euclidean coordinates from a CGA point.

    Args:
        algebra: CliffordAlgebra(4, 1).
        pt: [..., dim] CGA multivector.

    Returns:
        [..., 3] Euclidean coordinates.
    """
    return torch.stack([pt[..., 1], pt[..., 2], pt[..., 4]], dim=-1)


def squared_distance(
    algebra: CliffordAlgebra, p1: torch.Tensor, p2: torch.Tensor
) -> torch.Tensor:
    """Squared Euclidean distance between two CGA points.

    |x - y|^2 = -2 * <P(x) * P(y)>_0

    Args:
        p1, p2: [..., dim] CGA points (via embed_point).

    Returns:
        [...] squared distance.
    """
    gp = algebra.geometric_product(p1, p2)
    return -2.0 * gp[..., 0]


# ========================================================================
#  Geometric object embeddings
# ========================================================================


def embed_sphere(
    algebra: CliffordAlgebra, center: torch.Tensor, radius: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Embed a sphere as a CGA grade-1 vector.

    S = P(c) - 0.5 * r^2 * einf

    Args:
        algebra: CliffordAlgebra(4, 1).
        center: [..., 3] center coordinates.
        radius: scalar radius (float or tensor).

    Returns:
        [..., dim] sphere multivector.
    """
    _ensure_cga(algebra)
    _, einf_mv = cga_basis(algebra)
    *batch_dims, _ = center.shape
    dim = algebra.dim

    pt_center = embed_point(algebra, center)
    r_sq = (radius ** 2) if isinstance(radius, torch.Tensor) else (float(radius) ** 2)

    if isinstance(r_sq, torch.Tensor):
        while r_sq.dim() < len(batch_dims):
            r_sq = r_sq.unsqueeze(-1)

    return pt_center - 0.5 * r_sq * einf_mv.expand(*batch_dims, dim)


def embed_plane(
    algebra: CliffordAlgebra, normal: torch.Tensor, distance: torch.Tensor
) -> torch.Tensor:
    """Embed a plane as a CGA grade-1 vector.

    pi = n + d * einf

    where n is the unit normal (in e1,e2,e3) and d is signed distance.

    Args:
        algebra: CliffordAlgebra(4, 1).
        normal: [..., 3] unit normal.
        distance: [...] signed distance from origin.

    Returns:
        [..., dim] plane multivector.
    """
    _ensure_cga(algebra)
    _, einf_mv = cga_basis(algebra)
    *batch_dims, _ = normal.shape
    dim = algebra.dim

    pi = torch.zeros(*batch_dims, dim, device=normal.device, dtype=normal.dtype)
    pi[..., 1] = normal[..., 0]
    pi[..., 2] = normal[..., 1]
    pi[..., 4] = normal[..., 2]

    if distance.dim() < len(batch_dims):
        distance = distance.unsqueeze(-1)
    while distance.dim() < len(batch_dims) + 1:
        distance = distance.unsqueeze(-1)

    return pi + distance * einf_mv.expand(*batch_dims, dim)


# ========================================================================
#  CGA Rotors
# ========================================================================


def translation_rotor(algebra: CliffordAlgebra, t: torch.Tensor) -> torch.Tensor:
    """CGA translation rotor.

    T = exp(-t * einf / 2) = 1 - t * einf / 2

    Since (t * einf)^2 = 0 for Euclidean t, the exp terminates at linear term.

    Args:
        algebra: CliffordAlgebra(4, 1).
        t: [..., 3] translation vector.

    Returns:
        [..., dim] rotor (scalar + bivector).
    """
    _ensure_cga(algebra)
    *batch_dims, _ = t.shape
    dim = algebra.dim
    device = t.device

    # t * einf = t1 * e1*(e5+e4) + t2 * e2*(e5+e4) + t3 * e3*(e5+e4)
    # = t1*(e1e4 + e1e5) + t2*(e2e4 + e2e5) + t3*(e3e4 + e3e5)
    t_einf = torch.zeros(*batch_dims, dim, device=device, dtype=t.dtype)

    # e1e4: bits 0|3 = 1|8 = 9
    # e1e5: bits 0|4 = 1|16 = 17
    # e2e4: bits 1|3 = 2|8 = 10
    # e2e5: bits 1|4 = 2|16 = 18
    # e3e4: bits 2|3 = 4|8 = 12
    # e3e5: bits 2|4 = 4|16 = 20
    biv_indices = [
        (9, 17),   # e1*e4, e1*e5
        (10, 18),  # e2*e4, e2*e5
        (12, 20),  # e3*e4, e3*e5
    ]
    for i, (b4, b5) in enumerate(biv_indices):
        t_einf[..., b4] = t[..., i]
        t_einf[..., b5] = t[..., i]

    one = torch.zeros(*batch_dims, dim, device=device, dtype=t.dtype)
    one[..., 0] = 1.0
    return one - 0.5 * t_einf


def rotation_rotor(
    algebra: CliffordAlgebra, angle: torch.Tensor, plane: str = "e12"
) -> torch.Tensor:
    """Euclidean rotation rotor in CGA.

    R = exp(-B/2) where B = angle * unit_bivector in e1,e2,e3.

    Args:
        algebra: CliffordAlgebra(4, 1).
        angle: [...] rotation angle.
        plane: 'e12' (bits 0,1 → idx 3), 'e13' (bits 0,2 → idx 5), 'e23' (bits 1,2 → idx 6).

    Returns:
        [..., dim] rotor.
    """
    _ensure_cga(algebra)
    plane_map = {"e12": 3, "e13": 5, "e23": 6}
    biv_idx = plane_map.get(plane, 3)

    dim = algebra.dim
    if isinstance(angle, torch.Tensor):
        B = torch.zeros(*angle.shape, dim, device=angle.device, dtype=angle.dtype)
        B[..., biv_idx] = angle
    else:
        B = torch.zeros(dim)
        B[biv_idx] = float(angle)

    return algebra.exp_bivector(-0.5 * B)


def screw_rotor(
    algebra: CliffordAlgebra, t: torch.Tensor, angle: torch.Tensor, plane: str = "e12"
) -> torch.Tensor:
    """Screw rotor (translation + rotation) in CGA.

    M = T * R

    Args:
        t: [..., 3] translation.
        angle: [...] rotation angle.
        plane: Rotation plane.

    Returns:
        [..., dim] screw rotor.
    """
    T = translation_rotor(algebra, t)
    R = rotation_rotor(algebra, angle, plane)
    return algebra.geometric_product(T, R)


# ========================================================================
#  Dual, Outer Product, and Meet
# ========================================================================


def _pseudoscalar_inv(algebra: CliffordAlgebra) -> torch.Tensor:
    """I^-1 for Cl(4,1). I = e1e2e3e4e5, I^2 = -1, so I^-1 = -I."""
    d = algebra.dim
    dev = algebra._grade_index.device
    I_inv = torch.zeros(d, device=dev, dtype=torch.float32)
    I_inv[d - 1] = -1.0
    return I_inv


def dual(algebra: CliffordAlgebra, x: torch.Tensor) -> torch.Tensor:
    """Dual: x* = x * I^-1."""
    I_inv = _pseudoscalar_inv(algebra).to(device=x.device, dtype=x.dtype)
    return algebra.geometric_product(x, I_inv)


def outer_product(
    algebra: CliffordAlgebra, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Outer (wedge) product using a precomputed grade-preserving table.

    Builds the table lazily and caches it on the algebra instance.
    """
    if not hasattr(algebra, '_op_table') or algebra._op_table is None:
        object.__setattr__(algebra, '_op_table', build_outer_product_table(algebra))
    op = algebra._op_table.to(device=x.device, dtype=x.dtype)
    return torch.einsum("...i,...j,ijk->...k", x, y, op)


def meet(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Meet (intersection) of two CGA grade-1 objects.

    For CGA Cl(4,1), the meet of two grade-1 objects (spheres, planes, points)
    is computed as the grade-3 projection of (A * B * I^-1):

        A ∨ B = <A * B * I^{-1}>_3

    The result is a circle or line (grade 3 in CGA). Note: this differs from
    the general formula meet(A, B) = dual(dual(A) wedge dual(B)) because in
    Cl(4,1), the duals of grade-1 objects are grade 4, and their wedge product
    would be grade 8 (> 5 dimensions), thus vanishing.

    Examples:
        Circle = meet(Sphere, Sphere) or meet(Sphere, Plane)
        Line = meet(Plane, Plane)
    """
    I_inv = _pseudoscalar_inv(algebra).to(device=A.device, dtype=A.dtype)
    AB = algebra.geometric_product(A, B)
    return algebra.grade_projection(algebra.geometric_product(AB, I_inv), 3)


# ========================================================================
#  Factory
# ========================================================================


def create_cga_attractor(channels: int = 32, num_blocks: int = 4, vocab_size: int = 11, **kwargs):
    """Create a CliffordAttractor configured for Cl(4,1) CGA."""
    from .clifford_attractor import CliffordAttractor, CliffordAttractorConfig
    config = CliffordAttractorConfig(p=4, q=1, channels=channels, num_blocks=num_blocks, **kwargs)
    return CliffordAttractor(config, vocab_size=vocab_size)
