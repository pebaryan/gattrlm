"""
Cl(4,1) Conformal Geometric Algebra — Standalone Demo
=======================================================

Demonstrates CGA primitives and operations for spatial reasoning:
  - Cl(4,1) algebra construction (32 blades)
  - Null basis vectors (e₀, e∞) and their properties
  - Point embedding P(x) = e₀ + x + ½|x|² e∞ and null condition
  - Sphere and plane embedding
  - Translation, rotation, and screw rotors
  - Dual operation
  - Meet (intersection of spheres → circle, planes → line)
  - Gradients through all operations
  - Full CGA attractor model

Usage:  python examples/clifford41_cga_demo.py
"""

from __future__ import annotations

import time

import torch

from attractor.models.clifford_attractor import (
    CliffordAlgebra,
    CliffordAttractorConfig,
    CliffordAttractor,
)
from attractor.models.clifford_cga import (
    cga_basis,
    embed_point,
    extract_euclidean,
    squared_distance,
    embed_sphere,
    embed_plane,
    translation_rotor,
    rotation_rotor,
    screw_rotor,
    dual,
    meet,
    create_cga_attractor,
)


# ======================================================================
# 1.  Algebra and null basis
# ======================================================================
def algebra_basics() -> tuple[CliffordAlgebra, torch.Tensor, torch.Tensor]:
    """Construct Cl(4,1) and verify the null basis."""
    print("─" * 60)
    print("Part 1:  Cl(4,1) Algebra & Null Basis")
    print("─" * 60)

    alg = CliffordAlgebra(p=4, q=1)
    print(f"Algebra dimension: {alg.dim}  (32 blades)")

    # Grade structure
    print(f"Grade map: {alg._grade_index.tolist()}")
    print(f"  → 5 grades: scalar(1), vector(5), bivector(10), trivector(10), quadvector(5), pseudoscalar(1)")
    print()

    # ── Null basis ────────────────────────────────────────────────────
    #   e₀ = ½(e₅ - e₄)   ,   e∞ = e₄ + e₅
    #   e₀² = 0,  e∞² = 0,  e₀ · e∞ = -1
    e0, einf = cga_basis(alg)

    # Verify null condition e₀² = 0
    e0_sq = alg.geometric_product(e0.unsqueeze(0), e0.unsqueeze(0)).squeeze(0)
    eig_sq = alg.geometric_product(einf.unsqueeze(0), einf.unsqueeze(0)).squeeze(0)
    print(f"e₀²  = {e0_sq[0]:+.2e}    (should be ≈ 0)")
    print(f"e∞²  = {eig_sq[0]:+.2e}    (should be ≈ 0)")

    # Verify inner product: e₀ · e∞ = -1
    #   a · b = ½(a*b + b*a)  →  scalar part = -½|x|² for points
    e0_einf = alg.geometric_product(e0.unsqueeze(0), einf.unsqueeze(0)).squeeze(0)
    einf_e0 = alg.geometric_product(einf.unsqueeze(0), e0.unsqueeze(0)).squeeze(0)
    dot = 0.5 * (e0_einf + einf_e0)
    print(f"e₀ · e∞  = {dot[0]:+.4f}  (should be -1.0)")
    print()

    return alg, e0, einf


# ======================================================================
# 2.  Point embedding
# ======================================================================
def point_demo(alg: CliffordAlgebra) -> None:
    """Point embedding P(x) = e₀ + x + ½|x|² e∞."""
    print("─" * 60)
    print("Part 2:  Point Embedding")
    print("─" * 60)

    # Embed a single point
    x_3d = torch.tensor([1.0, 2.0, 3.0])
    pt = embed_point(alg, x_3d)
    print(f"P({x_3d.tolist()}) → multivector of shape {pt.shape}")

    # Null condition: P(x)² = 0
    pt_sq = alg.geometric_product(pt.unsqueeze(0), pt.unsqueeze(0)).squeeze(0)
    print(f"P(x)²  = {pt_sq[0]:+.4e}    (should be ≈ 0, float32 precision)")

    # Extract Euclidean coordinates back
    x_recovered = extract_euclidean(alg, pt)
    print(f"Recovered x  = {x_recovered.tolist()}")
    recon_error = (x_3d - x_recovered).norm().item()
    print(f"Reconstruction error: {recon_error:.2e}")
    print()

    # ── Squared distance via inner product ────────────────────────────
    #   P(a) · P(b) = -½ |a - b|²
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([3.0, 4.0, 0.0])
    pa = embed_point(alg, a)
    pb = embed_point(alg, b)
    d = squared_distance(alg, pa, pb)
    true_d = (a - b).norm().item()
    print(f"squared_distance(P(0), P(3,4,0)) = {d.item():.4f}  (expected {true_d:.1f}² = {true_d**2:.1f})")
    print()

    # ── Batch of points ───────────────────────────────────────────────
    batch = torch.randn(4, 3) * 2.0
    pts = embed_point(alg, batch)
    print(f"Batch of 4 points: shape {pts.shape}")
    nulls = alg.geometric_product(pts, pts)[:, 0]
    print(f"P(x)² values: {nulls.tolist()}  (all ≈ 0)")
    print()


# ======================================================================
# 3.  Sphere & Plane
# ======================================================================
def sphere_plane_demo(alg: CliffordAlgebra) -> None:
    """Sphere and plane embedding."""
    print("─" * 60)
    print("Part 3:  Spheres & Planes")
    print("─" * 60)

    # ── Sphere ────────────────────────────────────────────────────────
    #   S(c, r) = P(c) - ½ r² e∞
    sphere = embed_sphere(alg, torch.tensor([0.0, 0.0, 0.0]), radius=2.0)
    sphere_sq = alg.geometric_product(sphere.unsqueeze(0), sphere.unsqueeze(0)).squeeze(0)
    print(f"Sphere(0, r=2)² = {sphere_sq[0]:+.4f}  (should be r² = 4.0)")

    # A point is a sphere of radius 0
    pt_as_sphere = embed_sphere(alg, torch.tensor([1.0, 0.0, 0.0]), radius=0.0)
    pt = embed_point(alg, torch.tensor([1.0, 0.0, 0.0]))
    diff = (pt_as_sphere - pt).norm().item()
    print(f"P(x) == Sphere(x, r=0):  diff = {diff:.2e}")
    print()

    # ── Plane ─────────────────────────────────────────────────────────
    #   π(n, d) = n + d e∞
    plane = embed_plane(alg, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0]))
    plane_sq = alg.geometric_product(plane.unsqueeze(0), plane.unsqueeze(0)).squeeze(0)
    print(f"Plane(z=0)² = {plane_sq[0]:+.4f}  (should be 1.0)")
    print()


# ======================================================================
# 4.  CGA Rotors
# ======================================================================
def rotor_demo(alg: CliffordAlgebra) -> None:
    """Translation, rotation, and screw rotors."""
    print("─" * 60)
    print("Part 4:  CGA Rotors (Rigid Motions)")
    print("─" * 60)

    pt = embed_point(alg, torch.tensor([1.0, 0.0, 0.0]))

    # ── Translation ───────────────────────────────────────────────────
    #   T(t) = 1 - ½ t e∞   →   T*P(x)*T̃ = P(x + t)
    T = translation_rotor(alg, torch.tensor([2.0, 3.0, 0.0]))
    pt_translated = alg.geometric_product(
        alg.geometric_product(T, pt), alg.reverse(T)
    )
    x_recovered = extract_euclidean(alg, pt_translated)
    print(f"Translate P(1,0,0) by (2,3,0) → {x_recovered.tolist()}")

    # ── Rotation ──────────────────────────────────────────────────────
    #   R(θ, n) = cos(θ/2) + n sin(θ/2)
    R = rotation_rotor(alg, angle=torch.tensor(torch.pi / 2), plane="e12")
    pt_rotated = alg.geometric_product(
        alg.geometric_product(R, pt), alg.reverse(R)
    )
    x_recovered = extract_euclidean(alg, pt_rotated)
    print(f"Rotate P(1,0,0) by π/2 around e₁₂ → {x_recovered.tolist()}  (≈ (0,1,0))")

    # ── Screw (rotation + translation combined) ───────────────────────
    M = screw_rotor(alg, t=torch.tensor([0.0, 0.0, 2.0]),
                    angle=torch.tensor(torch.pi), plane="e12")
    pt_screwed = alg.geometric_product(
        alg.geometric_product(M, pt), alg.reverse(M)
    )
    x_recovered = extract_euclidean(alg, pt_screwed)
    print(f"Screw: translate(0,0,2) + rotate π → {x_recovered.tolist()}  (≈ (-1,0,2))")
    print()

    # ── Check rotor norm ──────────────────────────────────────────────
    for name, rotor in [("Translation", T), ("Rotation", R), ("Screw", M)]:
        # RR̃ should be 1 (scalar part = 1)
        rr = alg.geometric_product(rotor.unsqueeze(0), alg.reverse(rotor).unsqueeze(0)).squeeze(0)
        print(f"{name}:  RR̃ scalar = {rr[0]:+.4f}  (should be 1.0)")
    print()


# ======================================================================
# 5.  Dual & Meet
# ======================================================================
def meet_demo(alg: CliffordAlgebra) -> None:
    """Dual and meet (intersection) operations."""
    print("─" * 60)
    print("Part 5:  Dual & Meet Operations")
    print("─" * 60)

    # ── Dual ──────────────────────────────────────────────────────────
    #   dual(A) = A * I⁻¹
    #   In Cl(4,1) the pseudoscalar I² = -1, so I⁻¹ = -I
    sphere = embed_sphere(alg, torch.tensor([0.0, 0.0, 0.0]), radius=1.0)
    sphere_dual = dual(alg, sphere)
    print(f"Dual of sphere: non-zero entries = {(sphere_dual.abs() > 1e-6).sum().item()}")
    print()

    # ── Meet: sphere ∩ sphere = circle ────────────────────────────────
    s1 = embed_sphere(alg, torch.tensor([0.0, 0.0, 0.0]), radius=2.0)
    s2 = embed_sphere(alg, torch.tensor([1.0, 0.0, 0.0]), radius=2.0)
    circle = meet(alg, s1, s2)
    print(f"Meet of two spheres (circle):")
    print(f"  non-zero entries = {(circle.abs() > 1e-6).sum().item()}  (should be > 0)")
    print()

    # ── Meet: plane ∩ plane = line ────────────────────────────────────
    p1 = embed_plane(alg, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0]))
    p2 = embed_plane(alg, torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0]))
    line = meet(alg, p1, p2)
    print(f"Meet of two orthogonal planes (line):")
    print(f"  non-zero entries = {(line.abs() > 1e-6).sum().item()}  (should be > 0)")
    print()


# ======================================================================
# 6.  Gradient check
# ======================================================================
def gradient_demo(alg: CliffordAlgebra) -> None:
    """Verify all operations support backward gradients."""
    print("─" * 60)
    print("Part 6:  Gradient Check")
    print("─" * 60)

    # Point embedding with requires_grad
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    pt = embed_point(alg, x)
    loss = pt.sum()
    loss.backward()
    print(f"Point embedding gradient:  d(sum(P(x))) / dx  at (1,2,3)")
    print(f"  grad = {x.grad.tolist()}")
    print(f"  finite = {torch.isfinite(x.grad).all().item()}")
    print()

    # Rotor sandwich with gradient through all parameters
    t = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
    pt2 = embed_point(alg, torch.tensor([1.0, 0.0, 0.0]))
    T = translation_rotor(alg, t)
    pt_moved = alg.geometric_product(
        alg.geometric_product(T, pt2), alg.reverse(T)
    )
    loss2 = pt_moved.sum()
    loss2.backward()
    print(f"Translation rotor gradient:")
    print(f"  grad(d translation vector) = {t.grad.tolist()}")
    print(f"  finite = {torch.isfinite(t.grad).all().item()}")
    print()

    # Meet gradient
    c = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    s1 = embed_sphere(alg, c, radius=2.0)
    s2 = embed_sphere(alg, torch.tensor([1.0, 0.0, 0.0]), radius=2.0)
    circle = meet(alg, s1, s2)
    loss3 = circle.sum()
    loss3.backward()
    print(f"Meet gradient (w.r.t. sphere center):")
    print(f"  grad = {c.grad.tolist()}")
    print(f"  finite = {torch.isfinite(c.grad).all().item()}")
    print()


# ======================================================================
# 7.  Full CGA Attractor model
# ======================================================================
def cga_attractor_demo() -> None:
    """Build and run a full CGA DEQ model."""
    print("─" * 60)
    print("Part 7:  Full CGA Attractor Model")
    print("─" * 60)

    # Option A: factory function
    model = create_cga_attractor(
        channels=16,
        num_blocks=1,
        vocab_size=11,
        max_iter=10,
    )
    tokens = torch.randint(0, 10, (2, 9))
    logits = model(tokens)
    print(f"create_cga_attractor  →  output shape {logits.shape}")
    loss = logits.sum()
    loss.backward()
    print(f"  backward: all params finite = {all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)}")
    print()

    # Option B: manual construction
    cfg = CliffordAttractorConfig(
        p=4, q=1,
        channels=16,
        num_blocks=1,
        max_iter=10,
        tol=1e-4,
        anderson_m=3,
    )
    model2 = CliffordAttractor(cfg, vocab_size=11)
    logits2 = model2(tokens)
    loss2 = logits2.sum()
    loss2.backward()
    print(f"Manual CGA Attractor  →  output shape {logits2.shape}")
    print(f"  backward: all params finite = {all(torch.isfinite(p.grad).all().item() for p in model2.parameters() if p.grad is not None)}")
    print()


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Cl(4,1) Conformal Geometric Algebra Demo              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    alg, e0, einf = algebra_basics()
    point_demo(alg)
    sphere_plane_demo(alg)
    rotor_demo(alg)
    meet_demo(alg)
    gradient_demo(alg)
    cga_attractor_demo()

    print("─" * 60)
    print("All demonstrations completed successfully!")
    print("─" * 60)
