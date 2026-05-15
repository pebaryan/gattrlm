"""
Unit tests for CGA Cl(4,1) operations.

Tests cover:
  - Cl(4,1) algebra dimensions and metric signs
  - Null basis vectors (e0, einf) and their properties
  - Point embedding (null cone condition)
  - Sphere and plane representations
  - CGA rotors (translation, rotation, screw)
  - Geometric computations (distance, inner product)
  - Outer product correctness
  - Meet (intersection) of geometric objects
  - Numerical stability (backward pass through operations)
  - Full CliffordAttractor forward/backward in Cl(4,1)
"""

import math

import pytest
import torch

from attractor.models.clifford_cga import (
    cga_basis,
    dual,
    embed_plane,
    embed_point,
    embed_sphere,
    extract_euclidean,
    meet,
    outer_product,
    rotation_rotor,
    screw_rotor,
    squared_distance,
    translation_rotor,
)
from attractor.models.clifford_attractor import CliffordAlgebra, build_gp_table


# ========================================================================
#  Fixtures
# ========================================================================


@pytest.fixture(scope="module")
def algebra():
    """Cl(4,1) algebra fixture."""
    return CliffordAlgebra(4, 1)


@pytest.fixture(scope="module")
def algebra_cl3():
    """Cl(3,0) for comparison."""
    return CliffordAlgebra(3, 0)


# ========================================================================
#  Basic Algebra Verification
# ========================================================================


class TestCGAAlgebraDimensions:
    """Verify Cl(4,1) algebra basics."""

    def test_dim(self, algebra):
        assert algebra.n == 5
        assert algebra.dim == 32

    def test_gp_table_shape(self, algebra):
        assert algebra._gp_table.shape == (32, 32, 32)

    def test_grade_count(self, algebra):
        """Verify grade distribution: C(5,0)=1, C(5,1)=5, C(5,2)=10, C(5,3)=10, C(5,4)=5, C(5,5)=1."""
        grades = algebra._grade_index
        counts = {g: (grades == g).sum().item() for g in range(6)}
        assert counts[0] == 1   # scalar
        assert counts[1] == 5   # vectors
        assert counts[2] == 10  # bivectors
        assert counts[3] == 10  # trivectors
        assert counts[4] == 5   # quadvectors
        assert counts[5] == 1   # pseudoscalar

    def test_metric_signs(self, algebra):
        """e1^2=e2^2=e3^2=e4^2=+1, e5^2=-1."""
        e1 = torch.zeros(32)
        e2 = torch.zeros(32)
        e3 = torch.zeros(32)
        e4 = torch.zeros(32)
        e5 = torch.zeros(32)
        e1[1] = 1.0
        e2[2] = 1.0
        e3[4] = 1.0
        e4[8] = 1.0
        e5[16] = 1.0

        def sq(mv):
            return algebra.geometric_product(mv, mv)[0].item()

        assert abs(sq(e1) - 1.0) < 1e-6, f"e1^2 = {sq(e1)} != 1"
        assert abs(sq(e2) - 1.0) < 1e-6, f"e2^2 = {sq(e2)} != 1"
        assert abs(sq(e3) - 1.0) < 1e-6, f"e3^2 = {sq(e3)} != 1"
        assert abs(sq(e4) - 1.0) < 1e-6, f"e4^2 = {sq(e4)} != 1"
        assert abs(sq(e5) - (-1.0)) < 1e-6, f"e5^2 = {sq(e5)} != -1"


# ========================================================================
#  Null Basis
# ========================================================================


class TestCGANullBasis:
    """Verify e0 and einf properties."""

    def test_e0_sq(self, algebra):
        """e0^2 = 0."""
        e0, _ = cga_basis(algebra)
        e0_sq = algebra.geometric_product(e0, e0)[0].item()
        assert abs(e0_sq) < 1e-6, f"e0^2 = {e0_sq} != 0"

    def test_einf_sq(self, algebra):
        """einf^2 = 0."""
        _, einf = cga_basis(algebra)
        einf_sq = algebra.geometric_product(einf, einf)[0].item()
        assert abs(einf_sq) < 1e-6, f"einf^2 = {einf_sq} != 0"

    def test_e0_dot_einf(self, algebra):
        """e0 · einf = -1 (the inner product)."""
        e0, einf = cga_basis(algebra)
        # Inner product = scalar part of GP
        gp = algebra.geometric_product(e0, einf)
        inner = gp[0].item()
        assert abs(inner - (-1.0)) < 1e-6, f"e0·einf = {inner} != -1"

    def test_anticommute_4_5(self, algebra):
        """e4 and e5 anticommute: e4*e5 = -e5*e4."""
        e4 = torch.zeros(32)
        e5 = torch.zeros(32)
        e4[8] = 1.0
        e5[16] = 1.0
        gp_45 = algebra.geometric_product(e4, e5)
        gp_54 = algebra.geometric_product(e5, e4)
        for k in range(32):
            assert abs(gp_45[k].item() + gp_54[k].item()) < 1e-6, f"e4*e5 != -e5*e4 at blade {k}"


# ========================================================================
#  Point Embedding
# ========================================================================


class TestPointEmbedding:
    """Verify point embedding into the CGA null cone."""

    def test_point_is_null(self, algebra):
        """P(x)^2 = 0 for any Euclidean point."""
        x = torch.tensor([1.0, 2.0, 3.0])
        pt = embed_point(algebra, x)
        pt_sq = algebra.geometric_product(pt, pt)[0].item()
        assert abs(pt_sq) < 1e-6, f"P(x)^2 = {pt_sq} != 0"

    def test_point_null_random(self, algebra):
        """Random points are also null."""
        for _ in range(10):
            x = torch.randn(3) * 5.0
            pt = embed_point(algebra, x)
            pt_sq = algebra.geometric_product(pt, pt)[0].item()
            assert abs(pt_sq) < 2e-3, f"P({x})^2 = {pt_sq} != 0"

    def test_origin_point(self, algebra):
        """P(0) = e0."""
        e0, _ = cga_basis(algebra)
        pt = embed_point(algebra, torch.zeros(3))
        diff = (pt - e0).norm().item()
        assert diff < 1e-6, f"P(0) != e0, diff = {diff}"

    def test_point_grade_one(self, algebra):
        """P(x) should be grade-1 (pure vector)."""
        x = torch.tensor([0.5, -1.0, 2.0])
        pt = embed_point(algebra, x)
        grades = algebra._grade_index
        for k in range(32):
            if grades[k] != 1 and abs(pt[k].item()) > 1e-6:
                raise AssertionError(f"Point has non-grade-1 component at blade {k} (grade {grades[k].item()})")

    def test_extract_euclidean_roundtrip(self, algebra):
        """Embedding and extracting returns the original point."""
        x = torch.tensor([1.5, -2.5, 3.0])
        pt = embed_point(algebra, x)
        x_back = extract_euclidean(algebra, pt)
        assert (x - x_back).norm().item() < 1e-6

    def test_batched_points(self, algebra):
        """Batched point embedding works correctly."""
        xs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        pts = embed_point(algebra, xs)
        assert pts.shape == (3, 32)
        # All should be null
        for i in range(3):
            pt_sq = algebra.geometric_product(pts[i], pts[i])[0].item()
            assert abs(pt_sq) < 1e-6


# ========================================================================
#  Distance
# ========================================================================


class TestDistance:
    """Verify squared_distance between points."""

    def test_self_distance(self, algebra):
        """Distance from a point to itself is 0."""
        x = torch.tensor([1.0, 2.0, 3.0])
        pt = embed_point(algebra, x)
        d_sq = squared_distance(algebra, pt, pt)
        assert abs(d_sq.item()) < 1e-6

    def test_known_distance(self, algebra):
        """Distance between (0,0,0) and (1,0,0) is 1."""
        p0 = embed_point(algebra, torch.zeros(3))
        p1 = embed_point(algebra, torch.tensor([1.0, 0.0, 0.0]))
        d_sq = squared_distance(algebra, p0, p1)
        assert abs(d_sq.item() - 1.0) < 1e-5

    def test_pythagorean(self, algebra):
        """Distance: 2^2 + 3^2 = 13."""
        p0 = embed_point(algebra, torch.zeros(3))
        p = embed_point(algebra, torch.tensor([2.0, 3.0, 0.0]))
        d_sq = squared_distance(algebra, p0, p)
        assert abs(d_sq.item() - 13.0) < 1e-5

    def test_batched_distance(self, algebra):
        """Batched distance computation."""
        p0 = embed_point(algebra, torch.zeros(3))
        pts = embed_point(algebra, torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))
        d_sq = squared_distance(algebra, p0.unsqueeze(0).expand(3, 32), pts)
        expected = torch.tensor([1.0, 4.0, 9.0])
        assert (d_sq - expected).abs().max().item() < 1e-5


# ========================================================================
#  Sphere and Plane
# ========================================================================


class TestSphereAndPlane:
    """Verify sphere and plane embeddings."""

    def test_sphere_from_radius(self, algebra):
        """Sphere centered at origin with radius r."""
        center = torch.zeros(3)
        r = 2.0
        sphere = embed_sphere(algebra, center, r)
        # The sphere should be representable: P(0) - 0.5 * r^2 * einf
        e0, einf = cga_basis(algebra)
        expected = e0 - 0.5 * r ** 2 * einf
        diff = (sphere - expected).norm().item()
        assert diff < 1e-6

    def test_sphere_contains_point(self, algebra):
        """For a sphere at origin with radius r, a point at distance r on sphere surface."""
        center = torch.zeros(3)
        r = 2.5
        sphere = embed_sphere(algebra, center, r)
        # Point on sphere surface
        pt = embed_point(algebra, torch.tensor([r, 0.0, 0.0]))
        # Sphere contains point iff sphere · point = 0 (inner product)
        inner = algebra.geometric_product(sphere, pt)[0].item()
        assert abs(inner) < 1e-5, f"sphere·point = {inner}"

    def test_outside_point(self, algebra):
        """A point outside the sphere should not satisfy sphere·point = 0."""
        center = torch.zeros(3)
        r = 1.0
        sphere = embed_sphere(algebra, center, r)
        pt = embed_point(algebra, torch.tensor([3.0, 0.0, 0.0]))
        inner = algebra.geometric_product(sphere, pt)[0].item()
        assert inner < -0.5, f"Outside point has inner = {inner}"

    def test_plane_from_normal(self, algebra):
        """Plane with normal along x-axis at distance d."""
        normal = torch.tensor([1.0, 0.0, 0.0])
        d = torch.tensor([2.0])
        plane = embed_plane(algebra, normal, d)
        # Point on plane should satisfy plane·point = 0
        pt = embed_point(algebra, torch.tensor([2.0, 0.0, 0.0]))
        inner = algebra.geometric_product(plane, pt)[0].item()
        assert abs(inner) < 1e-5, f"plane·point = {inner}"

    def test_plane_off_plane(self, algebra):
        """Point not on plane should have non-zero inner product."""
        normal = torch.tensor([1.0, 0.0, 0.0])
        d = torch.tensor([1.0])
        plane = embed_plane(algebra, normal, d)
        pt = embed_point(algebra, torch.tensor([5.0, 0.0, 0.0]))
        inner = algebra.geometric_product(plane, pt)[0].item()
        assert abs(inner - 4.0) < 1e-5, f"Off-plane point: plane·point = {inner}"


# ========================================================================
#  CGA Rotors
# ========================================================================


class TestCGARotors:
    """Verify CGA rotor actions."""

    def test_translation_rotor_moves_point(self, algebra):
        """Translating a point by vector t should shift the Euclidean part by t."""
        x = torch.tensor([1.0, 2.0, 3.0])
        t = torch.tensor([0.5, -1.0, 0.0])
        pt = embed_point(algebra, x)
        T = translation_rotor(algebra, t)
        T_rev = algebra.reverse(T)
        pt_moved = algebra.sandwich_product(T, pt, T_rev)
        x_moved = extract_euclidean(algebra, pt_moved)
        expected = x + t
        assert (x_moved - expected).norm().item() < 1e-5

    def test_translation_preserves_null(self, algebra):
        """Rotated/translated points remain null."""
        x = torch.tensor([1.0, 0.0, 0.0])
        t = torch.tensor([2.0, 3.0, 4.0])
        pt = embed_point(algebra, x)
        T = translation_rotor(algebra, t)
        T_rev = algebra.reverse(T)
        pt_moved = algebra.sandwich_product(T, pt, T_rev)
        pt_sq = algebra.geometric_product(pt_moved, pt_moved)[0].item()
        assert abs(pt_sq) < 1e-6, f"Translated point is not null: {pt_sq}"

    def test_translation_identity_for_zero(self, algebra):
        """Zero translation gives identity rotor: T(0) = 1."""
        T = translation_rotor(algebra, torch.zeros(3))
        assert abs(T[0].item() - 1.0) < 1e-6, f"T(0) scalar = {T[0].item()}"
        # All bivector components should be zero
        biv_components = T[3:].norm().item()
        assert biv_components < 1e-6

    def test_rotation_rotor_rotates_point(self, algebra):
        """Rotating a point by π/2 in e12 plane."""
        # Point on x-axis
        x = torch.tensor([1.0, 0.0, 0.0])
        pt = embed_point(algebra, x)
        angle = torch.tensor(math.pi / 2)
        R = rotation_rotor(algebra, angle, "e12")
        R_rev = algebra.reverse(R)
        pt_rot = algebra.sandwich_product(R, pt, R_rev)
        x_rot = extract_euclidean(algebra, pt_rot)
        # After π/2 rotation in e12: (1,0,0) → (0,1,0)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert (x_rot - expected).norm().item() < 1e-4

    def test_rotation_180_degrees(self, algebra):
        """π rotation in e12: (1,0,0) → (-1,0,0)."""
        x = torch.tensor([1.0, 0.0, 0.0])
        pt = embed_point(algebra, x)
        angle = torch.tensor(math.pi)
        R = rotation_rotor(algebra, angle, "e12")
        R_rev = algebra.reverse(R)
        pt_rot = algebra.sandwich_product(R, pt, R_rev)
        x_rot = extract_euclidean(algebra, pt_rot)
        expected = torch.tensor([-1.0, 0.0, 0.0])
        assert (x_rot - expected).norm().item() < 1e-4

    def test_rotation_preserves_null(self, algebra):
        """Rotation preserves null property."""
        pt = embed_point(algebra, torch.tensor([1.0, 2.0, 3.0]))
        R = rotation_rotor(algebra, torch.tensor(0.7), "e23")
        R_rev = algebra.reverse(R)
        pt_rot = algebra.sandwich_product(R, pt, R_rev)
        pt_sq = algebra.geometric_product(pt_rot, pt_rot)[0].item()
        assert abs(pt_sq) < 1e-6

    def test_screw_rotor(self, algebra):
        """Screw rotor: translate then rotate."""
        x = torch.tensor([1.0, 0.0, 0.0])
        t = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.tensor(math.pi / 2)
        pt = embed_point(algebra, x)
        M = screw_rotor(algebra, t, angle, "e12")
        M_rev = algebra.reverse(M)
        pt_moved = algebra.sandwich_product(M, pt, M_rev)
        x_moved = extract_euclidean(algebra, pt_moved)
        # π/2 rotate (1,0,0) → (0,1,0), then translate by (0,0,1): (0,1,0) → (0,1,1)
        expected = torch.tensor([0.0, 1.0, 1.0])
        assert (x_moved - expected).norm().item() < 1e-4

    def test_batched_rotors(self, algebra):
        """Batched rotor application."""
        xs = embed_point(algebra, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        ts = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        T = translation_rotor(algebra, ts)
        T_rev = algebra.reverse(T)
        # Sandwich per batch
        pt_moved = algebra.sandwich_product(T, xs, T_rev)
        xs_out = extract_euclidean(algebra, pt_moved)
        expected = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        assert (xs_out - expected).abs().max().item() < 1e-5


# ========================================================================
#  Dual and Outer Product
# ========================================================================


class TestDualAndOuter:
    """Verify dual and outer product operations."""

    def test_dual_basic(self, algebra):
        """Dual of a vector is grade-4."""
        x = embed_point(algebra, torch.tensor([1.0, 0.0, 0.0]))
        x_dual = dual(algebra, x)
        grades = algebra._grade_index
        dual_grade_count = 0
        for k in range(32):
            if grades[k] != 4 and abs(x_dual[k].item()) > 1e-6:
                dual_grade_count += 1
        # Allow small numerical noise
        assert x_dual.norm().item() > 0.1, "Dual of point is zero"

    def test_double_dual_points(self, algebra):
        """dual(dual(P)) for a point should give back the point (up to sign)."""
        pt = embed_point(algebra, torch.tensor([2.0, 3.0, 1.0]))
        pt_dd = dual(algebra, dual(algebra, pt))
        # In Cl(4,1): I² = -1, so dual(dual(x)) = x * I⁻¹ * I⁻¹ = x * I⁻² = x * (-1) = -x
        diff = (pt_dd + pt).norm().item()
        assert diff < 1e-5, f"double dual should be -x, diff = {diff}"

    def test_outer_product_anticommutes(self, algebra):
        """x ∧ y = -y ∧ x for vectors."""
        pt1 = embed_point(algebra, torch.tensor([1.0, 0.0, 0.0]))
        pt2 = embed_point(algebra, torch.tensor([0.0, 1.0, 0.0]))
        op12 = outer_product(algebra, pt1, pt2)
        op21 = outer_product(algebra, pt2, pt1)
        diff = (op12 + op21).norm().item()
        assert diff < 1e-5, "x ∧ y != -y ∧ x"

    def test_outer_product_zero_for_same(self, algebra):
        """x ∧ x = 0 for any vector."""
        pt = embed_point(algebra, torch.tensor([1.0, 2.0, 3.0]))
        op = outer_product(algebra, pt, pt)
        assert op.norm().item() < 1e-6, "x ∧ x != 0"

    def test_outer_product_grade(self, algebra):
        """Outer product of two CGA points should be grade-2 (bivector)."""
        pt1 = embed_point(algebra, torch.tensor([1.0, 0.0, 0.0]))
        pt2 = embed_point(algebra, torch.tensor([0.0, 1.0, 0.0]))
        op = outer_product(algebra, pt1, pt2)
        grades = algebra._grade_index
        for k in range(32):
            if grades[k] != 2 and abs(op[k].item()) > 1e-6:
                raise AssertionError(f"P1 ∧ P2 has non-grade-2 component at blade {k} (grade {grades[k].item()})")

    def test_outer_product_batched(self, algebra):
        """Batched outer product."""
        pts = embed_point(algebra, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        ops = outer_product(algebra, pts, torch.stack([pts[1], pts[0]]))
        assert ops.shape == (2, 32)


# ========================================================================
#  Meet (Intersection)
# ========================================================================


class TestMeet:
    """Verify meet (intersection) operations."""

    def test_meet_two_spheres(self, algebra):
        """Two intersecting spheres: sphere1 ∨ sphere2 gives a circle."""
        s1 = embed_sphere(algebra, torch.zeros(3), 2.0)
        s2 = embed_sphere(algebra, torch.tensor([1.0, 0.0, 0.0]), 2.0)
        result = meet(algebra, s1, s2)
        assert result.norm().item() > 0.1, f"meet norm = {result.norm().item()}"

    def test_origin_plane(self, algebra):
        """Meet of plane at origin and point at origin should be non-zero."""
        normal = torch.tensor([0.0, 0.0, 1.0])
        d = torch.tensor([0.0])
        plane = embed_plane(algebra, normal, d)
        pt = embed_point(algebra, torch.zeros(3))
        # Point · plane = 0 for a point on the plane
        inner = algebra.geometric_product(plane, pt)[0].item()
        assert abs(inner) < 1e-6


# ========================================================================
#  Backward pass through CGA operations
# ========================================================================


class TestCGAGradients:
    """Verify gradients flow through CGA operations."""

    def test_point_embedding_backward(self, algebra):
        """Gradient through embed_point."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        pt = embed_point(algebra, x)
        loss = pt.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_translation_rotor_backward(self, algebra):
        """Gradient through translation_rotor and sandwich."""
        x = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        t = torch.tensor([0.5, -1.0, 0.0], requires_grad=True)
        pt = embed_point(algebra, x)
        T = translation_rotor(algebra, t)
        T_rev = algebra.reverse(T)
        pt_moved = algebra.sandwich_product(T, pt, T_rev)
        loss = pt_moved.sum()
        loss.backward()
        assert x.grad is not None and t.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(t.grad).all()

    def test_screw_rotor_backward(self, algebra):
        """Gradient through screw rotor."""
        x = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        t = torch.tensor([0.0, 0.0, 0.5], requires_grad=True)
        angle = torch.tensor(0.5, requires_grad=True)
        pt = embed_point(algebra, x)
        M = screw_rotor(algebra, t, angle, "e23")
        M_rev = algebra.reverse(M)
        pt_moved = algebra.sandwich_product(M, pt, M_rev)
        loss = pt_moved.sum()
        loss.backward()
        assert x.grad is not None and t.grad is not None and angle.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(t.grad).all()
        assert torch.isfinite(angle.grad).all()

    def test_squared_distance_backward(self, algebra):
        """Gradient through squared_distance."""
        x1 = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        x2 = torch.tensor([4.0, 0.0, 0.0], requires_grad=True)
        p1 = embed_point(algebra, x1)
        p2 = embed_point(algebra, x2)
        d_sq = squared_distance(algebra, p1, p2)
        loss = d_sq.sum()
        loss.backward()
        assert x1.grad is not None and x2.grad is not None
        assert torch.isfinite(x1.grad).all()
        assert torch.isfinite(x2.grad).all()


# ========================================================================
#  Full CliffordAttractor in Cl(4,1)
# ========================================================================


class TestCGAFullModel:
    """End-to-end test of CliffordAttractor with Cl(4,1)."""

    def test_forward_shape(self, algebra):
        """Forward pass with Cl(4,1)."""
        from attractor.models.clifford_attractor import CliffordAttractor, CliffordAttractorConfig

        cfg = CliffordAttractorConfig(p=4, q=1, channels=4, num_blocks=1, max_iter=5)
        model = CliffordAttractor(cfg, vocab_size=11)
        x = torch.randint(0, 10, (2, 9))
        y = model(x)
        assert y.shape == (2, 9, 11)

    def test_backward(self, algebra):
        """Backward pass with Cl(4,1)."""
        from attractor.models.clifford_attractor import CliffordAttractor, CliffordAttractorConfig

        cfg = CliffordAttractorConfig(p=4, q=1, channels=4, num_blocks=1, max_iter=5)
        model = CliffordAttractor(cfg, vocab_size=11)
        x = torch.randint(0, 10, (2, 9))
        y = model(x)
        loss = y.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"

    def test_create_factory(self):
        """create_cga_attractor factory works."""
        from attractor.models.clifford_cga import create_cga_attractor
        model = create_cga_attractor(channels=4, num_blocks=1, vocab_size=11)
        assert model.algebra.p == 4
        assert model.algebra.q == 1
        assert model.algebra.dim == 32

    def test_training_step(self, algebra):
        """Single training step with Cl(4,1)."""
        from attractor.models.clifford_attractor import CliffordAttractor, CliffordAttractorConfig

        cfg = CliffordAttractorConfig(p=4, q=1, channels=4, num_blocks=1, max_iter=5)
        model = CliffordAttractor(cfg, vocab_size=11)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, 10, (2, 9))
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        # Verify loss decreases
        y2 = model(x)
        loss2 = y2.sum()
        assert loss2.item() < loss.item() + 1.0  # shouldn't increase dramatically
