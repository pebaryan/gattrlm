"""
Unit tests for the Clifford Attractor module.

Tests cover:
  - Geometric product (GP) table correctness
  - Bivector exponential
  - Sandwich product (rotor action)
  - Grade operations (reverse, involution, projection)
  - Metric signs
  - Clifford algebra identities
  - Neural network layers (RotorLayer, CliffordLinear, LayerNorm, GeometricGELU)
  - DEQ fixed-point solver
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from attractor.models.clifford_attractor import (
    # Core algebra
    CliffordAlgebra,
    build_gp_table,
    _gp_blade,
    _blade_grade,
    _grade_index_map,
    _blade_metric_signs_precomputed,
    # Layers
    RotorLayer,
    CliffordLinear,
    CliffordLayerNorm,
    GeometricGELU,
    BladeSelector,
    CliffordAttractorBlock,
    # DEQ solver
    DEQFixedPoint,
    _solve_fixed_point,
    # Model
    CliffordAttractor,
    CliffordAttractorConfig,
    create_clifford_attractor,
)


# ========================================================================
#  Fixtures
# ========================================================================


@pytest.fixture
def cl3() -> CliffordAlgebra:
    """Cl(3,0) Euclidean algebra (8 blades)."""
    return CliffordAlgebra(p=3, q=0, r=0)


@pytest.fixture
def cl2() -> CliffordAlgebra:
    """Cl(2,0) Euclidean plane (4 blades)."""
    return CliffordAlgebra(p=2, q=0, r=0)


@pytest.fixture
def cl1_1() -> CliffordAlgebra:
    """Cl(1,1) Minkowski plane (4 blades)."""
    return CliffordAlgebra(p=1, q=1, r=0)


@pytest.fixture
def batch_shape() -> tuple:
    return (4, 3)  # (B, C)


# ========================================================================
#  GP Table Correctness
# ========================================================================


class TestGPTable:
    """Verify the geometric product table against known Clifford algebra identities."""

    def test_dimension_cl3(self, cl3):
        """Cl(3,0) should have dim = 2^3 = 8."""
        assert cl3.dim == 8
        assert cl3._gp_table.shape == (8, 8, 8)

    def test_dimension_cl2(self, cl2):
        """Cl(2,0) should have dim = 2^2 = 4."""
        assert cl2.dim == 4
        assert cl2._gp_table.shape == (4, 4, 4)

    def test_vector_square_positive(self, cl3):
        """In Cl(3,0): e1*e1 = e2*e2 = e3*e3 = 1."""
        for blade in [1, 2, 4]:  # e1=1, e2=2, e3=4
            x = torch.zeros(1, cl3.dim)
            x[0, blade] = 1.0
            result = cl3.geometric_product(x, x)
            assert result[0, 0].item() == pytest.approx(1.0), f"blade {blade}^2 should = 1"

    def test_vector_anticommute(self, cl3):
        """In Cl(3,0): e1*e2 = -e2*e1 (anticommutation)."""
        # e1*e2 = e12 (blade 3 = 0b011)
        assert cl3._gp_table[1, 2, 3] == 1.0, "e1*e2 should be e12"
        # e2*e1 = -e12
        assert cl3._gp_table[2, 1, 3] == -1.0, "e2*e1 should be -e12"

    def test_bivector_square_negative(self, cl3):
        """In Cl(3,0): e12*e12 = -1, e13*e13 = -1, e23*e23 = -1."""
        # Cl(3,0) blade indices: 3=e12(011), 5=e13(101), 6=e23(110)
        bivectors = [3, 5, 6]
        for b in bivectors:
            x = torch.zeros(1, cl3.dim)
            x[0, b] = 1.0
            result = cl3.geometric_product(x, x)
            assert result[0, 0].item() == pytest.approx(-1.0), f"bivector {b}^2 should be -1"

    def test_pseudoscalar_square(self, cl3):
        """In Cl(3,0): e123*e123 = -1."""
        ps = 7  # e123 (bitmask 111)
        x = torch.zeros(1, cl3.dim)
        x[0, ps] = 1.0
        result = cl3.geometric_product(x, x)
        assert result[0, 0].item() == pytest.approx(-1.0), "pseudoscalar^2 should be -1"

    def test_vector_times_bivector(self, cl3):
        """e1 * e12 = e2 (modulo sign)."""
        # e1*e12 = e1*e1*e2 = 1*e2 = e2
        assert cl3._gp_table[1, 3, 2] == 1.0, "e1*e12 should be e2"
        # e12*e1 = e1*e2*e1 = -e1*e1*e2 = -e2
        assert cl3._gp_table[3, 1, 2] == -1.0, "e12*e1 should be -e2"

    def test_negative_metric(self, cl1_1):
        """In Cl(1,1): e1^2 = 1, e2^2 = -1."""
        x_e1 = torch.zeros(1, 4); x_e1[0, 1] = 1.0
        assert cl1_1.geometric_product(x_e1, x_e1)[0, 0].item() == pytest.approx(1.0), "e1^2 should be 1"
        x_e2 = torch.zeros(1, 4); x_e2[0, 2] = 1.0
        assert cl1_1.geometric_product(x_e2, x_e2)[0, 0].item() == pytest.approx(-1.0), "e2^2 should be -1"

    def test_gp_table_symmetric_products(self, cl3):
        """Verify a few more products to catch regressions."""
        # e12 * e23 = e1*e2*e2*e3 = e1*e3 = e13
        assert cl3._gp_table[3, 6, 5] == 1.0, "e12*e23 should be e13"
        # e23 * e13 = e2*e3*e1*e3 = -e2*e3*e3*e1 = -e2*e1 = e12
        assert cl3._gp_table[6, 5, 3] == 1.0, "e23*e13 should be e12"
        # e123 * e1 = e1*e2*e3*e1 = -e1*e2*e1*e3 = e1*e1*e2*e3 = e23
        assert cl3._gp_table[7, 1, 6] == 1.0, "e123*e1 should be e23"

    def test_batched_gp(self, cl3):
        """Geometric product with batched inputs preserves shapes."""
        B, C = 4, 3
        x = torch.randn(B, C, cl3.dim)
        y = torch.randn(B, C, cl3.dim)
        result = cl3.geometric_product(x, y)
        assert result.shape == (B, C, cl3.dim)

    def test_gp_with_single_blade(self, cl3):
        """Geometric product of a single basis blade with itself (direct tensor API)."""
        # e1 * e1 = 1
        x = torch.zeros(1, cl3.dim)
        x[0, 1] = 1.0  # e1
        result = cl3.geometric_product(x, x)
        assert result[0, 0] == pytest.approx(1.0), "e1*e1 should be 1"
        assert result[0, 1:].abs().sum().item() == pytest.approx(0.0), "e1*e1 should have no non-scalar parts"


# ========================================================================
#  Grade Operations
# ========================================================================


class TestGradeOperations:
    """Verify grade projection, reverse, and involution."""

    def test_grade_index(self, cl3):
        """Grade index should correctly label each blade."""
        # Cl(3,0): grades are bit_count of index
        expected = [0, 1, 1, 2, 1, 2, 2, 3]  # 0..7
        for i in range(8):
            assert cl3._grade_index[i].item() == expected[i], f"blade {i} grade should be {expected[i]}"

    def test_grade_projection_scalar(self, cl3):
        """Grade-0 projection should isolate the scalar part."""
        x = torch.tensor([[1.0, 2, 3, 4, 5, 6, 7, 8]])
        scalar = cl3.grade_projection(x, 0)
        assert scalar[0, 0] == 1.0
        assert scalar[0, 1:].abs().sum().item() == pytest.approx(0.0)

    def test_grade_projection_vector(self, cl3):
        """Grade-1 projection should isolate vector part."""
        x = torch.tensor([[1.0, 2, 3, 4, 5, 6, 7, 8]])
        vec = cl3.grade_projection(x, 1)
        assert vec[0, 0] == pytest.approx(0.0)
        assert vec[0, 1] == 2.0  # e1
        assert vec[0, 2] == 3.0  # e2
        assert vec[0, 4] == 5.0  # e3 (not blade 3! blade 3 = e12 = grade 2)
        # All other blades should be zero
        non_vector = [0, 3, 5, 6, 7]
        for idx in non_vector:
            assert vec[0, idx].item() == pytest.approx(0.0), f"blade {idx} should be 0 in grade-1 projection"

    def test_reverse(self, cl3):
        """Reversion: (~x)_k = (-1)^{k(k-1)/2} x_k.

        Cl(3,0) blade indices (as bitmasks):
          0=0(000, scalar, g=0)  1=1(001, e1, g=1)    2=2(010, e2, g=1)
          3=3(011, e12, g=2)     4=4(100, e3, g=1)    5=5(101, e13, g=2)
          6=6(110, e23, g=2)     7=7(111, e123, g=3)
        """
        x = torch.ones(1, cl3.dim)
        rev = cl3.reverse(x)
        # Grade 0: (-1)^0 = 1
        assert rev[0, 0] == 1.0
        # Grade 1: (-1)^0 = 1  (blades 1, 2, 4)
        assert rev[0, 1] == 1.0
        assert rev[0, 2] == 1.0
        assert rev[0, 4] == 1.0
        # Grade 2: (-1)^1 = -1  (blades 3, 5, 6)
        assert rev[0, 3] == -1.0  # e12
        assert rev[0, 5] == -1.0  # e13
        assert rev[0, 6] == -1.0  # e23
        # Grade 3: (-1)^{3*2/2} = (-1)^3 = -1  (blade 7)
        assert rev[0, 7] == -1.0

    def test_grade_involution(self, cl3):
        """Grade involution: (x^)_k = (-1)^k x_k.

        Cl(3,0) blade indices:
          0(g=0, +1)  1(g=1, -1)  2(g=1, -1)  3(g=2, +1)
          4(g=1, -1)  5(g=2, +1)  6(g=2, +1)  7(g=3, -1)
        """
        x = torch.ones(1, cl3.dim)
        invol = cl3.grade_involution(x)
        assert invol[0, 0] == 1.0   # grade 0: +
        assert invol[0, 1] == -1.0  # grade 1: -
        assert invol[0, 2] == -1.0  # grade 1: -
        assert invol[0, 3] == 1.0   # grade 2: +
        assert invol[0, 4] == -1.0  # grade 1: -
        assert invol[0, 5] == 1.0   # grade 2: +
        assert invol[0, 6] == 1.0   # grade 2: +
        assert invol[0, 7] == -1.0  # grade 3: -

    def test_reverse_twice_is_identity(self, cl3):
        """~(~x) = x."""
        x = torch.randn(1, cl3.dim)
        assert torch.allclose(cl3.reverse(cl3.reverse(x)), x)

    def test_involution_twice_is_identity(self, cl3):
        """x^^ = x."""
        x = torch.randn(1, cl3.dim)
        assert torch.allclose(cl3.grade_involution(cl3.grade_involution(x)), x)


# ========================================================================
#  Metric Signs
# ========================================================================


class TestMetricSigns:
    """Verify blade metric sign computation."""

    def test_euclidean_signs(self, cl3):
        """In Cl(3,0): all blades have positive square."""
        signs = cl3.blade_metric_signs("cpu", torch.float32)
        for i in range(8):
            assert signs[i].item() == 1.0, f"blade {i} should have +1 metric sign"

    def test_minkowski_signs(self, cl1_1):
        """In Cl(1,1): blades with e2 have negative square."""
        signs = cl1_1.blade_metric_signs("cpu", torch.float32)
        # Blade 2 = e2: negative
        assert signs[2].item() == -1.0, "e2 should have negative metric"
        # Blade 1 = e1: positive
        assert signs[1].item() == 1.0, "e1 should have positive metric"
        # Blade 3 = e12: e1^2 * e2^2 = 1 * (-1) = -1
        assert signs[3].item() == -1.0, "e12 should have negative metric"
        # Blade 0 = scalar: positive
        assert signs[0].item() == 1.0, "scalar should have positive metric"

    def test_norm_sq_euclidean(self, cl3):
        """In Cl(3,0): ||x||^2 = sum(x_i^2) for arbitrary multivector."""
        x = torch.randn(1, cl3.dim)
        expected = (x ** 2).sum(dim=-1, keepdim=True)
        result = cl3.norm_sq(x)
        assert torch.allclose(result, expected)

    def test_norm_sq_minkowski(self, cl1_1):
        """In Cl(1,1): ||x||^2 = x0^2 + x1^2 - x2^2 - x12^2.

        Note: norm_sq clamps to 0 (for numerical stability in Euclidean mode),
        so we use a test vector with positive norm squared.
        """
        # x = a + b*e1  (e1 has positive metric, e2 has negative)
        idx_scalar, idx_e1, idx_e2, idx_e12 = 0, 1, 2, 3
        x = torch.zeros(1, 4)
        x[0, idx_scalar] = 3.0
        x[0, idx_e1] = 4.0
        result = cl1_1.norm_sq(x)
        expected = 9.0 + 16.0  # scalar^2 + e1^2
        assert result[0, 0].item() == pytest.approx(expected)


# ========================================================================
#  Bivector Exponential
# ========================================================================


class TestBivectorExp:
    """Verify bivector exponential produces valid rotors."""

    def test_zero_bivector_is_identity(self, cl3):
        """exp(0) = 1."""
        biv = torch.zeros(1, 1, cl3.dim)
        R = cl3.exp_bivector(biv)
        # Should be pure scalar = 1
        assert R[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
        assert R[0, 0, 1:].abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_rotor_norm_sq(self, cl3):
        """A rotor R = exp(-B/2) should satisfy ||R|| = 1 (in Euclidean metric)."""
        # Random bivector (grade-2 only)
        biv = torch.zeros(1, 1, cl3.dim)
        biv[0, 0, [3, 5, 6]] = torch.tensor([0.5, -0.3, 0.7])  # e12=3, e13=5, e23=6
        R = cl3.exp_bivector(-0.5 * biv)
        # Rotor norm should be 1
        norm_sq = cl3.norm_sq(R)
        assert norm_sq[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_rotor_identity_action(self, cl3):
        """R * x * R~ with R=1 should preserve x."""
        x = torch.randn(1, 1, cl3.dim)
        R = torch.zeros(1, 1, cl3.dim)
        R[0, 0, 0] = 1.0  # R = 1
        R_rev = cl3.reverse(R)
        result = cl3.sandwich_product(R, x, R_rev)
        assert torch.allclose(result, x, atol=1e-6)

    def test_rotor_preserves_norm(self, cl3):
        """Rotor sandwich should preserve multivector norm."""
        x = torch.randn(1, 1, cl3.dim)
        biv = torch.zeros(1, 1, cl3.dim)
        biv[0, 0, [3, 5, 6]] = torch.tensor([0.5, -0.3, 0.7])
        R = cl3.exp_bivector(-0.5 * biv)
        R_rev = cl3.reverse(R)
        result = cl3.sandwich_product(R, x, R_rev)
        orig_norm = cl3.norm_sq(x).sqrt()
        result_norm = cl3.norm_sq(result).sqrt()
        assert torch.allclose(orig_norm, result_norm, atol=1e-5)

    def test_batched_bivector_exp(self, cl3):
        """Bivector exponential works with batch dimensions."""
        B, C = 4, 3
        biv = torch.zeros(B, C, cl3.dim)
        # Cl(3,0) bivector indices: 3=e12, 5=e13, 6=e23
        biv_indices = torch.tensor([3, 5, 6])
        biv[..., biv_indices] = torch.randn(B, C, 3) * 0.5
        R = cl3.exp_bivector(biv)
        assert R.shape == (B, C, cl3.dim)
        # Each rotor should have unit norm
        norms = cl3.norm_sq(R)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_multiple_rotations_commute_same_plane(self, cl3):
        """Two rotors in the same plane commute."""
        biv1 = torch.zeros(1, 1, cl3.dim)
        biv1[0, 0, 3] = 0.3  # e12 (blade 3 = 0b011)
        biv2 = torch.zeros(1, 1, cl3.dim)
        biv2[0, 0, 3] = 0.5  # e12

        R1 = cl3.exp_bivector(-0.5 * biv1)
        R2 = cl3.exp_bivector(-0.5 * biv2)

        R12 = cl3.geometric_product(R2, R1)
        R21 = cl3.geometric_product(R1, R2)

        assert torch.allclose(R12, R21, atol=1e-6)


# ========================================================================
#  Sandwich Product
# ========================================================================


class TestSandwichProduct:
    """Verify sandwich product (rotor action) properties."""

    def test_vector_rotation_cl2(self, cl2):
        """In Cl(2,0), a rotor in the e12 plane rotates vectors.

        R = exp(-theta/2 * e12) rotates by angle theta in the e1-e2 plane.
        """
        theta = math.pi / 4  # 45 degrees
        biv = torch.zeros(1, 1, cl2.dim)
        biv[0, 0, 3] = 1.0  # e12 (blade 3 = 0b11)
        R = cl2.exp_bivector(-theta / 2 * biv)
        R_rev = cl2.reverse(R)

        # Rotate e1 by 45 degrees: should get (e1 + e2)/sqrt(2)
        e1 = torch.zeros(1, 1, cl2.dim)
        e1[0, 0, 1] = 1.0  # e1 (blade 1)
        rotated = cl2.sandwich_product(R, e1, R_rev)

        expected = torch.zeros(1, 1, cl2.dim)
        expected[0, 0, 1] = math.cos(theta)  # e1 component
        expected[0, 0, 2] = math.sin(theta)  # e2 component

        assert torch.allclose(rotated, expected, atol=1e-6)

    def test_180_degree_rotation(self, cl2):
        """Rotating by 180 degrees in e12 plane flips both axes."""
        theta = math.pi
        biv = torch.zeros(1, 1, cl2.dim)
        biv[0, 0, 3] = 1.0
        R = cl2.exp_bivector(-theta / 2 * biv)
        R_rev = cl2.reverse(R)

        vec = torch.zeros(1, 1, cl2.dim)
        vec[0, 0, 1] = 0.7
        vec[0, 0, 2] = -0.3
        rotated = cl2.sandwich_product(R, vec, R_rev)

        # 180 degree rotation = negation of vector components
        expected = -vec.clone()
        assert torch.allclose(rotated, expected, atol=1e-6)

    def test_sandwich_is_linear(self, cl3):
        """Sandwich product is linear: R*(ax+by)*R~ = a*R*x*R~ + b*R*y*R~."""
        a, b = 0.7, -0.3
        x = torch.randn(1, 1, cl3.dim)
        y = torch.randn(1, 1, cl3.dim)

        biv = torch.zeros(1, 1, cl3.dim)
        biv[0, 0, [3, 5, 6]] = torch.tensor([0.3, -0.5, 0.2])
        R = cl3.exp_bivector(-0.5 * biv)
        R_rev = cl3.reverse(R)

        # R*(ax+by)*R~
        combined = cl3.sandwich_product(R, a * x + b * y, R_rev)
        # a*R*x*R~ + b*R*y*R~
        expected = a * cl3.sandwich_product(R, x, R_rev) + b * cl3.sandwich_product(R, y, R_rev)

        assert torch.allclose(combined, expected, atol=1e-6)

    def test_sandwich_preserves_grade_structure(self, cl3):
        """Sandwich product preserves the grade structure (grade invariance).

        Cl(3,0) bivector indices: 3=e12(011), 5=e13(101), 6=e23(110).
        """
        x = torch.randn(1, 1, cl3.dim)
        biv = torch.zeros(1, 1, cl3.dim)
        biv[0, 0, [3, 5, 6]] = torch.tensor([0.3, -0.5, 0.2])
        R = cl3.exp_bivector(-0.5 * biv)
        R_rev = cl3.reverse(R)
        result = cl3.sandwich_product(R, x, R_rev)

        # Grade magnitudes should be preserved (up to numerical precision)
        for g in range(cl3.n + 1):
            x_g = cl3.grade_projection(x, g)
            r_g = cl3.grade_projection(result, g)
            if x_g.norm() > 1e-6:
                ratio = r_g.norm() / x_g.norm()
                assert ratio.item() == pytest.approx(1.0, abs=1e-3)

    def test_sandwich_batched(self, cl3):
        """Sandwich product with batched inputs."""
        B, C = 4, 3
        x = torch.randn(B, C, cl3.dim)
        R = torch.zeros(B, C, cl3.dim)
        R[..., 0] = 1.0
        R_rev = cl3.reverse(R)
        result = cl3.sandwich_product(R, x, R_rev)
        assert result.shape == (B, C, cl3.dim)
        assert torch.allclose(result, x, atol=1e-6)


# ========================================================================
#  Algebraic Identities
# ========================================================================


class TestAlgebraicIdentities:
    """Verify Clifford algebra identities hold numerically."""

    def test_einstein_notation(self, cl3):
        """Simple test: compute using full tensors and verify."""
        # e1*e2*e3 = e123
        e1 = torch.zeros(1, cl3.dim)
        e1[0, 1] = 1.0
        e2 = torch.zeros(1, cl3.dim)
        e2[0, 2] = 1.0
        e3 = torch.zeros(1, cl3.dim)
        e3[0, 4] = 1.0

        e12 = cl3.geometric_product(e1, e2)
        e123 = cl3.geometric_product(e12, e3)

        # e123 should have coefficient 1 in blade 7
        expected = torch.zeros(1, cl3.dim)
        expected[0, 7] = 1.0
        assert torch.allclose(e123, expected, atol=1e-6)

    def test_scalar_multiplication(self, cl3):
        """Scalars multiply as expected in geometric product."""
        x = 3.0 * torch.ones(1, cl3.dim)
        x[0, 1:] = 2.0
        y = 5.0 * torch.ones(1, cl3.dim)
        y[0, 1:] = 0.0  # pure scalar

        result = cl3.geometric_product(x, y)
        # 3*5 = 15 for scalar part
        assert result[0, 0].item() == pytest.approx(15.0)
        # Other parts scaled by 5
        assert result[0, 1].item() == pytest.approx(10.0)

    def test_associativity(self, cl3):
        """Geometric product is associative: (ab)c = a(bc)."""
        a = torch.randn(1, cl3.dim)
        b = torch.randn(1, cl3.dim)
        c = torch.randn(1, cl3.dim)

        ab_c = cl3.geometric_product(cl3.geometric_product(a, b), c)
        a_bc = cl3.geometric_product(a, cl3.geometric_product(b, c))

        assert torch.allclose(ab_c, a_bc, atol=1e-5)

    def test_anticommutation_identity(self, cl3):
        """In Cl(3,0): e1*e2 + e2*e1 = 0 (anticommute for distinct vectors)."""
        e1 = torch.zeros(1, cl3.dim)
        e1[0, 1] = 1.0
        e2 = torch.zeros(1, cl3.dim)
        e2[0, 2] = 1.0

        anti = cl3.geometric_product(e1, e2) + cl3.geometric_product(e2, e1)
        assert anti.abs().sum().item() == pytest.approx(0.0, abs=1e-6)


# ========================================================================
#  Neural Network Layers
# ========================================================================


class TestRotorLayer:
    """Verify RotorLayer construction and forward pass."""

    def test_initialization(self, cl3):
        """RotorLayer initializes with near-identity rotors."""
        layer = RotorLayer(cl3, channels=4)
        assert layer.channels == 4
        assert layer.biv_weights.shape == (4, 3)  # C(3,2) = 3 bivectors
        # Small random init
        assert layer.biv_weights.std().item() < 0.1

    def test_forward_shape(self, cl3):
        """Forward pass preserves shape."""
        layer = RotorLayer(cl3, channels=3)
        x = torch.randn(2, 3, cl3.dim)
        y = layer(x)
        assert y.shape == (2, 3, cl3.dim)

    def test_near_identity_init(self, cl3):
        """With small init, rotor should be close to identity."""
        layer = RotorLayer(cl3, channels=1, init_std=0.001)
        x = torch.randn(1, 1, cl3.dim)
        y = layer(x)
        assert torch.allclose(x, y, atol=0.01)

    def test_rotor_preserves_norm(self, cl3):
        """Rotor output should have same norm as input."""
        layer = RotorLayer(cl3, channels=2, init_std=0.1)
        x = torch.randn(3, 2, cl3.dim)
        y = layer(x)
        x_norm = cl3.norm_sq(x).sqrt()
        y_norm = cl3.norm_sq(y).sqrt()
        assert torch.allclose(x_norm, y_norm, atol=1e-5)

    def test_batched_forward(self, cl3):
        """RotorLayer with batched input (B, C, dim)."""
        layer = RotorLayer(cl3, channels=4)
        x = torch.randn(5, 4, cl3.dim)
        y = layer(x)
        assert y.shape == (5, 4, cl3.dim)


class TestCliffordLinear:
    """Verify CliffordLinear forward pass."""

    def test_shape(self, cl3):
        """Output has correct shape."""
        layer = CliffordLinear(cl3, in_channels=4, out_channels=8)
        x = torch.randn(2, 4, cl3.dim)
        y = layer(x)
        assert y.shape == (2, 8, cl3.dim)

    def test_no_bias(self, cl3):
        """CliffordLinear without bias."""
        layer = CliffordLinear(cl3, in_channels=4, out_channels=8, bias=False)
        assert layer.bias is None

    def test_blade_independence(self, cl3):
        """Each blade is mixed independently across channels."""
        layer = CliffordLinear(cl3, in_channels=4, out_channels=2)
        x = torch.randn(1, 4, cl3.dim)
        y = layer(x)
        # y[b, o, k] = sum_i W[o, i] * x[b, i, k]
        # So for blade k, output depends only on that blade's coefficients
        manual = torch.einsum("oi,...id->...od", layer.weight, x)
        if layer.bias is not None:
            manual = manual + layer.bias
        assert torch.allclose(y, manual, atol=1e-6)


class TestCliffordLayerNorm:
    """Verify CliffordLayerNorm."""

    def test_shape(self, cl3):
        """Output has same shape as input."""
        norm = CliffordLayerNorm(cl3, channels=4)
        x = torch.randn(2, 4, cl3.dim)
        y = norm(x)
        assert y.shape == x.shape

    def test_normalized_norm(self, cl3):
        """Output should have approximately unit norm."""
        norm = CliffordLayerNorm(cl3, channels=4, recover_scale=False)
        x = torch.randn(5, 4, cl3.dim) * 3.0 + 1.0  # varying magnitudes
        y = norm(x)
        norms = cl3.norm_sq(y).sqrt()
        assert norms.mean().item() == pytest.approx(1.0, abs=0.15)


class TestGeometricGELU:
    """Verify GeometricGELU."""

    def test_shape(self, cl3):
        """Output has same shape as input."""
        act = GeometricGELU(cl3, channels=4)
        x = torch.randn(2, 4, cl3.dim)
        y = act(x)
        assert y.shape == x.shape

    def test_preserves_direction(self, cl3):
        """GeometricGELU preserves direction of multivector."""
        act = GeometricGELU(cl3, channels=2)
        x = torch.randn(3, 2, cl3.dim)
        y = act(x)
        # Direction should be same (up to scalar multiplication)
        dot = (x * y).sum(dim=-1)
        norms = torch.sqrt((x ** 2).sum(dim=-1) * (y ** 2).sum(dim=-1))
        cos_sim = dot / norms.clamp(min=1e-8)
        assert torch.allclose(cos_sim, torch.ones_like(cos_sim), atol=1e-6)


class TestBladeSelector:
    """Verify BladeSelector."""

    def test_shape(self, cl3):
        """Output has same shape as input."""
        sel = BladeSelector(cl3, channels=4)
        x = torch.randn(2, 4, cl3.dim)
        y = sel(x)
        assert y.shape == x.shape

    def test_initial_pass_through(self, cl3):
        """With zero init logits, gate = 2*sigmoid(0) = 1, so near identity."""
        sel = BladeSelector(cl3, channels=3)
        x = torch.randn(2, 3, cl3.dim)
        y = sel(x)
        # 2 * sigmoid(0) = 1.0, so y ≈ x
        assert torch.allclose(y, x, atol=1e-2)


class TestCliffordAttractorBlock:
    """Verify CliffordAttractorBlock forward pass."""

    def test_shape(self, cl3):
        """Block preserves multivector shape."""
        block = CliffordAttractorBlock(cl3, channels=8)
        x = torch.randn(2, 8, cl3.dim)
        y = block(x)
        assert y.shape == x.shape

    def test_batched(self, cl3):
        """Block works with various batch sizes."""
        block = CliffordAttractorBlock(cl3, channels=6)
        for B in [1, 4, 17]:
            x = torch.randn(B, 6, cl3.dim)
            y = block(x)
            assert y.shape == (B, 6, cl3.dim)


# ========================================================================
#  DEQ Fixed-Point Solver
# ========================================================================


class TestDEQSolver:
    """Verify DEQ fixed-point solver with IFT gradients."""

    def test_simple_fixed_point(self):
        """Solve x* = f(x*) where f(x) = 0 (trivial)."""
        def f(x):
            return torch.zeros_like(x)

        x0 = torch.randn(4, 8)
        x_star = _solve_fixed_point(f, x0, max_iter=10, tol=1e-6)
        assert x_star.norm().item() == pytest.approx(0.0, abs=1e-5)

    def test_identity_fixed_point(self):
        """Solve x* = f(x*) where f(x) = x (every point is fixed)."""
        def f(x):
            return x

        x0 = torch.randn(4, 8)
        x_star = _solve_fixed_point(f, x0, max_iter=10, tol=1e-6)
        # Should converge to x0
        assert torch.allclose(x_star, x0, atol=1e-4)

    def test_contractive_map(self):
        """Solve x* = f(x*) where f(x) = alpha * x (|alpha| < 1)."""
        alpha = 0.5
        def f(x):
            return alpha * x

        x0 = torch.randn(4, 8)
        x_star = _solve_fixed_point(f, x0, max_iter=30, tol=1e-6)
        # Fixed point is x* = 0
        assert x_star.norm().item() == pytest.approx(0.0, abs=1e-4)

    def test_gradient_flow(self):
        """Gradients flow through the solver."""
        def f(x):
            return 0.5 * x

        x0 = torch.randn(2, 4, requires_grad=True)
        x_star = _solve_fixed_point(f, x0, max_iter=20, tol=1e-6)
        loss = x_star.sum()
        loss.backward()
        # Gradient should exist and be finite
        assert x0.grad is not None
        assert torch.isfinite(x0.grad).all()

    def test_batched_solver(self):
        """Solver handles batched inputs."""
        def f(x):
            return 0.3 * x

        x0 = torch.randn(8, 16)
        x_star = _solve_fixed_point(f, x0, max_iter=20, tol=1e-6)
        assert x_star.shape == (8, 16)
        assert x_star.norm().item() == pytest.approx(0.0, abs=1e-3)

    def test_batched_gradient(self):
        """Gradients flow through batched solver."""
        def f(x):
            return 0.5 * x + 0.1

        x0 = torch.randn(3, 5, requires_grad=True)
        x_star = _solve_fixed_point(f, x0, max_iter=20, tol=1e-4)
        loss = (x_star ** 2).sum()
        loss.backward()
        assert x0.grad is not None
        assert torch.isfinite(x0.grad).all()


# ========================================================================
#  Model Integration
# ========================================================================


class TestCliffordAttractor:
    """Verify the full CliffordAttractor model."""

    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        cfg = CliffordAttractorConfig(p=3, q=0, channels=8, num_blocks=2, max_iter=10)
        model = CliffordAttractor(cfg, vocab_size=11)
        x = torch.randint(0, 10, (2, 9))
        y = model(x)
        assert y.shape == (2, 9, 11)

    def test_backward(self):
        """Backward pass works (gradients flow through solver to all params)."""
        cfg = CliffordAttractorConfig(p=3, q=0, channels=8, num_blocks=2, max_iter=10)
        model = CliffordAttractor(cfg, vocab_size=11)
        x = torch.randint(0, 10, (2, 9))
        y = model(x)
        loss = y.sum()
        loss.backward()
        # All trainable parameters should have gradients after IFT backward
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"

    def test_different_configs(self):
        """Model works with different algebra signatures."""
        for p, q in [(2, 0), (3, 0), (1, 1)]:
            cfg = CliffordAttractorConfig(p=p, q=q, channels=4, num_blocks=1, max_iter=5)
            model = CliffordAttractor(cfg, vocab_size=11)
            x = torch.randint(0, 10, (1, 4))
            y = model(x)
            assert y.shape == (1, 4, 11)

    def test_create_factory(self):
        """Factory function creates a working model."""
        model = create_clifford_attractor(p=3, q=0, channels=8, num_blocks=2, vocab_size=11)
        x = torch.randint(0, 10, (1, 9))
        y = model(x)
        assert y.shape == (1, 9, 11)

    def test_training_step(self):
        """Model can perform a training step (forward + backward + optimizer)."""
        cfg = CliffordAttractorConfig(p=3, q=0, channels=8, num_blocks=2, max_iter=10)
        model = CliffordAttractor(cfg, vocab_size=11)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randint(0, 10, (2, 9))
        y = model(x)
        loss = F.cross_entropy(y.view(-1, 11), torch.randint(0, 10, (2 * 9,)))
        loss.backward()
        optimizer.step()
        # Loss should be finite
        assert torch.isfinite(loss)

    def test_deterministic(self):
        """Model produces same output for same input (eval mode)."""
        cfg = CliffordAttractorConfig(p=3, q=0, channels=8, num_blocks=2, max_iter=10)
        torch.manual_seed(42)
        model = CliffordAttractor(cfg, vocab_size=11)
        model.eval()
        x = torch.randint(0, 10, (1, 9))
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        assert torch.allclose(y1, y2)


# ========================================================================
#  Run
# ========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
