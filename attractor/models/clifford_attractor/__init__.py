"""Clifford-algebra extension to the Attractor framework.

Subpackage layout:
    algebra.py — Cl(p,q,r) kernel: GP tables, grade ops, rotor exp.
    layers.py  — Rotor / Linear / LayerNorm / GELU / BladeSelector / Block.
    model.py   — DEQ solver shim + CliffordAttractor config and model.

This __init__ re-exports the same public names that used to live in the
former clifford_attractor.py module, so existing imports continue to work:

    from attractor.models.clifford_attractor import CliffordAlgebra, ...
"""

from .algebra import (
    CliffordAlgebra,
    build_gp_table,
    _gp_blade,
    _blade_grade,
    _grade_index_map,
    _blade_metric_signs_precomputed,
    _extract_sparse_gp,
    _filter_sparse_op,
    _filter_sparse_ip,
)
from .layers import (
    RotorLayer,
    CliffordLinear,
    CliffordLayerNorm,
    GeometricGELU,
    BladeSelector,
    CliffordAttractorBlock,
)
from .model import (
    DEQFixedPoint,
    _solve_fixed_point,
    CliffordAttractorConfig,
    CliffordAttractor,
    create_clifford_attractor,
)

__all__ = [
    # algebra
    "CliffordAlgebra",
    "build_gp_table",
    # layers
    "RotorLayer",
    "CliffordLinear",
    "CliffordLayerNorm",
    "GeometricGELU",
    "BladeSelector",
    "CliffordAttractorBlock",
    # DEQ
    "DEQFixedPoint",
    "_solve_fixed_point",
    # model
    "CliffordAttractorConfig",
    "CliffordAttractor",
    "create_clifford_attractor",
]
