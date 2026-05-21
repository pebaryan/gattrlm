"""Hybrid Clifford language model: Attractor + Clifford MLP sublayer."""

from .config import CliffordLMConfig
from .clifford_lm import CliffordFPBlock, CliffordLM, CliffordSublayer, create_clifford_lm
from .native import (
    CliffordSelfAttention,
    NativeCliffordFPBlock,
    NativeCliffordLM,
    create_native_clifford_lm,
)

__all__ = [
    "CliffordLMConfig",
    "CliffordFPBlock",
    "CliffordLM",
    "CliffordSublayer",
    "CliffordSelfAttention",
    "NativeCliffordFPBlock",
    "NativeCliffordLM",
    "create_clifford_lm",
    "create_native_clifford_lm",
]
