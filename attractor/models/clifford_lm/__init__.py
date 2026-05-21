"""Hybrid Clifford language model: Attractor + Clifford MLP sublayer."""

from .config import CliffordLMConfig
from .clifford_lm import CliffordFPBlock, CliffordLM, CliffordSublayer, create_clifford_lm
from .native import (
    CliffordSelfAttention,
    NativeCliffordFPBlock,
    NativeCliffordLM,
    create_native_clifford_lm,
)
from .attn_only import AttnOnlyCliffordFPBlock
from .prelude import CliffordAttnPreludeBlock

__all__ = [
    "CliffordLMConfig",
    "CliffordFPBlock",
    "CliffordLM",
    "CliffordSublayer",
    "CliffordSelfAttention",
    "NativeCliffordFPBlock",
    "NativeCliffordLM",
    "AttnOnlyCliffordFPBlock",
    "CliffordAttnPreludeBlock",
    "create_clifford_lm",
    "create_native_clifford_lm",
]
