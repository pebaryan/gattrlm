from dataclasses import dataclass
from typing import Literal, Optional

import torch

from attractor.models.attractor.config import AttractorConfig


@dataclass
class CliffordLMConfig(AttractorConfig):
    """Hybrid Clifford language model config.

    Inherits the entire AttractorConfig surface (solver, IFT, LR scaling,
    LayerScale gating) and adds the Cl(p,q,r) signature plus the number of
    multivector channels used inside the Clifford MLP sublayer that replaces
    the standard MLP in each fixed-point block.
    """

    # Clifford algebra signature.
    clifford_p: int = 3
    clifford_q: int = 0
    clifford_r: int = 0

    # Number of multivector channels in the Clifford MLP. The Clifford lift
    # dimension is n_clifford_channels * (2 ** (clifford_p + clifford_q + clifford_r));
    # the lift size should be on the order of intermediate_size for parity
    # with a standard MLP.
    n_clifford_channels: int = 64
    n_clifford_hidden: Optional[int] = None  # defaults to n_clifford_channels

    # Two independent toggles. They control the FP block's two sublayers:
    #   clifford_attention=True  -> CliffordSelfAttention   (else standard attn)
    #   clifford_mlp=True        -> CliffordSublayer        (else standard MLP)
    # The 2x2 combinations:
    #   (F, F) -> standard FP block (= Attractor at the block level)
    #   (F, T) -> CliffordFPBlock         (CliffordLM, the original hybrid)
    #   (T, F) -> AttnOnlyCliffordFPBlock (Clifford attention only; "AttnOnlyCliffordLM")
    #   (T, T) -> NativeCliffordFPBlock   (NativeCliffordLM, fully Clifford)
    clifford_attention: bool = False
    clifford_mlp: bool = True

    # Apply Clifford attention to the *prelude* (encoder) blocks too.
    # When True, CliffordLM substitutes CliffordAttnPreludeBlock for the
    # default TransformerPreNormBlock in the prelude AND adds a learned
    # positional embedding to wte (since Clifford attention has no RoPE).
    clifford_attention_prelude: bool = False

    # Control variant: keep STANDARD attention in the prelude but suppress
    # RoPE by passing a no-op freqs_cis (all 1+0j). A learned positional
    # embedding is added in its place. Isolates the RoPE-loss penalty from
    # the Clifford-attention effect — pair with clifford_attention_prelude=False
    # for the clean control of PreludeOnlyCliffordLM.
    disable_rope_in_prelude: bool = False

    # Multivector RoPE: when True, every CliffordSelfAttention applies a
    # rotor sandwich on Q/K with position-dependent rotors (one bivector
    # axis per channel, frequency α_c = base^(-c/(C-1))). Gives Clifford
    # attention a native rotary positional encoding so the prelude no
    # longer needs the wpe substitute. See CliffordRotaryEmb.
    multivector_rope: bool = False
    multivector_rope_base: float = 10000.0

    # Clifford attention sub-config (only consulted when any clifford_*
    # attention flag is True). Heads split the channel space; each head's
    # score is the Clifford scalar product summed over channels_per_head.
    n_clifford_attn_heads: int = 4
    n_clifford_attn_channels_per_head: int = 4

    # Back-compat alias for `clifford_attention`. Old configs and old code paths
    # used `native_attention`; if it's set True we propagate to the new flag in
    # __post_init__.
    native_attention: bool = False

    model_class_name: Literal["CliffordLM", "NativeCliffordLM"] = "CliffordLM"

    def __post_init__(self):
        super().__post_init__()
        if self.n_clifford_hidden is None:
            self.n_clifford_hidden = self.n_clifford_channels
        # Promote the legacy flag — only one direction (True overrides);
        # explicit clifford_attention=True still wins regardless.
        if self.native_attention:
            self.clifford_attention = True

    def construct_model(self, **kwargs) -> torch.nn.Module:
        # NativeCliffordLM is a thin subclass; preserve the class identity
        # users that may rely on isinstance(...) checks (e.g. test_models.py).
        if self.clifford_attention and self.clifford_mlp:
            from attractor.models.clifford_lm.native import NativeCliffordLM
            return NativeCliffordLM(self, **kwargs)
        from attractor.models.clifford_lm.clifford_lm import CliffordLM
        return CliffordLM(self, **kwargs)
