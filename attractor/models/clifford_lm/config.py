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

    # Native Clifford attention (only consulted when native_attention=True).
    # Heads split the channel space; each head's score is the Clifford scalar
    # product summed over channels_per_head.
    native_attention: bool = False
    n_clifford_attn_heads: int = 4
    n_clifford_attn_channels_per_head: int = 4

    model_class_name: Literal["CliffordLM", "NativeCliffordLM"] = "CliffordLM"

    def __post_init__(self):
        super().__post_init__()
        if self.n_clifford_hidden is None:
            self.n_clifford_hidden = self.n_clifford_channels

    def construct_model(self, **kwargs) -> torch.nn.Module:
        if self.native_attention:
            from attractor.models.clifford_lm.native import NativeCliffordLM
            return NativeCliffordLM(self, **kwargs)
        from attractor.models.clifford_lm.clifford_lm import CliffordLM
        return CliffordLM(self, **kwargs)
