"""AttnOnlyCliffordFPBlock: the (Clifford attention + standard MLP) variant.

Built to fill out the 2x2 ablation over (attention algebra) x (MLP algebra)
inside the Attractor FP block. Pairs CliffordSelfAttention with the standard
BaseMLP so we can isolate which sublayer carries the equivariance signal.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from attractor.models.attractor.attractor import FixedPointBlock
from attractor.models.clifford_attractor import CliffordAlgebra
from attractor.models.clifford_lm.config import CliffordLMConfig
from attractor.models.clifford_lm.native import CliffordSelfAttention


class AttnOnlyCliffordFPBlock(FixedPointBlock):
    """FP block whose attention sublayer is Clifford-native; MLP is standard.

    Inherits LayerScale gates (raw_gamma_attn, raw_gamma_mlp) and the
    contractive init pattern from FixedPointBlock. The standard MLP built
    by the parent __init__ is kept; only self.attn is replaced.
    """

    def __init__(self, config: CliffordLMConfig, layer_id: int,
                 layer_scale_init: Optional[float] = None,
                 gamma_max: Optional[float] = None) -> None:
        super().__init__(config, layer_id=layer_id,
                         layer_scale_init=layer_scale_init,
                         gamma_max=gamma_max)
        algebra = CliffordAlgebra(config.clifford_p, config.clifford_q, config.clifford_r)

        del self.attn
        self.attn = CliffordSelfAttention(
            algebra=algebra,
            n_embd=config.n_embd,
            n_heads=config.n_clifford_attn_heads,
            channels_per_head=config.n_clifford_attn_channels_per_head,
            use_rope=bool(getattr(config, "multivector_rope", False)),
            max_seq_len=int(getattr(config, "block_size", 2048)),
            rope_base=float(getattr(config, "multivector_rope_base", 10000.0)),
        )

    def forward(self, x: Tensor, freqs_cis: Tensor,
                mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        # CliffordSelfAttention ignores freqs_cis (positional info already in x
        # from the prelude). Standard MLP path is unchanged from FixedPointBlock.
        kwargs.pop("ve", None)
        attn_out = self.attn(self.norm_1(x), mask=mask)
        if self.raw_gamma_attn is not None:
            attn_out = attn_out * (torch.sigmoid(self.raw_gamma_attn) * self.gamma_max)
        x = x + attn_out
        mlp_out = self.mlp(self.norm_2(x))
        if self.raw_gamma_mlp is not None:
            mlp_out = mlp_out * (torch.sigmoid(self.raw_gamma_mlp) * self.gamma_max)
        return x + mlp_out
