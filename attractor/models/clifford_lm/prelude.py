"""Clifford-attention prelude block.

Mirror of TransformerPreNormBlock but with CliffordSelfAttention in place
of CausalSelfAttention and standard BaseMLP in the FF position. The block
ignores `freqs_cis` because CliffordSelfAttention has no RoPE — when this
block is used, CliffordLM adds a learned positional embedding before the
prelude to compensate (see CliffordLM.__init__).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from attractor.models.clifford_attractor import CliffordAlgebra
from attractor.models.clifford_lm.config import CliffordLMConfig
from attractor.models.clifford_lm.native import CliffordSelfAttention


class CliffordAttnPreludeBlock(nn.Module):
    """Pre-norm transformer block with Clifford self-attention + standard MLP.

    Drop-in for TransformerPreNormBlock in the prelude. Accepts the same
    forward signature (`x, freqs_cis, mask=None, **kwargs`) but ignores
    `freqs_cis` since the multivector Q/K/V have no RoPE applied.
    """

    expanded = False

    def __init__(self, config: CliffordLMConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        algebra = CliffordAlgebra(config.clifford_p, config.clifford_q, config.clifford_r)
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CliffordSelfAttention(
            algebra=algebra,
            n_embd=config.n_embd,
            n_heads=config.n_clifford_attn_heads,
            channels_per_head=config.n_clifford_attn_channels_per_head,
            use_rope=bool(getattr(config, "multivector_rope", False)),
            max_seq_len=int(getattr(config, "block_size", 2048)),
            rope_base=float(getattr(config, "multivector_rope_base", 10000.0)),
        )
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)

    def forward(self, x: Tensor, freqs_cis: Tensor,
                mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        # freqs_cis is unused; positional info enters via a learned embedding
        # added before the prelude (CliffordLM injects it).
        kwargs.pop("ve", None)
        x = x + self.attn(self.norm_1(x), mask=mask)
        x = x + self.mlp(self.norm_2(x))
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")
