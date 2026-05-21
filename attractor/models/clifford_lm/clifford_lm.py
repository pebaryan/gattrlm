"""Hybrid Clifford language model.

CliffordLM is the standard Attractor architecture (causal transformer prelude
producing context c, weight-tied fixed-point head solved via Anderson + IFT)
with the FP block's MLP replaced by a Clifford rotor / geometric-product /
geometric-GELU sublayer. Everything else — attention, RoPE, LayerScale gates,
optimizer param tagging, training contract — is inherited from Attractor.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from attractor.models.attractor.attractor import Attractor, FixedPointBlock
from attractor.models.clifford_attractor import CliffordAlgebra, CliffordAttractorBlock
from attractor.models.clifford_lm.config import CliffordLMConfig


class CliffordSublayer(nn.Module):
    """Drop-in MLP replacement that routes through a multivector channel space.

    in_dim → fc_in → reshape to (..., channels, algebra.dim) →
    CliffordAttractorBlock (rotor sandwich, geometric product, geometric GELU,
    blade selector) → reshape back → fc_out → in_dim.

    Operates on arbitrary leading dims; for an FP block we get [B, S, in_dim].
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_dim: int,
        channels: int,
        hidden_channels: Optional[int] = None,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.dim = algebra.dim
        flat = channels * self.dim

        # Lift to multivector space. No bias so the lift starts neutral.
        self.fc_in = nn.Linear(in_dim, flat, bias=False)
        self.clif_block = CliffordAttractorBlock(
            algebra=algebra,
            channels=channels,
            hidden_channels=hidden_channels or channels,
            use_geometric_activation=True,
            use_blade_selector=True,
            init_std=init_std,
        )
        # Project back to embedding space.
        self.fc_out = nn.Linear(flat, in_dim, bias=False)

        # Match the standard MLP's output-projection init pattern: a small
        # std so the sublayer is near-zero at init, keeping the surrounding
        # FP block contractive at the start of training.
        out_std = math.sqrt(2.0 / (5.0 * in_dim))
        nn.init.trunc_normal_(self.fc_out.weight, mean=0.0, std=out_std,
                              a=-3 * out_std, b=3 * out_std)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., in_dim]
        leading = x.shape[:-1]
        h = self.fc_in(x)                                # [..., flat]
        h = h.reshape(-1, self.channels, self.dim)       # [N, C, dim]
        h = self.clif_block(h)
        h = h.reshape(*leading, self.channels * self.dim)
        return self.fc_out(h)


class CliffordFPBlock(FixedPointBlock):
    """Fixed-point block whose MLP sublayer is a CliffordSublayer.

    Inherits attention, LayerNorm, LayerScale gating, and the contractive
    init pattern from FixedPointBlock. Only the MLP is swapped.
    """

    def __init__(self, config: CliffordLMConfig, layer_id: int,
                 layer_scale_init: Optional[float] = None,
                 gamma_max: Optional[float] = None) -> None:
        super().__init__(config, layer_id=layer_id,
                         layer_scale_init=layer_scale_init,
                         gamma_max=gamma_max)
        # The standard MLP created by super().__init__ is unused; replace.
        del self.mlp
        algebra = CliffordAlgebra(config.clifford_p, config.clifford_q, config.clifford_r)
        self.mlp = CliffordSublayer(
            algebra=algebra,
            in_dim=config.n_embd,
            channels=config.n_clifford_channels,
            hidden_channels=config.n_clifford_hidden,
        )

    # forward() is inherited from FixedPointBlock; it calls self.mlp on
    # norm_2(x), gates the result with raw_gamma_mlp, and adds the skip.


class CliffordLM(Attractor):
    """Attractor LM with flag-selected Clifford sublayers in the FP head and
    optionally in the prelude.

    Flags on CliffordLMConfig:
      clifford_attention        — Clifford self-attention in the FP block
      clifford_mlp              — Clifford MLP sublayer in the FP block
      clifford_attention_prelude — Clifford self-attention in the prelude

    When the prelude is Clifford, CliffordSelfAttention has no RoPE, so we
    add a learned positional embedding `wpe` to wte's output to compensate.
    All other Attractor machinery — IFT, Anderson solver, optimizer param
    tagging, monitoring, loss heads — is inherited unchanged.
    """

    def __init__(self, config, objective=None, gradient_checkpointing: bool = False) -> None:
        super().__init__(config, objective=objective,
                         gradient_checkpointing=gradient_checkpointing)
        # If the prelude has Clifford attention, RoPE no longer encodes
        # position — add a learned absolute positional embedding.
        if getattr(config, "clifford_attention_prelude", False):
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
            nn.init.normal_(self.transformer.wpe.weight, std=0.02)

    def _encode(self, input_ids: Tensor, freqs_cis: Tensor,
                attention_mask: Optional[Tensor]) -> Tensor:
        # Inject the learned positional embedding (if present) right after
        # token embeddings. The rest of _encode is identical to Attractor's.
        x = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        if hasattr(self.transformer, "wpe"):
            pos = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
            x = x + self.transformer.wpe(pos).unsqueeze(0)
        for i, block in enumerate(self.transformer.prelude):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            if self.gradient_checkpointing:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask, ve=ve)
            else:
                x = block(x, freqs_cis, attention_mask, ve=ve)
        return x

    def _make_prelude_block(self, config, layer_id: int) -> nn.Module:
        if getattr(config, "clifford_attention_prelude", False):
            from attractor.models.clifford_lm.prelude import CliffordAttnPreludeBlock
            return CliffordAttnPreludeBlock(config, layer_id=layer_id)
        return super()._make_prelude_block(config, layer_id)

    def _make_fp_block(self, config, layer_id: int) -> nn.Module:
        ca = bool(getattr(config, "clifford_attention", False))
        cm = bool(getattr(config, "clifford_mlp", True))

        if ca and cm:
            # Both sublayers Clifford. Defined in native.py to keep the
            # CliffordSelfAttention import local to that module.
            from attractor.models.clifford_lm.native import NativeCliffordFPBlock
            cls = NativeCliffordFPBlock
        elif ca and not cm:
            from attractor.models.clifford_lm.attn_only import AttnOnlyCliffordFPBlock
            cls = AttnOnlyCliffordFPBlock
        elif cm and not ca:
            cls = CliffordFPBlock
        else:
            # Neither sublayer is Clifford. Fall back to the standard
            # FixedPointBlock — useful for sanity-checking flag plumbing.
            cls = FixedPointBlock

        return cls(
            config,
            layer_id=layer_id,
            layer_scale_init=config.layer_scale_init,
            gamma_max=config.gamma_max,
        )


# Convenience for callers that just want a configured model.
def create_clifford_lm(name: str = "clifford-small-140m", **overrides) -> CliffordLM:
    """Build a CliffordLM from a registered config name."""
    cfg = CliffordLMConfig.from_name(name, **overrides)
    return cfg.construct_model()
