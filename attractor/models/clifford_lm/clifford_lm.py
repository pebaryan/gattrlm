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
    """Attractor LM with Clifford MLP sublayers in the fixed-point head.

    All Attractor behavior — IFT, Anderson solver, optimizer param tagging,
    monitoring, loss heads — is inherited unchanged. The only override is
    the FP block factory.
    """

    def _make_fp_block(self, config, layer_id: int) -> nn.Module:
        return CliffordFPBlock(
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
