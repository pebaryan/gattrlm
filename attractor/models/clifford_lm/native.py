"""Native Clifford language model: Clifford attention + Clifford MLP in the FP block.

Path B from the architecture audit. Where CliffordLM is a hybrid (standard
transformer attention + Clifford MLP), NativeCliffordLM replaces *both*
sublayers of each FP block with Clifford operations:

  CliffordSelfAttention: lift x to multivector channels (Cl(p,q,r)), produce
  Q/K/V via per-head linear mixing, score with the Clifford scalar product
  <q,k> = grade-0(q * reverse(k)) summed over channels-per-head, apply causal
  softmax, output as scalar-weighted sum of V multivectors, project back.

  CliffordSublayer (from clifford_lm.clifford_lm): the existing rotor / GP /
  geometric-GELU pipeline replacing the standard MLP.

The standard transformer prelude is kept unchanged — it supplies RoPE
positional encoding, which the Clifford attention then operates on top of.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from attractor.models.attractor.attractor import FixedPointBlock
from attractor.models.clifford_attractor import CliffordAlgebra
from attractor.models.clifford_lm.clifford_lm import CliffordLM, CliffordSublayer
from attractor.models.clifford_lm.config import CliffordLMConfig


class CliffordSelfAttention(nn.Module):
    """Causal self-attention with multivector Q/K/V.

    Score for query position i, key position j:
        s_ij = sum_c  grade-0( q_{i,c} * reverse(k_{j,c}) ) / scale

    where q, k are multivectors of grade signature Cl(p,q,r), c indexes the
    channels in one head, and scale = sqrt(channels_per_head * algebra.dim).
    The reversion makes <q, k> a proper Clifford scalar product (the metric
    inner product on Cl(p,q,r)).

    Output is the scalar-attention-weighted sum of V multivectors, flattened
    across (head, channel, blade) and projected back to n_embd.
    """

    def __init__(self, algebra: CliffordAlgebra, n_embd: int,
                 n_heads: int, channels_per_head: int):
        super().__init__()
        self.algebra = algebra
        self.dim = algebra.dim
        self.n_heads = n_heads
        self.channels_per_head = channels_per_head
        self.total_channels = n_heads * channels_per_head

        # Lift x → 3 MV stacks (Q, K, V). No bias so init is neutral.
        self.qkv_lift = nn.Linear(n_embd, 3 * self.total_channels * self.dim, bias=False)
        self.out_proj = nn.Linear(self.total_channels * self.dim, n_embd, bias=False)

        # Cache the scalar (grade-0) slice of the Cayley table — that's all
        # we need to compute <q, k> = grade-0(q * reverse(k)).
        # gp_table[a, b, c] = coeff of blade c in (e_a * e_b); we want c=0.
        self.register_buffer("_scalar_gp", algebra._gp_table[:, :, 0].clone(),
                             persistent=False)

        # Match the FixedPointBlock contractive init: small out_proj std so
        # the sublayer is near-zero at start, keeping the FP map contractive.
        out_std = math.sqrt(2.0 / (5.0 * n_embd))
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=out_std,
                              a=-3 * out_std, b=3 * out_std)

        self.scale = 1.0 / math.sqrt(self.channels_per_head * self.dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: [B, S, n_embd]
        B, S, _ = x.shape
        H, C, A = self.n_heads, self.channels_per_head, self.dim

        qkv = self.qkv_lift(x)                                      # [B, S, 3*H*C*A]
        qkv = qkv.view(B, S, 3, H, C, A)
        q, k, v = qkv.unbind(dim=2)                                 # each [B, S, H, C, A]

        # Move seq inside: [B, H, S, C, A]
        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)

        # Reverse the key blades. Then Clifford scalar product is:
        #   <q_i, k_j> = grade-0( q_i * reverse(k_j) )
        #             = sum_{a,b} q[a] * scalar_gp[a, b] * reverse(k)[b]
        # Summed across channels-per-head.
        k_rev = self.algebra.reverse(k)
        scalar_gp = self._scalar_gp.to(dtype=q.dtype)               # [A, A]
        # Einsum letters: n=batch, h=head, i=query, j=key, c=channel,
        # a/d=blade indices into the Cayley scalar slice.
        scores = torch.einsum(
            "nhica,ad,nhjcd->nhij", q, scalar_gp, k_rev
        ) * self.scale                                              # [B, H, S, S]

        # Causal mask: query i may attend to keys j <= i.
        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        if mask is not None:
            # additive mask (e.g. padding); shape broadcastable to scores
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)                            # [B, H, S, S]

        # Output: scalar-weighted sum of V multivectors.
        out = torch.einsum("nhij,nhjca->nhica", attn, v)            # [B, H, S, C, A]
        out = out.permute(0, 2, 1, 3, 4).contiguous()               # [B, S, H, C, A]
        out = out.view(B, S, self.total_channels * self.dim)
        return self.out_proj(out)


class NativeCliffordFPBlock(FixedPointBlock):
    """FP block whose attention AND MLP are both Clifford-native."""

    def __init__(self, config: CliffordLMConfig, layer_id: int,
                 layer_scale_init: Optional[float] = None,
                 gamma_max: Optional[float] = None) -> None:
        super().__init__(config, layer_id=layer_id,
                         layer_scale_init=layer_scale_init,
                         gamma_max=gamma_max)
        algebra = CliffordAlgebra(config.clifford_p, config.clifford_q, config.clifford_r)

        # Swap out the standard attention.
        del self.attn
        self.attn = CliffordSelfAttention(
            algebra=algebra,
            n_embd=config.n_embd,
            n_heads=config.n_clifford_attn_heads,
            channels_per_head=config.n_clifford_attn_channels_per_head,
        )
        # And the MLP (same swap as CliffordFPBlock).
        del self.mlp
        self.mlp = CliffordSublayer(
            algebra=algebra,
            in_dim=config.n_embd,
            channels=config.n_clifford_channels,
            hidden_channels=config.n_clifford_hidden,
        )

    def forward(self, x: Tensor, freqs_cis: Tensor,
                mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        # Clifford attention doesn't consume freqs_cis (positional info comes
        # from the standard-attention prelude, which has already encoded
        # position into x before the FP solver sees it).
        kwargs.pop("ve", None)
        attn_out = self.attn(self.norm_1(x), mask=mask)
        if self.raw_gamma_attn is not None:
            attn_out = attn_out * (torch.sigmoid(self.raw_gamma_attn) * self.gamma_max)
        x = x + attn_out
        mlp_out = self.mlp(self.norm_2(x))
        if self.raw_gamma_mlp is not None:
            mlp_out = mlp_out * (torch.sigmoid(self.raw_gamma_mlp) * self.gamma_max)
        return x + mlp_out


class NativeCliffordLM(CliffordLM):
    """CliffordLM with Clifford attention inside every FP block.

    The standard transformer prelude / coda / embedding stack is inherited
    unchanged from Attractor; only the FP-head block factory is overridden.
    """

    def _make_fp_block(self, config, layer_id: int) -> nn.Module:
        return NativeCliffordFPBlock(
            config,
            layer_id=layer_id,
            layer_scale_init=config.layer_scale_init,
            gamma_max=config.gamma_max,
        )


def create_native_clifford_lm(name: str = "native-clifford-small-140m", **overrides) -> NativeCliffordLM:
    """Build a NativeCliffordLM from a registered config name."""
    cfg = CliffordLMConfig.from_name(name, **overrides)
    cfg.native_attention = True
    return NativeCliffordLM(cfg)
