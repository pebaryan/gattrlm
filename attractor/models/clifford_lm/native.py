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


class CliffordRotaryEmb(nn.Module):
    """Multivector RoPE — position-dependent rotor sandwich on Q/K channels.

    For each head channel c and position m, applies
        q'_{m,c} = R_{m,c} · q_{m,c} · R̃_{m,c}
    where R_{m,c} = exp(-½ · m · α_c · B_c) is a fixed-axis rotor.

    Per-channel choices:
      B_c — unit bivector axis, cycled through algebra's grade-2 basis blades.
            For Cl(3,0) the basis is {e₁₂, e₁₃, e₂₃} (3 axes); channels are
            assigned `c mod n_bivectors`.
      α_c — frequency, α_c = base^(-c / max(C-1, 1)), so log-frequency spans
            [-log(base), 0] across channels. Mirrors standard RoPE's spectral
            spread.

    Relative-position property: with both Q and K rotated under the same
    scheme, ⟨q'_m, k'_n⟩ = grade-0(q'_m · reverse(k'_n)) equals
    ⟨R_{m-n} q R̃_{m-n}, k⟩ when the per-channel rotors commute (single
    axis per channel), so the score is a function of (m - n) within each
    channel; summing channels preserves it overall.
    """

    def __init__(self, algebra: CliffordAlgebra,
                 channels_per_head: int, max_seq_len: int,
                 base: float = 10000.0):
        super().__init__()
        self.algebra = algebra
        self.channels_per_head = channels_per_head
        self.max_seq_len = max_seq_len
        D = algebra.dim

        biv_indices = algebra.bivector_indices().tolist()
        n_bivs = len(biv_indices)
        if n_bivs == 0:
            raise ValueError(
                "CliffordRotaryEmb requires the algebra to have at least one "
                "grade-2 basis blade (Cl(p,q,r) with p+q+r >= 2)."
            )

        # α_c = base^(-c / (C-1)); c=0 -> 1, c=C-1 -> 1/base.
        denom = max(channels_per_head - 1, 1)
        freqs = torch.tensor(
            [base ** (-(c / denom)) for c in range(channels_per_head)],
            dtype=torch.float32,
        )

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # Bivector tensor [max_seq_len, C, D]: pick one bivector axis per
        # channel and scale by -half-angle (rotor convention).
        biv = torch.zeros(max_seq_len, channels_per_head, D)
        for c in range(channels_per_head):
            axis_idx = biv_indices[c % n_bivs]
            biv[:, c, axis_idx] = -0.5 * positions * freqs[c]

        # exp_bivector is differentiable but we only use it once at __init__.
        rotors = algebra.exp_bivector(biv)               # [max_seq_len, C, D]
        rotors_rev = algebra.reverse(rotors)
        self.register_buffer("_R", rotors.contiguous(), persistent=False)
        self.register_buffer("_R_rev", rotors_rev.contiguous(), persistent=False)

    def forward(self, q_mv: Tensor, position_offset: int = 0) -> Tensor:
        """Apply per-(position, channel) rotor sandwich to multivectors.

        Args:
            q_mv: [B, H, S, C, D] multivector tensor.
            position_offset: optional starting position (for KV-cache).

        Returns:
            Rotated multivector with same shape.
        """
        B, H, S, C, D = q_mv.shape
        if position_offset + S > self.max_seq_len:
            raise ValueError(
                f"position_offset + S = {position_offset + S} exceeds "
                f"CliffordRotaryEmb.max_seq_len = {self.max_seq_len}"
            )
        # Per-position, per-channel rotors. Broadcast over (B, H).
        R = self._R[position_offset:position_offset + S].to(dtype=q_mv.dtype)
        R_rev = self._R_rev[position_offset:position_offset + S].to(dtype=q_mv.dtype)
        R = R[None, None, :, :, :]           # [1, 1, S, C, D]
        R_rev = R_rev[None, None, :, :, :]

        # Sandwich: out = R · q · R~ (geometric product is broadcast-friendly).
        tmp = self.algebra.geometric_product(R, q_mv)        # [B, H, S, C, D]
        out = self.algebra.geometric_product(tmp, R_rev)
        return out


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

    Optional multivector RoPE: when `use_rope=True`, per-position rotors are
    applied to Q and K before scoring (see CliffordRotaryEmb), giving the
    Clifford analogue of standard RoPE's relative-position encoding.
    """

    def __init__(self, algebra: CliffordAlgebra, n_embd: int,
                 n_heads: int, channels_per_head: int,
                 use_rope: bool = False, max_seq_len: int = 2048,
                 rope_base: float = 10000.0):
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

        self.use_rope = use_rope
        if use_rope:
            self.rope = CliffordRotaryEmb(
                algebra, channels_per_head=channels_per_head,
                max_seq_len=max_seq_len, base=rope_base,
            )
        else:
            self.rope = None

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

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

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
            use_rope=bool(getattr(config, "multivector_rope", False)),
            max_seq_len=int(getattr(config, "block_size", 2048)),
            rope_base=float(getattr(config, "multivector_rope_base", 10000.0)),
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
