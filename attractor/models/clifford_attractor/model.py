"""CliffordAttractor model: DEQ fixed-point over a stack of CliffordAttractorBlocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from attractor.models._ift import IFTAttach, IFTContext
from attractor.models.attractor.solvers import anderson_solve, fpi_solve

from .algebra import CliffordAlgebra
from .layers import CliffordAttractorBlock


# ========================================================================
#  DEQ Fixed-Point Solver with Implicit Differentiation
# ========================================================================


def _solve_fixed_point(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-4,
    anderson_m: int = 5,
    anderson_beta: float = 1.0,
) -> torch.Tensor:
    """Solve x* = f(x*) with DEQ autograd (constant memory).

    Forward uses the canonical batched Anderson solver from
    attractor.models.attractor.solvers; backward attaches IFT-based gradients
    via the shared IFTAttach autograd Function. Gradient flows back to x0 and,
    if f is a bound method of an nn.Module, to that module's parameters.
    """
    with torch.no_grad():
        x0_d = x0.detach()
        if anderson_m > 0 and max_iter >= 3:
            x_star_ng, _ = anderson_solve(
                f, x0_d, max_iter=max_iter, tol=tol,
                m=anderson_m, beta=anderson_beta,
            )
        else:
            x_star_ng, _ = fpi_solve(f, x0_d, max_iter=max_iter, tol=tol)

    y_s = x_star_ng.detach().requires_grad_(True)
    y_out = f(y_s)
    if not y_out.requires_grad:
        return x_star_ng.detach()

    params: Tuple[torch.Tensor, ...] = ()
    if hasattr(f, "__self__") and hasattr(f.__self__, "parameters"):
        params = tuple(p for p in f.__self__.parameters() if p.requires_grad)

    bw_kwargs = dict(
        bw_type="onestep",
        bw_max_iter=max(int(max_iter), 1),
        bw_min_iter=0,
        bw_tol=float(tol),
        anderson_m=int(anderson_m),
        anderson_beta=float(anderson_beta),
        adjoint_clip=1.0,
    )
    inputs_for_grad = (x0,) + params
    iftc = IFTContext(y_out, y_s, bw_kwargs, inputs_for_grad)
    return IFTAttach.apply(x_star_ng.detach(), iftc, *inputs_for_grad)


class DEQFixedPoint:
    """Back-compat shim. Use _solve_fixed_point or call .apply(...) directly.

    Historically a torch.autograd.Function subclass; now a thin dispatcher
    over the unified solver. The public surface is unchanged: callers do
    DEQFixedPoint.apply(f, x0, max_iter, tol, anderson_m, anderson_beta).
    """

    @staticmethod
    def apply(f, x0, max_iter, tol, anderson_m, anderson_beta):
        return _solve_fixed_point(f, x0, max_iter, tol, anderson_m, anderson_beta)


# ========================================================================
#  Config + Model
# ========================================================================


@dataclass
class CliffordAttractorConfig:
    """Configuration for the Clifford Attractor.

    Attributes:
        p, q, r: Clifford algebra signature Cl(p,q,r).
        channels: Number of multivector channels.
        hidden_channels: Hidden channel count (default = channels * 2).
        num_blocks: Depth of the fixed-point map f.
        num_rotors: Number of rotor heads.
        use_blade_selector: Enable per-grade gating.
        use_geometric_activation: Enable GeometricGELU.
        max_iter: Maximum fixed-point iterations.
        tol: Convergence tolerance.
        anderson_m: Anderson memory (0 = plain fixed-point).
        output_mode: 'scalar' extracts grade-0, 'linear' projects full MV.
        init_std: Rotor weight init std.
    """
    p: int = 3
    q: int = 0
    r: int = 0
    channels: int = 32
    hidden_channels: Optional[int] = None
    num_blocks: int = 4
    num_rotors: int = 4
    use_blade_selector: bool = True
    use_geometric_activation: bool = True
    max_iter: int = 50
    tol: float = 1e-4
    anderson_m: int = 5
    output_mode: str = "scalar"
    init_std: float = 0.01
    max_seq_len: int = 512
    use_sequence_mixer: bool = True


class CliffordAttractor(nn.Module):
    """Geometric Clifford/Rotor-based Attractor Model.

    Architecture:
        1. Token → multivector embedding.
        2. Fixed-point solve: X* = f(X*) with batched Anderson acceleration
           and implicit differentiation for gradients.
        3. Multivector → scalar output projection.

    The fixed-point map f is a stack of CliffordAttractorBlocks using
    rotor sandwiches, geometric products, and geometric activations.
    """

    def __init__(self, config: CliffordAttractorConfig, vocab_size: int = 11):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.algebra = CliffordAlgebra(config.p, config.q, config.r)

        self.token_embed = nn.Embedding(vocab_size, config.channels * self.algebra.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.channels * self.algebra.dim)
        self.use_sequence_mixer = config.use_sequence_mixer
        if self.use_sequence_mixer:
            self.sequence_mixer = nn.GRU(
                input_size=config.channels * self.algebra.dim,
                hidden_size=config.channels * self.algebra.dim,
                batch_first=True,
            )
            self.sequence_gate = nn.Parameter(torch.tensor(0.5))
        self.output_gate = nn.Parameter(torch.tensor(-2.0))

        hidden = config.hidden_channels or (config.channels * 2)
        self.blocks = nn.ModuleList([
            CliffordAttractorBlock(
                self.algebra, config.channels, hidden,
                use_geometric_activation=config.use_geometric_activation,
                use_blade_selector=config.use_blade_selector,
                init_std=config.init_std,
            )
            for _ in range(config.num_blocks)
        ])

        self.output_proj = nn.Linear(config.channels * self.algebra.dim, vocab_size)

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        x_mv = x.view(-1, self.config.channels, self.algebra.dim)
        for block in self.blocks:
            x_mv = block(x_mv)
        return x_mv.reshape(x.shape)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_solver_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        B, S = input_ids.shape
        if S > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len={self.config.max_seq_len}. "
                "Increase CliffordAttractorConfig.max_seq_len for longer contexts."
            )

        emb = self.token_embed(input_ids)
        pos = torch.arange(S, device=input_ids.device, dtype=torch.long)
        emb = emb + self.pos_embed(pos).unsqueeze(0)

        if self.use_sequence_mixer:
            mixed, _ = self.sequence_mixer(emb)
            mix_gate = torch.sigmoid(self.sequence_gate)
            emb = mix_gate * mixed + (1.0 - mix_gate) * emb

        x0 = emb.reshape(B * S, -1)

        cfg = self.config
        x_star = _solve_fixed_point(
            self._f, x0,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            anderson_m=cfg.anderson_m,
        )

        output_mix = torch.sigmoid(self.output_gate)
        x_out = output_mix * x_star + (1.0 - output_mix) * x0
        logits = self.output_proj(x_out)
        logits = logits.view(B, S, -1)

        if return_solver_stats:
            return logits, {}
        return logits


def create_clifford_attractor(
    p: int = 3,
    q: int = 0,
    channels: int = 32,
    num_blocks: int = 4,
    vocab_size: int = 11,
    **kwargs,
) -> CliffordAttractor:
    """Create a CliffordAttractor with given parameters."""
    config = CliffordAttractorConfig(p=p, q=q, channels=channels, num_blocks=num_blocks, **kwargs)
    return CliffordAttractor(config, vocab_size=vocab_size)
