"""Implicit Function Theorem (IFT) gradient attachment for DEQ fixed points.

Shared by Attractor and CliffordAttractor: the forward solver runs under
no_grad (any contractive iteration scheme works), then this module attaches
the IFT-based adjoint gradient to the fixed-point tensor via a custom
autograd Function. Routing differentiable inputs through the Function input
list (rather than register_hook) is what makes DDP's per-parameter hooks
fire on those inputs, which is required for them to receive valid gradients.
"""

from typing import Optional, Sequence

import torch
from torch import Tensor


def _maybe_clip_jv(Jv: Tensor, v: Tensor, adjoint_clip: Optional[float]) -> Tensor:
    """Rescale J^T v when its per-sample norm exceeds adjoint_clip * ||v||.
    Keeps the Neumann-1 adjoint approximation safe if the head drifts toward
    a less contractive regime during training."""
    if adjoint_clip is None:
        return Jv
    B = Jv.size(0)
    v_norm = v.reshape(B, -1).norm(dim=1).clamp_min(1e-12)
    Jv_norm = Jv.reshape(B, -1).norm(dim=1)
    bound = float(adjoint_clip) * v_norm
    scale = torch.where(
        Jv_norm > bound,
        bound / Jv_norm.clamp_min(1e-12),
        torch.ones_like(Jv_norm),
    )
    return Jv * scale.view(B, *([1] * (Jv.ndim - 1)))


class IFTContext:
    """Side-channel used by IFTAttach to hold tensors that carry an autograd
    graph but should not be passed as Function inputs (doing so would cause
    double-backward errors when the outer engine re-traverses them)."""

    def __init__(self, y_out: Tensor, y_s: Tensor, bw_kwargs: dict,
                 inputs_for_grad: Sequence[Tensor]):
        self.y_out = y_out
        self.y_s = y_s
        self.bw_kwargs = bw_kwargs
        self.inputs_for_grad = tuple(inputs_for_grad)


class IFTAttach(torch.autograd.Function):
    """Attach IFT-based gradients to a fixed point y* of f.

    Forward is a no-op (returns y_star_value unchanged). Backward solves the
    adjoint system (I - J_f^T) u = v, then propagates u through f's graph to
    compute gradients for every tensor listed in iftc.inputs_for_grad. The
    trailing *grad_inputs positional args MUST be the same tensors (in the
    same order) — passing them through the Function input list is what makes
    DDP's per-parameter hooks fire on each of them.

    bw_kwargs keys:
      bw_type: "onestep" (Neumann-1), "anderson", or "picard"
      bw_max_iter, bw_min_iter, bw_tol: iteration limits for non-onestep
      anderson_m, anderson_beta: only used when bw_type == "anderson"
      adjoint_clip: only used when bw_type == "onestep"
    """

    @staticmethod
    def forward(ctx, y_star_value, iftc: "IFTContext", *grad_inputs):
        ctx.iftc = iftc
        return y_star_value

    @staticmethod
    def backward(ctx, grad_y_star):
        iftc: IFTContext = ctx.iftc
        y_out, y_s, kw = iftc.y_out, iftc.y_s, iftc.bw_kwargs
        v = grad_y_star.contiguous()

        if kw["bw_type"] == "onestep":
            (Jv,) = torch.autograd.grad(
                y_out, y_s, v, retain_graph=True, create_graph=False
            )
            Jv = _maybe_clip_jv(Jv, v, kw["adjoint_clip"])
            u = Jv + v
        else:
            def T_op(u):
                (Ju,) = torch.autograd.grad(
                    y_out, y_s, u, retain_graph=True, create_graph=False
                )
                return Ju + v

            if kw["bw_type"] == "anderson":
                from attractor.models.attractor.solvers import anderson_solve
                with torch.no_grad():
                    u, _ = anderson_solve(
                        T_op,
                        v.detach().clone(),
                        max_iter=kw["bw_max_iter"],
                        tol=kw["bw_tol"],
                        m=kw["anderson_m"],
                        beta=kw["anderson_beta"],
                        min_iter=kw["bw_min_iter"],
                    )
            else:  # "picard"
                u = v.detach().clone()
                for it in range(kw["bw_max_iter"]):
                    u_new = T_op(u)
                    diff = (u_new - u).reshape(u.size(0), -1).norm(dim=1)
                    ref = u_new.reshape(u.size(0), -1).norm(dim=1).clamp_min(1e-9)
                    u = u_new
                    if (it + 1) >= kw["bw_min_iter"] and \
                       (diff / ref).max().item() < kw["bw_tol"]:
                        break

        all_inputs = iftc.inputs_for_grad
        diff_targets = tuple(t for t in all_inputs if t.requires_grad)
        if diff_targets and y_out.requires_grad:
            try:
                diff_grads = torch.autograd.grad(
                    y_out, diff_targets, u,
                    retain_graph=False, create_graph=False, allow_unused=True,
                )
            except RuntimeError as e:
                states = ", ".join(
                    f"{i}:rg={t.requires_grad},leaf={t.is_leaf}"
                    for i, t in enumerate(diff_targets)
                )
                raise RuntimeError(
                    f"{e}\nIFTAttach.backward: y_out.rg={y_out.requires_grad}, "
                    f"n_targets={len(diff_targets)}, states=[{states}]"
                ) from e
            grad_lookup = dict(zip(map(id, diff_targets), diff_grads))
        else:
            grad_lookup = {}

        def _grad_for(t: Tensor):
            g = grad_lookup.get(id(t), None)
            return torch.zeros_like(t) if g is None else g

        input_grads = tuple(
            _grad_for(t) if t.requires_grad else None for t in all_inputs
        )
        return (None, None) + input_grads
