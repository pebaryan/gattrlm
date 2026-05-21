"""Rotation-equivariance benchmark: MLP vs CliffordAttractor.

Task
----
Predict y = R * v * R~ for a 3D vector v and a rotor R. Both v and R are
encoded as 8-blade Cl(3,0) multivectors (vector v in blades e1,e2,e3;
rotor R in {1, e12, e13, e23}). The model sees the concatenated (v, R)
features (16 floats) and must output the rotated vector y.

Two generalization probes:

1. Angle extrapolation. Train on rotation angles in [0, pi/2]; test on
   [pi/2, pi]. A truly rotation-equivariant operator handles any angle,
   so it should extrapolate.

2. Axis extrapolation. Train on rotations whose axis is in a single 30°
   cone around +z; test on rotations with axes drawn uniformly from the
   whole sphere. An equivariant model handles any axis by construction;
   an MLP must memorize the training axis distribution.

Both probes hold the model fixed after training and measure MSE on the
vector blades of the output. We also report parameter counts and total
training time so the comparison is on a level field.

Run:
    python experiments/benchmark_equivariance.py            # CPU is fine
    python experiments/benchmark_equivariance.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from attractor.models.clifford_attractor import (
    CliffordAlgebra,
    CliffordAttractorBlock,
    _solve_fixed_point,
)
from attractor.models.clifford_lm.native import CliffordSelfAttention


# Indices of e1, e2, e3 in Cl(3,0)
VEC_IDX = (1, 2, 4)
# Indices of 1, e12, e13, e23 (scalar + bivectors) — the rotor subspace
ROTOR_IDX = (0, 3, 5, 6)


# ============================================================================
# Data generation
# ============================================================================

def random_unit_vec(n: int, gen: torch.Generator) -> torch.Tensor:
    """[n, 3] uniformly distributed on the unit sphere."""
    v = torch.randn(n, 3, generator=gen)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def axis_angle_to_rotor(
    algebra: CliffordAlgebra, axis: torch.Tensor, angle: torch.Tensor,
) -> torch.Tensor:
    """Build the rotor R = exp(-(angle/2) * B) where B is the unit bivector
    dual to `axis` in Cl(3,0).

    axis: [..., 3] (unit), angle: [...] in radians.
    Returns: [..., 8] multivector with non-zero entries on {1, e12, e13, e23}.
    """
    # The unit bivector dual to axis (a1, a2, a3) is
    #   B = a3 e12 - a2 e13 + a1 e23
    # so that the corresponding rotor exp(-theta/2 * B) rotates by theta about axis.
    biv = torch.zeros(*axis.shape[:-1], algebra.dim, device=axis.device, dtype=axis.dtype)
    biv[..., 3] = axis[..., 2]
    biv[..., 5] = -axis[..., 1]
    biv[..., 6] = axis[..., 0]
    biv = -0.5 * angle.unsqueeze(-1) * biv
    return algebra.exp_bivector(biv)


def vec_to_mv(algebra: CliffordAlgebra, v: torch.Tensor) -> torch.Tensor:
    """[..., 3] vectors -> [..., 8] vector multivectors in Cl(3,0)."""
    mv = torch.zeros(*v.shape[:-1], algebra.dim, device=v.device, dtype=v.dtype)
    mv[..., 1] = v[..., 0]
    mv[..., 2] = v[..., 1]
    mv[..., 4] = v[..., 2]
    return mv


def apply_rotor(
    algebra: CliffordAlgebra, R: torch.Tensor, x_mv: torch.Tensor,
) -> torch.Tensor:
    """y = R x R~  (full multivector input is fine)."""
    return algebra.sandwich_product(R, x_mv, algebra.reverse(R))


def sample_axis_in_cone(n: int, cone_deg: float, gen: torch.Generator) -> torch.Tensor:
    """Uniform unit vectors inside a cone of half-angle `cone_deg` around +z.

    cone_deg=180 recovers the full sphere; cone_deg=0 gives only +z.
    """
    cos_max = math.cos(math.radians(cone_deg))
    u = torch.rand(n, generator=gen) * (1 - cos_max) + cos_max     # cos(theta) in [cos_max, 1]
    sin_t = (1 - u * u).clamp_min(0.0).sqrt()
    phi = torch.rand(n, generator=gen) * 2 * math.pi
    return torch.stack([sin_t * phi.cos(), sin_t * phi.sin(), u], dim=-1)


def make_dataset(
    algebra: CliffordAlgebra, n: int,
    angle_lo: float, angle_hi: float,
    axis_cone_deg: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (input_features, target_vec_mv) pairs.

    input_features: [n, 16] = concat(v_mv [8], R [8])
    target_vec_mv:  [n,  8] = (R v R~)
    """
    gen = torch.Generator().manual_seed(seed)
    v = random_unit_vec(n, gen)
    v_mv = vec_to_mv(algebra, v)

    axis = sample_axis_in_cone(n, axis_cone_deg, gen)
    angle = torch.rand(n, generator=gen) * (angle_hi - angle_lo) + angle_lo
    R = axis_angle_to_rotor(algebra, axis, angle)
    y_mv = apply_rotor(algebra, R, v_mv)

    x_features = torch.cat([v_mv, R], dim=-1)  # [n, 16]
    return x_features, y_mv


# ============================================================================
# Models
# ============================================================================

class MLPBaseline(nn.Module):
    """Plain MLP. No geometric structure."""

    def __init__(self, in_dim: int = 16, out_dim: int = 8, hidden: int = 96, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CliffordRegressor(nn.Module):
    """CliffordAttractor-style regressor.

    Lifts (v, R) -> C multivector channels, applies a stack of
    CliffordAttractorBlocks under a DEQ solver (rotor sandwich, geometric
    product, geometric GELU, blade selector), then reads the first channel
    back out as the predicted multivector.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int = 8,
        hidden_channels: int = 8,
        num_blocks: int = 2,
        max_iter: int = 8,
        tol: float = 1e-3,
        anderson_m: int = 3,
    ):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.max_iter = max_iter
        self.tol = tol
        self.anderson_m = anderson_m
        D = algebra.dim

        # Lift two multivectors (v, R) -> C channels (kept on per-blade level
        # via per-blade Linear-like channel mixing; we use a single Linear
        # which is a per-blade-flattened map for simplicity).
        self.input_proj = nn.Linear(2 * D, channels * D, bias=False)

        self.blocks = nn.ModuleList([
            CliffordAttractorBlock(
                algebra, channels, hidden_channels,
                use_geometric_activation=True,
                use_blade_selector=True,
            )
            for _ in range(num_blocks)
        ])

        # Project C channels -> 1 output multivector
        self.output_proj = nn.Linear(channels * D, D, bias=False)

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        # x flat [B, C*D] -> reshape [B, C, D]
        x_mv = x.view(-1, self.channels, self.algebra.dim)
        for blk in self.blocks:
            x_mv = blk(x_mv)
        return x_mv.reshape(x.shape)

    def forward(self, x_features: torch.Tensor) -> torch.Tensor:
        # x_features: [B, 16]
        h = self.input_proj(x_features)             # [B, C*D]
        h_star = _solve_fixed_point(
            self._f, h,
            max_iter=self.max_iter,
            tol=self.tol,
            anderson_m=self.anderson_m,
            anderson_beta=1.0,
        )
        return self.output_proj(h_star)             # [B, D]


class NativeCliffordRegressor(nn.Module):
    """Multi-layer transformer-style regressor using CliffordSelfAttention.

    The 16-d input (concat of v_mv and R, each 8-blade Cl(3,0) multivectors)
    is reshaped to a 2-token sequence of 8-d features (so the two
    multivectors are two positions). A stack of pre-norm blocks applies
    Clifford self-attention (multivector Q/K/V, grade-0 attention scores)
    between the two tokens, plus a feed-forward MLP. The first token's
    final hidden state is projected to the 8-d output multivector.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        n_embd: int = 8,
        n_heads: int = 4,
        channels_per_head: int = 4,
        n_layers: int = 4,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.n_embd = n_embd
        D = algebra.dim
        self.D = D
        assert n_embd == D, ("This regressor treats each multivector as one "
                             "scalar token of width algebra.dim, so n_embd "
                             "must equal algebra.dim.")

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict(dict(
                norm1=nn.LayerNorm(n_embd),
                attn=CliffordSelfAttention(algebra, n_embd, n_heads, channels_per_head),
                norm2=nn.LayerNorm(n_embd),
                ff=nn.Sequential(
                    nn.Linear(n_embd, n_embd * ff_mult),
                    nn.GELU(),
                    nn.Linear(n_embd * ff_mult, n_embd),
                ),
            ))
            self.layers.append(block)

        self.out_norm = nn.LayerNorm(n_embd)
        self.out_proj = nn.Linear(n_embd, D)

    def forward(self, x_features: torch.Tensor) -> torch.Tensor:
        # x_features: [B, 16] = concat(v_mv [8], R [8])
        B = x_features.shape[0]
        x = x_features.view(B, 2, self.n_embd)  # [B, 2, 8]
        for blk in self.layers:
            x = x + blk["attn"](blk["norm1"](x))
            x = x + blk["ff"](blk["norm2"](x))
        # Read the first token (the "v" slot)
        h = self.out_norm(x[:, 0])              # [B, 8]
        return self.out_proj(h)                  # [B, 8]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ============================================================================
# Training + evaluation
# ============================================================================

def vec_blade_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over the three vector blades only (we only care about y = vector)."""
    idx = torch.tensor(VEC_IDX, device=pred.device)
    p = pred.index_select(-1, idx)
    t = target.index_select(-1, idx)
    return (p - t).pow(2).mean()


def train(
    model: nn.Module, name: str,
    x_train: torch.Tensor, y_train: torch.Tensor,
    epochs: int, batch_size: int, lr: float, device: str,
    verbose: bool = True,
) -> list[float]:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = x_train.shape[0]
    losses: list[float] = []
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss, batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = x_train[idx]
            by = y_train[idx]
            opt.zero_grad()
            pred = model(bx)
            loss = vec_blade_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        sched.step()
        losses.append(epoch_loss / max(batches, 1))
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  [{name}] epoch {epoch:3d}  train_vec_mse={losses[-1]:.6f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
    print(f"  [{name}] {epochs} epochs in {time.time() - t0:.1f}s")
    return losses


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 512) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            pred = model(x[i:i + batch])
            losses.append(vec_blade_loss(pred, y[i:i + batch]).item())
    return sum(losses) / max(len(losses), 1)




# ============================================================================
# Main
# ============================================================================

def run_probe(name: str, x_train, y_train, eval_sets: dict[str, tuple],
              builders: dict[str, callable], *,
              epochs: int, batch: int, device: str):
    print("\n" + "#" * 70)
    print(f"# Probe: {name}")
    print("#" * 70)

    models = {}
    for model_name, build in builders.items():
        m = build()
        models[model_name] = m
        print(f"  {model_name:<22} : {count_params(m):>7,} params")

    timings = {}
    for model_name, model in models.items():
        print(f"  Training {model_name} for {epochs} epochs ...")
        t0 = time.time()
        train(model, model_name, x_train, y_train,
              epochs=epochs, batch_size=batch, lr=1e-3, device=device, verbose=False)
        timings[model_name] = time.time() - t0

    rows = {}
    for model_name, model in models.items():
        rows[model_name] = {k: evaluate(model, xs, ys) for k, (xs, ys) in eval_sets.items()}

    # Print a wide table.
    name_w = max(len(n) for n in builders) + 2
    print()
    header = f"  {'split':<22}" + "".join(f"{n:>{name_w + 4}}" for n in builders)
    print(header)
    print("  " + "-" * (22 + (name_w + 4) * len(builders)))
    for split in eval_sets:
        line = f"  {split:<22}"
        for model_name in builders:
            line += f"{rows[model_name][split]:>{name_w + 4}.6f}"
        print(line)

    # Headline gap (extrapol / in-dist) per model.
    in_key = next(iter(eval_sets))
    out_key = list(eval_sets)[-1]
    print(f"\n  Extrapolation gap ({out_key} / {in_key}):")
    for model_name in builders:
        gap = rows[model_name][out_key] / max(rows[model_name][in_key], 1e-12)
        print(f"    {model_name:<22} {gap:>7.2f}x")

    print(f"\n  Wall-clock: " + ", ".join(
        f"{n} {timings[n]:.1f}s" for n in builders
    ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    print("=" * 70)
    print("Rotation-equivariance benchmark: MLP vs CliffordAttractor")
    print("=" * 70)
    print(f"Device: {device}")

    algebra = CliffordAlgebra(3, 0).to(device)

    # Builders. We target ~10-21k params per arm — the absolute counts vary
    # but the geometric architectures are deliberately given fewer params,
    # which makes the equivariance result a lower bound on their advantage.
    def build_mlp():
        return MLPBaseline(in_dim=16, out_dim=8, hidden=96, depth=3)

    def build_clif():
        return CliffordRegressor(
            algebra,
            channels=24, hidden_channels=48, num_blocks=2,
            max_iter=8, tol=1e-3, anderson_m=3,
        )

    def build_native():
        return NativeCliffordRegressor(
            algebra,
            n_embd=algebra.dim,
            n_heads=4,
            channels_per_head=4,
            n_layers=4,
            ff_mult=2,
        )

    builders = {
        "MLP": build_mlp,
        "Clifford": build_clif,
        "NativeClifford": build_native,
    }

    # --- Probe 1: angle extrapolation ---------------------------------------
    print("\n[Probe 1] Angle extrapolation:")
    print("          Train angles [0, pi/2]; test [0, pi/2] vs [pi/2, pi].")
    print("          Axes uniform over the sphere in both train and test.")
    x_train, y_train = make_dataset(algebra, args.n_train,
                                    0.0, math.pi / 2, axis_cone_deg=180,
                                    seed=args.seed)
    x_train, y_train = x_train.to(device), y_train.to(device)
    eval_sets_angle = {}
    xs, ys = make_dataset(algebra, args.n_test, 0.0, math.pi / 2,
                          axis_cone_deg=180, seed=args.seed + 1)
    eval_sets_angle["in-dist (0..pi/2)"] = (xs.to(device), ys.to(device))
    xs, ys = make_dataset(algebra, args.n_test, math.pi / 2, math.pi,
                          axis_cone_deg=180, seed=args.seed + 2)
    eval_sets_angle["extrapol (pi/2..pi)"] = (xs.to(device), ys.to(device))
    rows_angle = run_probe("Angle extrapolation",
                           x_train, y_train, eval_sets_angle,
                           builders,
                           epochs=args.epochs, batch=args.batch, device=device)

    # --- Probe 2: axis extrapolation ---------------------------------------
    print("\n[Probe 2] Axis extrapolation:")
    print("          Train rotation axes inside a 30° cone around +z.")
    print("          Test (a) same cone, (b) full sphere.")
    print("          Angles are uniform [0, pi] for both train and test.")
    x_train, y_train = make_dataset(algebra, args.n_train, 0.0, math.pi,
                                    axis_cone_deg=30, seed=args.seed + 10)
    x_train, y_train = x_train.to(device), y_train.to(device)
    eval_sets_axis = {}
    xs, ys = make_dataset(algebra, args.n_test, 0.0, math.pi,
                          axis_cone_deg=30, seed=args.seed + 11)
    eval_sets_axis["in-dist (30° cone)"] = (xs.to(device), ys.to(device))
    xs, ys = make_dataset(algebra, args.n_test, 0.0, math.pi,
                          axis_cone_deg=180, seed=args.seed + 12)
    eval_sets_axis["extrapol (sphere)"] = (xs.to(device), ys.to(device))
    rows_axis = run_probe("Axis extrapolation",
                          x_train, y_train, eval_sets_axis,
                          builders,
                          epochs=args.epochs, batch=args.batch, device=device)

    # --- Headline summary ---------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — extrapolation gap (extrapol MSE / in-dist MSE; lower is better)")
    print("=" * 70)

    def gap(rows, in_key, out_key):
        return {name: rows[name][out_key] / max(rows[name][in_key], 1e-12)
                for name in builders}

    g_angle = gap(rows_angle, "in-dist (0..pi/2)", "extrapol (pi/2..pi)")
    g_axis = gap(rows_axis, "in-dist (30° cone)", "extrapol (sphere)")

    name_w = max(len(n) for n in builders) + 2

    def fmt_row(label, gaps):
        out = f"  {label:<22}"
        for n in builders:
            out += f"{gaps[n]:>{name_w + 3}.2f}x"
        return out

    print(f"  {'probe':<22}" + "".join(f"{n:>{name_w + 4}}" for n in builders))
    print("  " + "-" * (22 + (name_w + 4) * len(builders)))
    print(fmt_row("angle", g_angle))
    print(fmt_row("axis", g_axis))


if __name__ == "__main__":
    main()
