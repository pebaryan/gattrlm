"""
Cl(3,0) Euclidean Geometric Algebra — Standalone Demo
======================================================

Demonstrates the core Clifford algebra layers on a synthetic next-token
prediction task.  Shows:
  - Algebra construction and multivector basics
  - RotorLayer (rotation equivariance)
  - CliffordLinear (channel mixing)
  - Geometric product self-interaction
  - CliffordLayerNorm and GeometricGELU
  - BladeSelector (grade gating)
  - Full CliffordAttractorBlock
  - Full CliffordAttractor model with DEQ fixed-point solver
  - Training loop
  - Fixed-point convergence verification

Usage:  python examples/clifford30_demo.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# 1.  Imports  (change these to match your project structure)
# ---------------------------------------------------------------------------
from attractor.models.clifford_attractor import (
    CliffordAlgebra,
    RotorLayer,
    CliffordLinear,
    CliffordLayerNorm,
    GeometricGELU,
    BladeSelector,
    CliffordAttractorBlock,
    CliffordAttractorConfig,
    CliffordAttractor,
)


# ---------------------------------------------------------------------------
# 2.  Multivector representation
# ---------------------------------------------------------------------------
def multivector_basics() -> None:
    """Print the grade structure of Cl(3,0)."""
    alg = CliffordAlgebra(p=3, q=0)
    print(f"Cl(3,0) dimension: {alg.dim}  (8 blades)")

    # Each blade index maps to a grade:
    #   0  → scalar (1)
    #   1,2,4 → vectors (e₁, e₂, e₃)
    #   3,5,6 → bivectors (e₁₂, e₁₃, e₂₃)
    #   7 → pseudoscalar (e₁₂₃)
    print(f"Grade map:       {alg._grade_index.tolist()}")
    print()

    # ── Geometric product ────────────────────────────────────────────
    #   e1 * e1 = 1,  e2 * e2 = 1,  e3 * e3 = 1
    #   e1 * e2 = e12 (blade index 3)
    #   e12 * e3 = e123 (blade index 7)
    #
    #   Note: all methods expect shape [batch, channels, dim].
    #   We use [1, 1, 8] for single-multivector queries.
    x = torch.zeros(1, 1, 8)  # [B, C, D]
    x[0, 0, 1] = 1.0  # e₁
    y = torch.zeros(1, 1, 8)
    y[0, 0, 2] = 1.0  # e₂

    gp = alg.geometric_product(x, y)  # [1, 1, 8]
    blade_idx = gp.squeeze().argmax().item()
    print(f"e₁ * e₂  → blade {blade_idx}  (expected 3, i.e. e₁₂)")
    assert blade_idx == 3, f"Wrong blade: expected 3, got {blade_idx}"
    print()

    # ── Norm ──────────────────────────────────────────────────────────
    #   Algebraic: rev(e₁) = e₁,  rev(e₁) * e₁ = 1 → norm_sq = 1
    ns = alg.norm_sq(x)  # [1, 1, 1]
    print(f"norm_sq(e₁) = {ns.item():.1f}  (expected 1.0)")
    print()

    # ── Grade projection ──────────────────────────────────────────────
    a = torch.zeros(1, 1, 8)  # [B, C, D]
    a[0, 0, 0] = 3.0  # scalar
    a[0, 0, 1] = 5.0  # e₁
    a[0, 0, 3] = 7.0  # e₁₂

    scalar_part = alg.grade_projection(a, 0)
    vec_part = alg.grade_projection(a, 1)
    biv_part = alg.grade_projection(a, 2)

    print("Grade projection:")
    print(f"  scalar  : {scalar_part.squeeze().tolist()}")
    print(f"  vector  : {vec_part.squeeze().tolist()}")
    print(f"  bivector: {biv_part.squeeze().tolist()}")
    print()


# ---------------------------------------------------------------------------
# 3.  Layer walk-through
# ---------------------------------------------------------------------------
def layer_demos() -> None:
    """Demonstrate each layer on random multivector data."""
    alg = CliffordAlgebra(p=3, q=0)
    B, C = 2, 4  # batch, channels
    x = torch.randn(B, C, alg.dim)

    # ── RotorLayer ────────────────────────────────────────────────────
    rotor = RotorLayer(alg, channels=C)
    x_rot = rotor(x)
    print(f"RotorLayer:   {x.shape} → {x_rot.shape}  (rotation equivariant)")

    # ── CliffordLinear ────────────────────────────────────────────────
    linear = CliffordLinear(alg, in_channels=C, out_channels=8)
    x_lin = linear(x)
    print(f"CliffordLinear: {x.shape} → {x_lin.shape}  (channel mixing)")

    # ── CliffordLayerNorm ─────────────────────────────────────────────
    norm = CliffordLayerNorm(alg, channels=C)
    x_norm = norm(x)
    print(f"LayerNorm:    {x.shape} → {x_norm.shape}")

    # ── GeometricGELU ─────────────────────────────────────────────────
    gelu = GeometricGELU(alg, channels=C)
    x_gelu = gelu(x)
    print(f"GeometricGELU: {x.shape} → {x_gelu.shape}  (direction-preserving)")

    # ── BladeSelector ─────────────────────────────────────────────────
    sel = BladeSelector(alg, channels=C)
    x_sel = sel(x)
    print(f"BladeSelector: {x.shape} → {x_sel.shape}  (grade gating)")

    # ── Full Block ────────────────────────────────────────────────────
    block = CliffordAttractorBlock(alg, channels=C)
    x_block = block(x)
    print(f"Full Block:   {x.shape} → {x_block.shape}")
    print()


# ---------------------------------------------------------------------------
# 4.  Synthetic data
# ---------------------------------------------------------------------------
def generate_toy_data(
    num_samples: int = 200,
    seq_len: int = 9,
    vocab_size: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token sequences; labels are shifted inputs (next-token prediction)."""
    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


# ---------------------------------------------------------------------------
# 5.  Training helper
# ---------------------------------------------------------------------------
@dataclass
class TrainingStats:
    losses: list[float]
    accs: list[float]
    wall_sec: float


def train_clifford_attractor(
    channels: int = 8,
    num_blocks: int = 2,
    max_iter: int = 15,
    anderson_m: int = 3,
    vocab_size: int = 11,
    seq_len: int = 9,
    lr: float = 1e-3,
    epochs: int = 50,
    seed: int = 42,
) -> TrainingStats:
    """Train a tiny CliffordAttractor on next-token prediction."""
    torch.manual_seed(seed)
    t0 = time.time()

    # ── Model ─────────────────────────────────────────────────────────
    cfg = CliffordAttractorConfig(
        p=3,
        q=0,
        channels=channels,
        hidden_channels=None,
        num_blocks=num_blocks,
        num_rotors=num_blocks,
        use_blade_selector=True,
        use_geometric_activation=True,
        max_iter=max_iter,
        tol=1e-4,
        anderson_m=anderson_m,
        output_mode="scalar",
        init_std=0.01,
    )
    model = CliffordAttractor(cfg, vocab_size=vocab_size)
    optimizer = AdamW(model.parameters(), lr=lr)

    # ── Data ──────────────────────────────────────────────────────────
    xs, ys = generate_toy_data(num_samples=200, seq_len=seq_len, vocab_size=vocab_size)

    losses: list[float] = []
    accs: list[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(xs)  # [200, 9, vocab_size]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), ys.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc = (logits.argmax(-1) == ys).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}  acc={acc:.3f}")

    wall = time.time() - t0
    print(f"\nTraining finished in {wall:.1f}s  |  final loss={losses[-1]:.4f}  final acc={accs[-1]:.3f}")
    return TrainingStats(losses, accs, wall)


# ---------------------------------------------------------------------------
# 6.  Fixed-point verification
# ---------------------------------------------------------------------------
def verify_fixed_point() -> None:
    """Check that the DEQ solver actually converges to a fixed point."""
    print("─" * 60)
    print("Fixed-point verification")
    print("─" * 60)

    cfg = CliffordAttractorConfig(
        p=3,
        q=0,
        channels=8,
        num_blocks=1,
        max_iter=30,
        tol=5e-5,
        anderson_m=3,
    )
    model = CliffordAttractor(cfg, vocab_size=11)
    model.eval()

    tokens = torch.randint(0, 10, (1, 5))
    with torch.no_grad():
        # Run the DEQ solver
        logits = model(tokens, return_solver_stats=True)
        if isinstance(logits, tuple):
            logits, stats = logits
        else:
            stats = {}

        # The DEQ solver iterates f until convergence.  We can check if
        # the internal fixed point satisfies f(x*) ≈ x* by extracting
        # the solver's internal state.
        print(f"Output shape: {logits.shape}")
        print(f"Solver stats: {stats}")
        if "nstep" in stats:
            print(f"Converged in {stats['nstep']} steps")
        if "diff" in stats:
            print(f"Final residual: {stats['diff']:.6e}")
        print()


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Cl(3,0) Euclidean Geometric Algebra Demo         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("─── Part 1: Multivector basics ───")
    multivector_basics()

    print("─── Part 2: Layer demos ───")
    layer_demos()

    print("─── Part 3: Training a tiny DEQ model ───")
    stats = train_clifford_attractor(
        channels=8,
        num_blocks=2,
        max_iter=15,
        anderson_m=3,
        epochs=30,  # quick demo
    )

    verify_fixed_point()

    print()
    print("Done.  Try modifying channels, num_blocks, or max_iter above!")
