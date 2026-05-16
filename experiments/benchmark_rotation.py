"""
Benchmark: Learning 3D Rotor Compositions
=========================================
Compares gattrlm (CliffordAttractor + DEQ) vs gaflowlm (CFSTransformerBlock)
on a geometric regression task: predict R(v) = R v R~ given input vector v
and a sequence of random rotors.

Both models use Cl(3,0) - 8D multivectors.
Matched parameter counts (~20K). Same data. Same training budget.
"""

import sys
import os
import time
import math
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Path setup ----------------------------------------------------------
GATTRLM_PATH = "D:/code/gattrlm"
GAFLOWLM_PATH = "D:/code/gaflowlm"
for p in [GATTRLM_PATH, GAFLOWLM_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Reproducibility ------------------------------------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =======================================================================
# 1. DATA GENERATION - Random 3D rotations in Cl(3,0)
# =======================================================================

BLADE_ORDER = ["1", "e1", "e2", "e12", "e3", "e13", "e23", "e123"]

def make_rotor_data(
    algebra,
    n_samples: int = 5000,
    n_rotations: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random vectors and apply n_rotations random rotors.

    Returns:
        x:  [n_samples, 8] - input multivector (vector in e1,e2,e3)
        y:  [n_samples, 8] - output after R v R~ composition
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n_samples, 8)
    y = torch.zeros(n_samples, 8)

    for i in range(n_samples):
        # Random unit vector (embedded in blades 1,2,4 for e1,e2,e3)
        v = torch.randn(3, generator=g)
        v = v / (v.norm() + 1e-8)
        v_mv = torch.zeros(8)
        v_mv[1] = v[0].item()  # e1
        v_mv[2] = v[1].item()  # e2
        v_mv[4] = v[2].item()  # e3
        x[i] = v_mv.clone()

        # Apply a composition of random rotors
        result = v_mv.unsqueeze(0).unsqueeze(0)  # [1, 1, 8]
        for _ in range(n_rotations):
            # Random rotation axis (unit vector)
            axis = torch.randn(3, generator=g)
            axis = axis / (axis.norm() + 1e-8)
            angle = torch.rand(1, generator=g).item() * 2.0 * math.pi

            # Build bivector: angle * (axis_e1*e23 + axis_e2*e31 + axis_e3*e12)
            # In Cl(3,0), the bivector basis is: e12(idx=3), e13(idx=5), e23(idx=6)
            # A rotation by angle theta around unit axis (a1,a2,a3) has bivector:
            #   B = theta * (a1*e23 + a2*e31 + a3*e12)
            # rotor R = exp(B/2)
            biv = torch.zeros(1, 1, 8)
            biv[0, 0, 3] = angle * axis[2].item() / 2.0   # e12 component
            biv[0, 0, 5] = -angle * axis[1].item() / 2.0  # e13 component (e31 = -e13)
            biv[0, 0, 6] = angle * axis[0].item() / 2.0   # e23 component
            R = algebra.exp_bivector(biv)

            # Sandwich: R * v * R~
            R_rev = algebra.reverse(R)
            result = algebra.sandwich_product(R, result, R_rev)

        y[i] = result.squeeze()

    return x, y


def blade_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss on vector blades (e1, e2, e3) only."""
    # Blades 1, 2, 4 are e1, e2, e3
    vec_mask = torch.tensor([0, 1, 1, 0, 1, 0, 0, 0], device=pred.device, dtype=torch.bool)
    diff = (pred - target)[..., vec_mask]
    return diff.pow(2).mean()


# =======================================================================
# 2. MODEL WRAPPERS
# =======================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -- 2a. Gattrlm Regressor  ---------------------------------------------
class GattrlmRotationRegressor(nn.Module):
    """Maps [B, 1, 8] -> [B, 1, 8] using CliffordAttractorBlock + DEQ.

    Architecture:
        Input -> Linear(8 -> C*8) -> reshape(B, C, 8)
            -> [DEQ(f, x0)] -> reshape(B, C*8) -> Linear(C*8 -> 8) -> Output
    where f is a stack of N CliffordAttractorBlocks.

    The DEQ solver iterates f to find the fixed point x* = f(x*),
    giving implicit depth proportional to convergence iterations.
    """

    def __init__(
        self,
        algebra,
        channels: int = 6,
        hidden_channels: int = 12,
        num_blocks: int = 2,
        use_deq: bool = True,
        max_iter: int = 10,
        tol: float = 1e-3,
        anderson_m: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.use_deq = use_deq
        self.max_iter = max_iter
        self.tol = tol
        self.anderson_m = anderson_m

        self.input_proj = nn.Linear(algebra.dim, channels * algebra.dim)

        # Build the DEQ block stack
        from attractor.models.clifford_attractor import (
            CliffordAttractorBlock, DEQFixedPoint
        )
        self.DEQFixedPoint = DEQFixedPoint
        self.blocks = nn.ModuleList([
            CliffordAttractorBlock(
                algebra, channels, hidden_channels,
                use_geometric_activation=True,
                use_blade_selector=True,
            )
            for _ in range(num_blocks)
        ])

        self.output_proj = nn.Linear(channels * algebra.dim, algebra.dim)

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        """Fixed-point map: apply all blocks."""
        h = x
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] where D=8 for Cl(3,0). L=1 in our benchmark."""
        B, L, D = x.shape

        # Project to channel space
        h = self.input_proj(x)                    # [B, L, C*D]
        h = h.view(B * L, self.channels, D)       # [B*L, C, D]

        if self.use_deq:
            # DEQ solver finds fixed point x* = f(x*)
            h = self.DEQFixedPoint.apply(
                self._f, h, self.max_iter, self.tol,
                self.anderson_m, 1.0,
            )
        else:
            # Just stack blocks (no DEQ)
            for _ in range(self.max_iter):
                h = self._f(h)

        h = h.view(B, L, self.channels * D)       # [B, L, C*D]
        out = self.output_proj(h)                  # [B, L, D]
        return out


# -- 2b. Gaflowlm Regressor  --------------------------------------------
class GaflowlmRotationRegressor(nn.Module):
    """Maps [B, L, 8] -> [B, L, 8] using CFSTransformerBlocks.

    Uses stacked Transformer blocks with Clifford Frame Attention (CFA),
    which incorporates geometric products into attention computations.
    """

    def __init__(
        self,
        engine,
        mv_dim: int = 8,
        n_heads: int = 2,
        ff_dim: int = 128,
        num_blocks: int = 8,
        dropout: float = 0.0,
        use_higher_order: bool = False,
    ):
        super().__init__()
        from gaflowlm.models.cfs_arch import CFSTransformerBlock

        self.blocks = nn.ModuleList([
            CFSTransformerBlock(
                mv_dim=mv_dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                engine=engine,
                dropout=dropout,
                use_higher_order=use_higher_order,
            )
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(mv_dim, mv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, mv_dim]"""
        h = x
        for block in self.blocks:
            h = block(h)
        out = self.output_proj(h)
        return out


# =======================================================================
# 3. TRAINING LOOP
# =======================================================================

def train_model(
    model: nn.Module,
    name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    verbose: bool = True,
    log_every: int = 10,
) -> dict:
    """Train a model on the rotation prediction task.

    Returns:
        History dict with 'epoch', 'train_loss', 'test_loss', 'time'.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_train = x_train.shape[0]
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_vec_loss": [],
        "test_vec_loss": [],
        "time_per_epoch": [],
    }

    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        epoch_vec_loss = 0.0
        n_batches = 0

        epoch_start = time.time()
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            bx = x_train[idx].to(device).unsqueeze(1)    # [B, 1, 8]
            by = y_train[idx].to(device).unsqueeze(1)    # [B, 1, 8]

            optimizer.zero_grad()
            pred = model(bx)

            loss = F.mse_loss(pred, by)
            vec_loss = blade_mse(pred, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_vec_loss += vec_loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Evaluate
        model.eval()
        with torch.no_grad():
            tx = x_test.to(device).unsqueeze(1)
            ty = y_test.to(device).unsqueeze(1)
            tpred = model(tx)
            test_loss = F.mse_loss(tpred, ty).item()
            test_vec_loss = blade_mse(tpred, ty).item()

        avg_loss = epoch_loss / n_batches
        avg_vec_loss = epoch_vec_loss / n_batches

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["test_loss"].append(test_loss)
        history["train_vec_loss"].append(avg_vec_loss)
        history["test_vec_loss"].append(test_vec_loss)
        history["time_per_epoch"].append(epoch_time)

        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={avg_loss:.6f} | "
                f"test_loss={test_loss:.6f} | "
                f"test_vec_loss={test_vec_loss:.6f} | "
                f"{epoch_time:.2f}s"
            )

    total_time = time.time() - total_start
    history["total_time"] = total_time
    print(f"  Total: {total_time:.1f}s ({total_time / n_epochs:.2f}s/epoch)")

    return history


# =======================================================================
# 4. MAIN
# =======================================================================

def main():
    print("=" * 65)
    print("  Benchmark: Learning 3D Rotations with GA Models")
    print("=" * 65)

    # -- Generate data ----------------------------------------------------
    print("\n[1] Generating synthetic rotation data...")
    from attractor.models.clifford_attractor import CliffordAlgebra

    algebra = CliffordAlgebra(3, 0)
    print(f"    Algebra: Cl(3,0), dim={algebra.dim}, blades: {algebra.dim}")

    # Generate data with 3 sequential rotor applications
    n_train, n_test = 4000, 1000
    x_all, y_all = make_rotor_data(algebra, n_samples=n_train + n_test,
                                    n_rotations=3, seed=42)
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    print(f"    Train: {x_train.shape[0]} samples")
    print(f"    Test:  {x_test.shape[0]} samples")

    # Check data is sensible
    vec_diff = (y_train[:, [1, 2, 4]] - x_train[:, [1, 2, 4]]).norm(dim=1)
    print(f"    Avg |delta_vector| after rotation: {vec_diff.mean().item():.3f} "
          f"(should be > 0)")

    # -- Build models ----------------------------------------------------
    print("\n[2] Building models with matched parameter counts (~20K)...")

    # Gattrlm: CliffordAttractorBlock + DEQ
    # Design: C channels, 2 blocks, hidden_channels for internal expansion
    # Target ~20K params
    gattrlm_model = GattrlmRotationRegressor(
        algebra,
        channels=38,
        hidden_channels=76,
        num_blocks=2,
        use_deq=True,
        max_iter=10,
        tol=1e-3,
        anderson_m=3,
    )
    gattrlm_params = count_params(gattrlm_model)
    print(f"    Gattrlm (CliffordAttractor+DEQ): {gattrlm_params:,} params")

    # Gaflowlm: CFSTransformerBlock stack
    from gaflowlm.clifford.engine import CliffordEngine
    engine = CliffordEngine(k=3)
    # Convert engine tables to float32 to avoid mixed-dtype errors in LayerNorm
    # Also must convert internal sparse-signs buffers used by geometric_product
    float64_attrs = [
        'cayley', 'bivector_mask', 'grade_masks', 'pseudoscalar_mask',
        'reverse_signs', 'scalar_mask', 'vector_mask',
        '_gp_signs', '_inner_signs', '_wedge_signs',
    ]
    for attr in float64_attrs:
        t = getattr(engine, attr, None)
        if t is not None and t.dtype == torch.float64:
            setattr(engine, attr, t.float())
    print(f"    Gaflowlm CliffordEngine: {engine.n} blades (dtype={engine.cayley.dtype})")

    gaflowlm_model = GaflowlmRotationRegressor(
        engine,
        mv_dim=8,
        n_heads=4,
        ff_dim=128,
        num_blocks=8,
        dropout=0.0,
        use_higher_order=False,
    )
    gaflowlm_params = count_params(gaflowlm_model)
    print(f"    Gaflowlm (CFSTransformerBlock): {gaflowlm_params:,} params")
    print(f"    Ratio: {max(gattrlm_params, gaflowlm_params) / min(gattrlm_params, gaflowlm_params):.2f}x")

    # -- Train models ----------------------------------------------------
    print("\n[3] Training models...")

    n_epochs = 100
    histories = {}

    for model_cls, name, model in [
        (GattrlmRotationRegressor, "gattrlm", gattrlm_model),
        (GaflowlmRotationRegressor, "gaflowlm", gaflowlm_model),
    ]:
        print(f"\n  -- {name} --")
        try:
            hist = train_model(
                model, name,
                x_train, y_train,
                x_test, y_test,
                n_epochs=n_epochs,
                lr=1e-3,
                batch_size=64,
                log_every=10,
            )
            histories[name] = hist
        except Exception as e:
            print(f"  ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()

    # -- Report results --------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    if histories:
        print(f"\n{'Metric':<30} {'gattrlm':<20} {'gaflowlm':<20}")
        print("-" * 70)

        for name in histories:
            h = histories[name]
            final_train = h["train_loss"][-1]
            final_test = h["test_loss"][-1]
            final_vec = h["test_vec_loss"][-1]
            total_time = h["total_time"]
            best_test = min(h["test_loss"])
            best_vec = min(h["test_vec_loss"])

            n_params = count_params(locals()[f"{name}_model"])

            print(f"\n  {name}:")
            print(f"  {'Parameters':<30} {n_params:<20,}")
            print(f"  {'Total time':<30} {total_time:<20.1f}s")
            print(f"  {'Final train loss (MSE)':<30} {final_train:<20.6f}")
            print(f"  {'Final test loss (MSE)':<30} {final_test:<20.6f}")
            print(f"  {'Final test vec loss (MSE)':<30} {final_vec:<20.6f}")
            print(f"  {'Best test loss':<30} {best_test:<20.6f}")
            print(f"  {'Best test vec loss':<30} {best_vec:<20.6f}")

            # Determine final epoch for loss threshold comparison
            threshold = 1e-3
            epoch_to_threshold = None
            for e, l in enumerate(h["test_loss"]):
                if l < threshold:
                    epoch_to_threshold = e
                    break
            if epoch_to_threshold is not None:
                print(f"  {'Epoch to test_loss < 1e-3':<30} {epoch_to_threshold:<20}")
            else:
                print(f"  {'Epoch to test_loss < 1e-3':<30} {'Not reached':<20}")

        # Per-epoch comparison
        print(f"\n  Per-epoch test loss:")
        print(f"  {'Epoch':<8} {'gattrlm':<20} {'gaflowlm':<20}")
        print(f"  {'-'*50}")
        for i in range(0, n_epochs, max(1, n_epochs // 10)):
            g = histories["gattrlm"]["test_loss"][i] if "gattrlm" in histories else float('nan')
            f = histories["gaflowlm"]["test_loss"][i] if "gaflowlm" in histories else float('nan')
            print(f"  {i:<8} {g:<20.6f} {f:<20.6f}")
        # Final
        g = histories["gattrlm"]["test_loss"][-1] if "gattrlm" in histories else float('nan')
        f = histories["gaflowlm"]["test_loss"][-1] if "gaflowlm" in histories else float('nan')
        print(f"  {'final':<8} {g:<20.6f} {f:<20.6f}")

    # -- Save results ----------------------------------------------------
    print("\n  Saving results...")
    results = {
        "config": {
            "n_train": n_train,
            "n_test": n_test,
            "n_rotations": 3,
            "n_epochs": n_epochs,
            "gattrlm_params": count_params(gattrlm_model),
            "gaflowlm_params": count_params(gaflowlm_model),
        },
        "histories": {
            name: {
                k: (v if isinstance(v, list) else v)
                for k, v in h.items()
            }
            for name, h in histories.items()
        },
    }
    torch.save(results, "experiments/benchmark_rotation_results.pt")
    print("  Saved to experiments/benchmark_rotation_results.pt")

    print("\n" + "=" * 65)
    print("  Done!")
    print("=" * 65)


if __name__ == "__main__":
    main()
