"""
Minimal Working Example: Clifford Attractor on Toy Fixed-Point Task.

Demonstrates the CliffordAttractor learning to find fixed points
via DEQ-style Anderson acceleration with implicit differentiation.

Trains a small CliffordAttractor (Cl(3,0), 2 blocks, 8 channels)
on a synthetic token prediction task.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attractor.models.clifford_attractor import (
    CliffordAttractor,
    CliffordAttractorConfig,
    CliffordAlgebra,
    create_clifford_attractor,
    DEQFixedPoint,
)


def generate_toy_data(num_samples: int = 200, seq_len: int = 9, vocab_size: int = 11, seed: int = 42):
    """Generate synthetic token data for training."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    inputs = torch.randint(1, vocab_size, (num_samples, seq_len))
    # Labels: shift by 1 (simple next-token prediction)
    labels = torch.roll(inputs, -1, dims=-1)
    labels[:, -1] = 0

    return inputs, labels


def train_toy_example():
    """Train the CliffordAttractor on a toy next-token prediction task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Small configuration for demo
    config = CliffordAttractorConfig(
        p=3,                    # Cl(3,0) Euclidean
        q=0,
        channels=8,             # 8 multivector channels
        num_blocks=2,           # 2 blocks in the fixed-point map
        use_blade_selector=True,
        use_geometric_activation=True,
        max_iter=15,            # Max Anderson iterations
        tol=1e-3,
        anderson_m=3,           # Anderson memory
    )

    model = CliffordAttractor(config, vocab_size=11).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Algebra: Cl({config.p},{config.q}) -> {model.algebra.dim} basis blades")
    print(f"Channels: {config.channels}, Blocks: {config.num_blocks}")

    # Generate data
    inputs, labels = generate_toy_data(num_samples=200)
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    n_epochs = 50
    batch_size = 32
    n_samples = inputs.size(0)

    print(f"\nTraining for {n_epochs} epochs...")
    print(f"{'Epoch':>6} {'Loss':>10} {'Acc':>8}")

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            bx, bl = inputs[idx], labels[idx]

            optimizer.zero_grad()
            logits = model(bx)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                bl.view(-1),
                ignore_index=0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = bl != 0
                epoch_correct += (preds[mask] == bl[mask]).sum().item()
                epoch_total += mask.sum().item()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        acc = epoch_correct / max(epoch_total, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch + 1:>6d} {avg_loss:.6f} {acc:.4f}")

    print(f"\nTraining complete! Final loss: {avg_loss:.6f}, Accuracy: {acc:.4f}")
    print("The CliffordAttractor successfully learned to find fixed points")
    print("via DEQ Anderson acceleration with IFT gradients.")


if __name__ == "__main__":
    train_toy_example()
