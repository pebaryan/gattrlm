"""Toy training demo for CliffordLM.

Trains a tiny CliffordLM on a copy task (predict the right half of the
sequence from the left half) and prints a loss curve. The copy task is
a useful smoke test for an LM: it requires attention to work (no attention
=> can't copy), and a working model drops the loss from ~log(vocab) toward
near-zero within a few hundred steps.

Usage:
    python experiments/clifford_lm_toy_train.py
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import attractor


# ---- Task: copy ----

def make_copy_batch(batch_size: int, half_len: int, vocab: int, sep_id: int,
                    device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (input_ids, labels) for the copy task.

    Sequence: [a_1, ..., a_L, SEP, a_1, ..., a_L]  (length 2L+1)
    Labels are -100 (ignored by CE) on the left half and SEP position;
    real labels on the copy half so we only score how well the model
    repeats the prefix.
    """
    prefix = torch.randint(0, vocab - 1, (batch_size, half_len), device=device)
    sep = torch.full((batch_size, 1), sep_id, device=device, dtype=torch.long)
    seq = torch.cat([prefix, sep, prefix], dim=1)

    labels = torch.full_like(seq, -100)
    # Predicting token at position i comes from the logits at position i-1
    # (next-token convention), so the targets for the copy half live at
    # positions L .. 2L (i.e. the SEP slot and the L-1 copy tokens).
    labels[:, half_len:2 * half_len] = prefix
    return seq, labels


# ---- Model ----

def build_tiny_model(vocab: int, native: bool = False,
                     device: str = "cpu") -> attractor.CliffordLM:
    """Build a tiny CliffordLM (or NativeCliffordLM if native=True)."""
    name = "native-clifford-small-140m" if native else "clifford-small-140m"
    cfg = attractor.CliffordLMConfig.from_name(
        name,
        # Shrink for CPU speed
        n_embd=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        block_size=64,
        vocab_size=vocab,
        padding_multiple=16,
        n_backbone_layers=2,
        n_fp_blocks=1,
        # Clifford MLP: Cl(3,0), 16 channels * 8 blades = 128 hidden
        clifford_p=3, clifford_q=0,
        n_clifford_channels=16,
        n_clifford_hidden=16,
        # Clifford attention (only used when native=True)
        n_clifford_attn_heads=4,
        n_clifford_attn_channels_per_head=4,
        # DEQ solver
        solver="anderson",
        max_iter=6,
        min_iter=2,
        tol=1e-2,
        backward_type="onestep",
        # No fused head (no triton on CPU)
        use_fused_head="",
        # Stability
        init_strategy="scaled-zero",
    )
    model = cfg.construct_model().to(device)
    return model


# ---- Training ----

def train(steps: int = 300, batch: int = 32, half_len: int = 8,
          vocab: int = 17, device: str = "cpu", log_every: int = 20,
          seed: int = 42, native: bool = False) -> list[float]:
    torch.manual_seed(seed)
    sep_id = vocab - 1

    model = build_tiny_model(vocab, native=native, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {n_params:,} params, vocab={vocab}, "
          f"seq_len={2 * half_len + 1}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    losses: list[float] = []

    t0 = time.time()
    for step in range(1, steps + 1):
        seq, labels = make_copy_batch(batch, half_len, vocab, sep_id, device=device)
        out = model(seq, labels=labels)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))

        if step == 1 or step % log_every == 0 or step == steps:
            iters = model._last_solver_info.get("iters", "?")
            rel = model._last_solver_info.get("rel_residual", 0.0)
            print(f"step {step:4d}  loss={loss.item():.4f}  "
                  f"fp_iters={iters}  fp_rel={rel:.2e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({steps / elapsed:.1f} steps/s)")

    # Demo: greedy copy a held-out prefix
    model.eval()
    with torch.no_grad():
        seq, _ = make_copy_batch(1, half_len, vocab, sep_id, device=device)
        prefix = seq[0, :half_len].tolist()
        # Run forward and decode greedy from the SEP onward
        out = model(seq, return_logits=True)
        preds = out["logits"][0, half_len:2 * half_len].argmax(dim=-1).tolist()
        n_correct = sum(p == t for p, t in zip(preds, prefix))
        print(f"\nCopy demo (held-out): prefix={prefix}")
        print(f"                       predicted={preds}")
        print(f"  → {n_correct}/{half_len} tokens copied correctly")

    return losses


# ---- Smoke assertion ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true",
                        help="Use NativeCliffordLM (Clifford attention + Clifford MLP)")
    parser.add_argument("--both", action="store_true",
                        help="Train both hybrid and native; print a side-by-side summary.")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    runs = []
    if args.both:
        configs = [("hybrid (CliffordLM)", False), ("native (NativeCliffordLM)", True)]
    else:
        label = "native (NativeCliffordLM)" if args.native else "hybrid (CliffordLM)"
        configs = [(label, args.native)]

    for label, native in configs:
        print(f"\n=== {label} ===")
        losses = train(steps=args.steps, native=native)
        early = sum(losses[:20]) / 20
        late = sum(losses[-20:]) / 20
        delta = early - late
        print(f"\nLoss decreased from {early:.3f} → {late:.3f}  (Δ={delta:.3f})")
        assert delta > 0.5, (
            f"{label}: loss did not decrease enough (Δ={delta:.3f}); "
            "model may be broken or stuck."
        )
        runs.append((label, early, late, delta))

    if len(runs) > 1:
        print("\n=== Summary ===")
        for label, early, late, delta in runs:
            print(f"  {label:<32s}  loss {early:6.3f} → {late:6.3f}  Δ={delta:.3f}")
    print("\nPASS: model(s) learning the copy task.")


if __name__ == "__main__":
    main()
