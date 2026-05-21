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

# Two preset sizes:
#   "tiny" — runs on CPU in ~30s (default; the original demo).
#   "big"  — meaningful test on a single GPU (~10-20M params).
SIZE_PRESETS = {
    "tiny": dict(n_embd=128, num_attention_heads=4, intermediate_size=256,
                 n_backbone_layers=2, n_clifford_channels=16,
                 n_clifford_attn_heads=4, n_clifford_attn_channels_per_head=4),
    "big": dict(n_embd=384, num_attention_heads=6, intermediate_size=1536,
                n_backbone_layers=4, n_clifford_channels=48,
                n_clifford_attn_heads=6, n_clifford_attn_channels_per_head=8),
}


def build_model(vocab: int, arch: str = "clifford", size: str = "tiny",
                device: str = "cpu") -> torch.nn.Module:
    """Build a model. arch in {"attractor", "clifford", "native_clifford"}."""
    preset = SIZE_PRESETS[size]
    if arch == "attractor":
        cfg_name = "attractor-small-140m"
        cfg_cls = attractor.create_config
    else:
        cfg_name = ("native-clifford-small-140m" if arch == "native_clifford"
                    else "clifford-small-140m")
        cfg_cls = attractor.CliffordLMConfig.from_name

    cfg = cfg_cls(
        cfg_name,
        n_embd=preset["n_embd"],
        num_attention_heads=preset["num_attention_heads"],
        num_key_value_heads=preset["num_attention_heads"],
        intermediate_size=preset["intermediate_size"],
        block_size=64,
        vocab_size=vocab,
        padding_multiple=16,
        n_backbone_layers=preset["n_backbone_layers"],
        n_fp_blocks=1,
        solver="anderson",
        max_iter=6,
        min_iter=2,
        tol=1e-2,
        backward_type="onestep",
        use_fused_head="",
        init_strategy="scaled-zero",
        **({} if arch == "attractor" else dict(
            clifford_p=3, clifford_q=0,
            n_clifford_channels=preset["n_clifford_channels"],
            n_clifford_hidden=preset["n_clifford_channels"],
            n_clifford_attn_heads=preset["n_clifford_attn_heads"],
            n_clifford_attn_channels_per_head=preset["n_clifford_attn_channels_per_head"],
        )),
    )
    return cfg.construct_model().to(device)


# Back-compat for the previous main() signature.
def build_tiny_model(vocab: int, native: bool = False,
                     device: str = "cpu") -> torch.nn.Module:
    arch = "native_clifford" if native else "clifford"
    return build_model(vocab, arch=arch, size="tiny", device=device)


# ---- Training ----

def train(steps: int = 300, batch: int = 32, half_len: int = 8,
          vocab: int = 17, device: str = "cpu", log_every: int = 20,
          seed: int = 42, arch: str = "clifford", size: str = "tiny",
          ) -> dict:
    """Train one architecture on the copy task. Returns a summary dict."""
    torch.manual_seed(seed)
    sep_id = vocab - 1

    model = build_model(vocab, arch=arch, size=size, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__} ({arch}, {size}): {n_params:,} params, "
          f"vocab={vocab}, seq_len={2 * half_len + 1}, device={device}")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

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
    steps_per_sec = steps / elapsed
    print(f"\nDone in {elapsed:.1f}s ({steps_per_sec:.1f} steps/s)")

    peak_mem_gb = None
    if device.startswith("cuda"):
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"Peak VRAM: {peak_mem_gb:.2f} GB")

    # Demo: greedy copy a held-out prefix
    model.eval()
    with torch.no_grad():
        seq, _ = make_copy_batch(1, half_len, vocab, sep_id, device=device)
        prefix = seq[0, :half_len].tolist()
        out = model(seq, return_logits=True)
        preds = out["logits"][0, half_len:2 * half_len].argmax(dim=-1).tolist()
        n_correct = sum(p == t for p, t in zip(preds, prefix))
        print(f"\nCopy demo (held-out): prefix={prefix}")
        print(f"                       predicted={preds}")
        print(f"  → {n_correct}/{half_len} tokens copied correctly")

    return {
        "arch": arch,
        "losses": losses,
        "params": n_params,
        "elapsed": elapsed,
        "steps_per_sec": steps_per_sec,
        "peak_mem_gb": peak_mem_gb,
        "copy_correct": n_correct,
        "copy_total": half_len,
    }


# ---- Smoke assertion ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["attractor", "clifford", "native_clifford"],
                        default="clifford")
    parser.add_argument("--all", action="store_true",
                        help="Train all three architectures and print a side-by-side summary.")
    parser.add_argument("--size", choices=list(SIZE_PRESETS), default="tiny")
    parser.add_argument("--device", default="cpu", help="cpu or cuda or cuda:0 etc")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--half-len", type=int, default=8)
    args = parser.parse_args()

    archs = ["attractor", "clifford", "native_clifford"] if args.all else [args.arch]

    summaries = []
    for arch in archs:
        print(f"\n=== {arch} ({args.size}, device={args.device}) ===")
        summary = train(
            steps=args.steps, batch=args.batch, half_len=args.half_len,
            device=args.device, arch=arch, size=args.size,
        )
        losses = summary["losses"]
        early = sum(losses[:20]) / 20
        late = sum(losses[-20:]) / 20
        delta = early - late
        print(f"\nLoss decreased from {early:.3f} → {late:.3f}  (Δ={delta:.3f})")
        assert delta > 0.5, (
            f"{arch}: loss did not decrease enough (Δ={delta:.3f}); broken?"
        )
        summary["early"] = early
        summary["late"] = late
        summaries.append(summary)

    if len(summaries) > 1:
        print("\n=== Summary ===")
        header = (f"{'arch':<18s} {'params':>10s} {'steps/s':>9s} "
                  f"{'peak VRAM':>10s} {'loss start':>11s} {'loss end':>9s} "
                  f"{'copy':>5s}")
        print(header)
        for s in summaries:
            mem = f"{s['peak_mem_gb']:.2f} GB" if s['peak_mem_gb'] is not None else "    --   "
            print(f"{s['arch']:<18s} {s['params']:>10,d} "
                  f"{s['steps_per_sec']:>9.1f} "
                  f"{mem:>10s} {s['early']:>11.3f} {s['late']:>9.3f} "
                  f"{s['copy_correct']:>2d}/{s['copy_total']:<2d}")
    print("\nPASS: all selected model(s) learning the copy task.")


if __name__ == "__main__":
    main()
