"""Sample text from a trained CliffordLM checkpoint.

Loads a Fabric-saved checkpoint, rebuilds the model from model_config.json,
strips the usual prefixes from the state dict, and runs ModelingCliffordLM's
generate() loop.

Usage:
    python experiments/sample_clifford.py \\
        --run_dir D:/code/gattrlm/runs/clifford-small-140m-wikitext-long \\
        --prompt "The cat sat on" \\
        --max_new 80
"""

import argparse
import json
import sys
import time
from pathlib import Path

# The training checkpoint's pickle references modules that live in scripts/
# (e.g. `cost`), so put that on sys.path before torch.load.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch

from attractor.models.clifford_lm import CliffordLMConfig
from attractor.tokenizer import Tokenizer
from receval.models.clifford import ModelingCliffordLM


def find_checkpoint(run_dir: Path, which: str = "best") -> Path:
    ck_dir = run_dir / "checkpoints-SingleDeviceStrategy"
    if not ck_dir.exists():
        # Try other strategies too
        for d in run_dir.iterdir():
            if d.name.startswith("checkpoints-"):
                ck_dir = d
                break
    candidates = sorted(ck_dir.glob(f"{which}-*"))
    if not candidates:
        # fall back to anything in there
        candidates = sorted(ck_dir.iterdir())
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {ck_dir}")
    return candidates[0]


def clean_state_dict(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        for prefix in ["module.", "_orig_mod.", "model.", "_forward_module."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--which", default="best", choices=["best", "last"])
    p.add_argument("--prompt", default="The cat sat on")
    p.add_argument("--prompts_file", default=None)
    p.add_argument("--max_new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--greedy", action="store_true",
                   help="Greedy decoding (disables sampling)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_steps", type=int, default=None,
                   help="Override DEQ solver max_iter at inference time")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    print(f"Run dir: {run_dir}")

    with open(run_dir / "model_config.json") as f:
        cfg_dict = json.load(f)
    # Drop fields that aren't constructor args; CliffordLMConfig will
    # re-derive them in __post_init__.
    for k in ["init", "_class_name", "head_size", "n_head", "n_query_groups",
              "n_layers_in_prelude", "n_layers_in_recurrent_block",
              "n_layers_in_coda", "mean_recurrence", "mean_backprop_depth",
              "n_layer"]:
        cfg_dict.pop(k, None)
    if isinstance(cfg_dict.get("rope_settings"), dict):
        from attractor.models.config import RoPESettings
        cfg_dict["rope_settings"] = RoPESettings(**cfg_dict["rope_settings"])
    cfg = CliffordLMConfig(**cfg_dict)
    print(f"Model: {cfg.name}  ({cfg.n_embd}d, "
          f"backbone={cfg.n_layers_in_prelude}+fp={cfg.n_layers_in_recurrent_block}, "
          f"Cl({cfg.clifford_p},{cfg.clifford_q},{cfg.clifford_r}))")

    model = ModelingCliffordLM(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built ModelingCliffordLM: {n_params / 1e6:.1f}M params")

    ckpt_path = find_checkpoint(run_dir, which=args.which)
    print(f"Checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    sd = clean_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded; missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"  missing examples: {missing[:3]}")
    if unexpected[:3]:
        print(f"  unexpected examples: {unexpected[:3]}")
    model.eval()

    tok = Tokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
    if tok.pad_id is None:
        tok.pad_id = 0
    print(f"Tokenizer vocab: {tok.vocab_size}")

    if args.num_steps is not None:
        model.apply_eval_solver(max_iter=args.num_steps)
        print(f"Solver max_iter overridden to {args.num_steps}")

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        prompts = [args.prompt]

    do_sample = not args.greedy
    print()
    for i, prompt in enumerate(prompts):
        ids = tok.encode(prompt)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ctx = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ctx,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                top_k=args.top_k if do_sample else None,
                do_sample=do_sample,
                use_cache=False,  # safer with the DEQ head
            )
        dt = time.time() - t0
        new_tokens = out[0, len(ids):].tolist()
        new_text = tok.decode(new_tokens)
        print(f"[{i + 1}/{len(prompts)}]  {len(new_tokens)} tok in {dt:.2f}s  "
              f"({len(new_tokens) / max(dt, 1e-6):.1f} tok/s)  mode="
              f"{'sample' if do_sample else 'greedy'}")
        print(f"  prompt:     {prompt!r}")
        print(f"  continued:  {new_text!r}")
        print()

    if hasattr(model, "solver_summary"):
        stats = model.solver_summary()
        if stats.get("calls", 0):
            print(f"DEQ solver: mean_iters={stats['mean_iters']:.1f}  "
                  f"mean_rel_res={stats['mean_rel_res']:.2e}  "
                  f"frac_converged={stats['frac_converged']:.2%}")


if __name__ == "__main__":
    main()
