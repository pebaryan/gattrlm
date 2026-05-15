"""Train *vanilla* TRM (TinyRecursiveReasoningModel_ACTV1) on Sudoku-Extreme.

This is a faithful port of TRM's training loop from
TinyRecursiveModels/pretrain.py, adapted to:
  * read the puzzle dataset from the layout we already built under
    ``experiments/eqlm_sudoku/data/puzzle_dataset.py``
  * substitute AdamW for the (build-heavy) adam-atan2 optimiser; the paper
    text explicitly says they use AdamW with β1=0.9, β2=0.95
  * skip Hydra/coolname/wandb plumbing -- we use a flat dataclass + jsonargparse
    and write metrics to a JSONL file alongside the checkpoint.

Everything that affects the *model* (TRM block code, deep supervision via
ACT carry, EMA, sparse puzzle embeddings + SignSGD, stablemax CE, q-halt
loss, eval halting) is unchanged.

Usage (2 GPUs):
    torchrun --standalone --nproc_per_node=2 \\
        -m experiments.eqlm_sudoku.trm.train_trm \\
        --data_dir /scratch1/feinashl/data/sudoku-extreme-1k-aug-1000 \\
        --out_dir  /scratch1/feinashl/eqlm_sudoku/run-trm-att-7m \\
        --variant att --epochs 50000 --eval_interval 5000 \\
        --global_batch_size 768
"""
import copy
import json
import math
import os
import random
import socket
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Allow being run from anywhere
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.eqlm_sudoku.data.puzzle_dataset import (  # noqa: E402
    IGNORE_LABEL_ID,
    PuzzleDataset,
    PuzzleDatasetConfig,
)
from experiments.eqlm_sudoku.trm.trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1,
)
from experiments.eqlm_sudoku.trm.losses import ACTLossHead  # noqa: E402
from experiments.eqlm_sudoku.trm.ema import EMAHelper  # noqa: E402
from experiments.eqlm_sudoku.trm.sparse_embedding import (  # noqa: E402
    CastedSparseEmbeddingSignSGD_Distributed,
)


# --------------------------------------------------------------------------- #
# CLI / config
# --------------------------------------------------------------------------- #

@dataclass
class TrainTRMConfig:
    data_dir: str = ""
    out_dir: str = ""
    variant: str = "att"  # "att" or "mlp_t"

    # TRM hyperparameters (defaults match README's pretrain_att_sudoku)
    H_cycles: int = 3
    L_cycles: int = 6
    H_layers: int = 0
    L_layers: int = 2
    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "rope"
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True
    forward_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Optim (matches paper: AdamW β1=0.9 β2=0.95)
    epochs: int = 50_000
    eval_interval: int = 5_000
    global_batch_size: int = 768
    seed: int = 0
    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2_000
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1.0
    puzzle_emb_lr: float = 1e-4
    puzzle_emb_weight_decay: float = 1.0

    # EMA
    ema: bool = True
    ema_rate: float = 0.999

    # Loss
    loss_type: str = "stablemax_cross_entropy"

    # Logging
    log_interval: int = 50
    keep_last_only: bool = True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _rank_world():
    if _is_dist():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _seed_all(seed: int, rank: int):
    s = seed + rank
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def _build_dataloader(cfg: TrainTRMConfig, split: str, rank: int, world: int) -> DataLoader:
    pcfg = PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=[cfg.data_dir],
        global_batch_size=cfg.global_batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=cfg.eval_interval if split == "train" else 1,
        rank=rank,
        num_replicas=world,
    )
    ds = PuzzleDataset(pcfg, split=split)
    return DataLoader(
        ds, batch_size=None, num_workers=1, prefetch_factor=8,
        pin_memory=True, persistent_workers=True,
    )


def _make_model(cfg: TrainTRMConfig, metadata, rank: int, world: int):
    """Build TRM + ACT loss head, mirroring TinyRecursiveModels/pretrain.py:create_model."""
    cfg_dict = dict(
        batch_size=cfg.global_batch_size // world,
        seq_len=int(metadata.seq_len),
        vocab_size=int(metadata.vocab_size),
        num_puzzle_identifiers=int(metadata.num_puzzle_identifiers),
        H_cycles=cfg.H_cycles,
        L_cycles=cfg.L_cycles,
        H_layers=cfg.H_layers,
        L_layers=cfg.L_layers,
        hidden_size=cfg.hidden_size,
        expansion=cfg.expansion,
        num_heads=cfg.num_heads,
        pos_encodings=cfg.pos_encodings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        halt_max_steps=cfg.halt_max_steps,
        halt_exploration_prob=cfg.halt_exploration_prob,
        forward_dtype=cfg.forward_dtype,
        mlp_t=(cfg.variant == "mlp_t"),
        puzzle_emb_len=cfg.puzzle_emb_len,
        no_ACT_continue=cfg.no_ACT_continue,
        # TRM uses puzzle_emb_ndim=hidden_size in cfg yaml
        puzzle_emb_ndim=cfg.hidden_size,
    )
    with torch.device("cuda"):
        inner = TinyRecursiveReasoningModel_ACTV1(cfg_dict)
        if rank == 0:
            print(inner)
        model = ACTLossHead(inner, loss_type=cfg.loss_type)
    return model, cfg_dict


def _build_optimizers(cfg: TrainTRMConfig, model, world: int):
    """Match TRM's optimiser groups: SignSGD for puzzle_emb buffers, AdamW for the rest.

    TRM uses adam-atan2; the paper text says AdamW (β1=0.9, β2=0.95). Substituting
    AdamW avoids the build-from-source dependency without changing the optimisation
    story per the paper.
    """
    inner = model.model  # ACTLossHead -> TinyRecursiveReasoningModel_ACTV1
    optim_signsgd = CastedSparseEmbeddingSignSGD_Distributed(
        inner.puzzle_emb.buffers(),
        lr=0.0,
        weight_decay=cfg.puzzle_emb_weight_decay,
        world_size=world,
    )
    optim_adamw = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        fused=torch.cuda.is_available(),
    )
    return [optim_signsgd, optim_adamw], [cfg.puzzle_emb_lr, cfg.lr]


def _cosine_warmup_lr(current_step: int, base_lr: float, num_warmup: int,
                      num_total: int, min_ratio: float) -> float:
    """Match TRM's cosine schedule with linear warmup
    (cosine_schedule_with_warmup_lr_lambda from pretrain.py)."""
    if current_step < num_warmup:
        return base_lr * float(current_step) / float(max(1, num_warmup))
    progress = float(current_step - num_warmup) / float(max(1, num_total - num_warmup))
    return base_lr * (
        min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    )


def _train_step(cfg: TrainTRMConfig, model, batch: dict, carry, optimizers,
                optimizer_lrs, step: int, total_steps: int, world: int, rank: int):
    """Mirror of TinyRecursiveModels/pretrain.py:train_batch."""
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    if carry is None:
        with torch.device("cuda"):
            carry = model.module.initial_carry(batch) if hasattr(model, "module") else model.initial_carry(batch)

    new_carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

    ((1 / cfg.global_batch_size) * loss).backward()

    lr_this = None
    for opt, base_lr in zip(optimizers, optimizer_lrs):
        lr_this = _cosine_warmup_lr(step, base_lr, cfg.lr_warmup_steps,
                                    total_steps, cfg.lr_min_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr_this
        opt.step()
        opt.zero_grad()

    reduced = None
    if metrics:
        keys = sorted(metrics.keys())
        vals = torch.stack([metrics[k] for k in keys])
        if world > 1:
            dist.reduce(vals, dst=0)
        if rank == 0:
            v = vals.cpu().numpy()
            d = {k: v[i] for i, k in enumerate(keys)}
            count = max(d.get("count", 1), 1)
            reduced = {f"train/{k}": (val / (cfg.global_batch_size if k.endswith("loss") else count))
                       for k, val in d.items()}
            reduced["train/lr"] = lr_this

    return new_carry, reduced


@torch.inference_mode()
def _eval(cfg: TrainTRMConfig, model, eval_loader: DataLoader, world: int, rank: int):
    """Mirror of TinyRecursiveModels/pretrain.py:evaluate (no evaluators).
    Each test batch: rebuild carry, then loop inner forwards until all_finish."""
    inner = model.module if hasattr(model, "module") else model
    metric_sums: dict = {}
    metric_count: dict = {}
    n_batches = 0
    n_inference_steps_total = 0

    for set_name, batch, global_batch_size in eval_loader:
        n_batches += 1
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.device("cuda"):
            carry = inner.initial_carry(batch)
        steps = 0
        while True:
            carry, _loss, metrics, _preds, all_finish = model(
                carry=carry, batch=batch, return_keys=[]
            )
            steps += 1
            if all_finish:
                break
        n_inference_steps_total += steps

        # Aggregate
        for k, v in metrics.items():
            metric_sums[k] = metric_sums.get(k, torch.zeros_like(v)) + v
            metric_count[k] = metric_count.get(k, 0) + 1

    # All-reduce across ranks
    if world > 1:
        for k, v in metric_sums.items():
            dist.all_reduce(v)

    out = {}
    if rank == 0 and metric_sums:
        count = max(float(metric_sums.get("count", torch.tensor(1.0))), 1.0)
        for k, v in metric_sums.items():
            out[f"eval/{k}"] = float(v.item() / count if k != "count" else v.item())
        out["eval/n_batches"] = n_batches
        out["eval/avg_inference_steps"] = n_inference_steps_total / max(n_batches, 1)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(cfg: TrainTRMConfig):
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = _rank_world()
    _seed_all(cfg.seed, rank)
    out_dir = Path(cfg.out_dir)

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        print(f"[{time.ctime()}] host={socket.gethostname()} world={world}")
        print(f"  variant={cfg.variant}  data_dir={cfg.data_dir}")
        print(f"  out_dir={cfg.out_dir}")

    train_loader = _build_dataloader(cfg, "train", rank, world)
    test_loader = _build_dataloader(cfg, "test", rank, world)
    metadata = train_loader.dataset.metadata
    if rank == 0:
        print(f"  metadata: {metadata.model_dump()}")

    model, cfg_dict = _make_model(cfg, metadata, rank, world)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"  TRM-{cfg.variant} params: {n/1e6:.3f}M")
        print(f"  H_cycles={cfg.H_cycles} L_cycles={cfg.L_cycles}  L_layers={cfg.L_layers}  "
              f"halt_max_steps={cfg.halt_max_steps}")
        with open(out_dir / "model_config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

    # Broadcast params from rank 0 (match TRM's pretrain.py init)
    if world > 1:
        with torch.no_grad():
            for p in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(p, src=0)

    optimizers, optimizer_lrs = _build_optimizers(cfg, model, world)

    # DDP wrap — find_unused_parameters needed because ACT halting is stochastic
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=True, broadcast_buffers=False,
        )

    # EMA over (non-puzzle-emb) trainable parameters
    ema = EMAHelper(mu=cfg.ema_rate) if cfg.ema else None
    if ema is not None:
        ema.register(model)

    # Total optim steps (for LR schedule). Matches TRM's pretrain.py:
    #   total_steps = epochs * total_groups * mean_puzzle_examples / global_batch_size
    total_steps = int(
        cfg.epochs * metadata.total_groups * metadata.mean_puzzle_examples
        / cfg.global_batch_size
    )
    if rank == 0:
        print(f"  total optim steps target = {total_steps:,}")

    train_t0 = time.time()
    last_log_t = time.time()
    step = 0
    train_epochs_per_iter = cfg.eval_interval
    total_iters = max(cfg.epochs // train_epochs_per_iter, 1)
    carry = None

    for outer in range(total_iters):
        if rank == 0:
            print(f"[{time.ctime()}] Outer iter {outer}/{total_iters} (epoch={outer * train_epochs_per_iter})")
        model.train()
        for set_name, batch, _ in train_loader:
            carry, reduced = _train_step(
                cfg, model, batch, carry, optimizers, optimizer_lrs,
                step, total_steps, world, rank,
            )
            if ema is not None:
                ema.update(model)

            step += 1
            if rank == 0 and step % cfg.log_interval == 0:
                dt = time.time() - last_log_t
                last_log_t = time.time()
                elapsed = time.time() - train_t0
                steps_per_s = step / max(elapsed, 1e-6)
                eta_s = max(total_steps - step, 0) / max(steps_per_s, 1e-6)
                eta_h, eta_m = divmod(int(eta_s), 3600)
                eta_m = eta_m // 60
                msg = f"  step={step:>7d}"
                if reduced:
                    for k in ("train/lm_loss", "train/q_halt_loss", "train/exact_accuracy",
                              "train/accuracy", "train/steps", "train/lr"):
                        if k in reduced:
                            msg += f"  {k.split('/')[-1]}={reduced[k]:.4f}"
                msg += f"  dt/{cfg.log_interval}={dt:.2f}s eta={eta_h:d}h{eta_m:02d}m"
                print(msg)

        # ---- eval at end of outer iter (use EMA copy if enabled) -----------
        if ema is not None:
            saved = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            ema.ema(model)
        try:
            metrics = _eval(cfg, model, test_loader, world, rank)
        finally:
            if ema is not None:
                for n, p in model.named_parameters():
                    if n in saved:
                        p.data.copy_(saved[n])

        if rank == 0:
            print(f"[{time.ctime()}] EVAL step={step}  metrics={metrics}")
            with open(out_dir / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "epoch": outer * train_epochs_per_iter, **metrics}) + "\n")

            inner = model.module if hasattr(model, "module") else model
            ckpt_path = out_dir / ("last.pt" if cfg.keep_last_only else f"step-{step:08d}.pt")
            payload = {
                "step": step,
                "model_state": inner.state_dict(),
                "ema_state": ema.state_dict() if ema is not None else None,
                "model_config": cfg_dict,
                "train_config": asdict(cfg),
                "metrics": metrics,
            }
            torch.save(payload, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

    if _is_dist():
        dist.destroy_process_group()
    if rank == 0:
        print(f"[{time.ctime()}] training complete.")


if __name__ == "__main__":
    from jsonargparse import CLI
    cfg: TrainTRMConfig = CLI(TrainTRMConfig, as_positional=False)  # type: ignore[arg-type]
    main(cfg)
