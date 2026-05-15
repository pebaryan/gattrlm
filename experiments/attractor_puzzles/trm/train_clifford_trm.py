"""
Clifford Attractor Training Script for Puzzle Tasks (Sudoku, Maze, ARC).

Extends the Attractor Puzzles TRM training loop to use CliffordAttractor
with geometric rotor-based attractor dynamics instead of standard DEQ.

Usage:
    torchrun --nnodes=1 --nproc-per-node=1 experiments/attractor_puzzles/trm/train_clifford_trm.py
        --dataset-paths /path/to/sudoku-extreme
        --output-dir ./outputs/clifford-attractor-sudoku
        --p 3 --q 0 --channels 16 --num-blocks 3
        --lr 1e-3 --deq-max-iter 30 --jacobian-reg-lambda 0.01
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from attractor.models.clifford_attractor import (
    CliffordAttractor,
    CliffordAttractorConfig,
    create_clifford_attractor,
    _fixed_point_iteration,
    _anderson_acceleration,
)
from experiments.attractor_puzzles.data.puzzle_dataset import (
    PuzzleDataset,
    PuzzleDatasetConfig,
    IGNORE_LABEL_ID,
)
from experiments.attractor_puzzles.trm.losses import softmax_cross_entropy


# ========================================================================
#  Configuration
# ========================================================================


@dataclass
class TrainCliffordConfig:
    # Clifford algebra
    p: int = 3
    q: int = 0
    r: int = 0
    channels: int = 16
    hidden_channels: Optional[int] = None
    num_blocks: int = 3
    num_rotors: int = 4
    use_blade_selector: bool = True
    use_geometric_activation: bool = True
    output_mode: str = "linear"

    # DEQ / Solver
    deq_max_iter: int = 30
    deq_min_iter: int = 5
    deq_tol: float = 1e-4
    deq_anderson_m: int = 5
    deq_anderson_beta: float = 1.0

    # Training
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000
    batch_size: int = 64
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    # Loss
    loss_type: str = "softmax_cross_entropy"
    jacobian_reg_lambda: float = 0.01

    # Data
    dataset_paths: List[str] = field(default_factory=list)
    seed: int = 42

    # Distributed
    rank: int = 0
    num_replicas: int = 1

    # Output
    output_dir: str = "./outputs/clifford-attractor-sudoku"
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 5000


# ========================================================================
#  Jacobian Regularization
# ========================================================================


def jacobian_reg(f, x, vec=None):
    """Power-iteration Jacobian regularization for the fixed-point map.

    Computes ||J_f(x) * v||² where v is a random unit vector, using
    two forward-mode AD-like passes (via torch.autograd.grad).
    """
    if vec is None:
        vec = torch.randn_like(x)
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # f(x) requires grad
    x_in = x.detach().requires_grad_(True)
    fx = f(x_in)

    # J * v = grad(f(x) · v_random, x)
    # First, compute f(x) · v
    fx_dot_v = (fx * vec).sum()
    Jv = torch.autograd.grad(fx_dot_v, x_in, create_graph=True)[0]

    # ||J * v||²
    reg = (Jv ** 2).sum(dim=-1).mean()
    return reg


# ========================================================================
#  Cllifford Puzzle Model Wrapper
# ========================================================================


class CliffordPuzzleModel(nn.Module):
    """Wrapper for CliffordAttractor on puzzle tasks.

    Handles input embedding, the Clifford attractor dynamics, and
    output projection to vocabulary logits.
    """

    def __init__(
        self,
        config: TrainCliffordConfig,
        vocab_size: int = 11,
        seq_len: int = 81,
        num_puzzle_ids: int = 1,
    ):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Build CliffordAttractor
        attractor_cfg = CliffordAttractorConfig(
            p=config.p,
            q=config.q,
            r=config.r,
            channels=config.channels,
            hidden_channels=config.hidden_channels or (config.channels * 2),
            num_blocks=config.num_blocks,
            num_rotors=config.num_rotors,
            use_blade_selector=config.use_blade_selector,
            use_geometric_activation=config.use_geometric_activation,
            max_iter=config.deq_max_iter,
            tol=config.deq_tol,
            anderson_m=config.deq_anderson_m,
            anderson_beta=config.deq_anderson_beta,
            solver="anderson",
            output_mode=config.output_mode,
        )
        self.attractor = CliffordAttractor(attractor_cfg, input_dim=seq_len, output_dim=vocab_size, vocab_size=vocab_size)

        # Puzzle-specific: we use token embeddings for input
        # (Already handled by CliffordAttractor's token_embed)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_solver_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass through the Clifford attractor.

        Args:
            input_ids: [B, seq_len] token indices.
            return_solver_stats: Whether to return solver info.

        Returns:
            logits: [B, seq_len, vocab_size]
            Optional dict with solver stats.
        """
        return self.attractor(input_ids, return_solver_stats=return_solver_stats)


# ========================================================================
#  Training Setup
# ========================================================================


def setup_distributed() -> Tuple[int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def build_model(cfg: TrainCliffordConfig, metadata) -> CliffordPuzzleModel:
    """Build the Clifford puzzle model from config and dataset metadata."""
    model = CliffordPuzzleModel(
        config=cfg,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_ids=metadata.num_puzzle_identifiers,
    )
    return model


def build_optimizer(model: nn.Module, cfg: TrainCliffordConfig):
    """Build AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )


def build_scheduler(optimizer, cfg: TrainCliffordConfig):
    """Build cosine schedule with linear warmup."""
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_dataloader(cfg: TrainCliffordConfig) -> PuzzleDataset:
    """Build puzzle dataset."""
    dataset_cfg = PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.dataset_paths,
        global_batch_size=cfg.batch_size * cfg.num_replicas,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=cfg.rank,
        num_replicas=cfg.num_replicas,
    )
    return PuzzleDataset(dataset_cfg, split="train")


# ========================================================================
#  Training Loop
# ========================================================================


def train(cfg: TrainCliffordConfig):
    """Main training loop."""
    rank, world_size = setup_distributed()
    cfg.rank = rank
    cfg.num_replicas = world_size
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        print(f"Config: {asdict(cfg)}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")

    # Data
    dataset = build_dataloader(cfg)
    # Get metadata from first batch
    for set_name, batch, effective_batch in dataset:
        metadata_keys = ["seq_len", "vocab_size"]
        metadata = type('obj', (object,), {
            "seq_len": batch["inputs"].shape[1],
            "vocab_size": batch["inputs"].max().item() + 2,
            "num_puzzle_identifiers": 1,
        })
        break

    # Model
    model = build_model(cfg, metadata).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Training loop
    step = 0
    best_loss = float("inf")
    log_buffer = []

    while step < cfg.max_steps:
        for set_name, batch, effective_batch in dataset:
            if step >= cfg.max_steps:
                break

            input_ids = batch["inputs"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with Clifford attractor
            model.train()
            logits = model(input_ids, return_solver_stats=False)

            # Compute loss
            mask = labels != IGNORE_LABEL_ID
            loss = softmax_cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_LABEL_ID,
            )
            loss = loss / mask.sum(-1).clamp_min(1).unsqueeze(-1)
            loss = loss.sum()

            # Jacobian regularization (every 5 steps)
            jac_loss = torch.zeros((), device=device)
            if cfg.jacobian_reg_lambda > 0 and step % 5 == 0:
                # Compute Jacobian of attractor fixed-point map
                with torch.enable_grad():
                    emb = model.module.attractor.token_embed(input_ids[:1]) if hasattr(model, 'module') else model.attractor.token_embed(input_ids[:1])
                    x0 = emb.view(1, -1, cfg.channels, 1 << (cfg.p + cfg.q))[:, 0]

                    def f_map(x):
                        x_mv = x.view(1, cfg.channels, 1 << (cfg.p + cfg.q))
                        for block in (model.module.attractor.blocks if hasattr(model, 'module') else model.attractor.blocks):
                            x_mv = block(x_mv.unsqueeze(1)).squeeze(1)
                        return x_mv.view(1, -1)

                    jac_loss = cfg.jacobian_reg_lambda * jacobian_reg(f_map, x0.view(1, -1))

            total_loss = loss + jac_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            if rank == 0:
                log_buffer.append({
                    "step": step,
                    "loss": loss.item(),
                    "jac_loss": jac_loss.item(),
                    "total_loss": total_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                })

                if step % cfg.log_every == 0 and log_buffer:
                    avg_loss = sum(d["loss"] for d in log_buffer[-cfg.log_every:]) / len(log_buffer[-cfg.log_every:])
                    avg_total = sum(d["total_loss"] for d in log_buffer[-cfg.log_every:]) / len(log_buffer[-cfg.log_every:])
                    print(f"Step {step:>6d} | Loss: {avg_loss:.4f} | Total: {avg_total:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

                if step % cfg.eval_every == 0 and step > 0:
                    # Quick accuracy check
                    with torch.no_grad():
                        model.eval()
                        logits_eval = model(input_ids[:16], return_solver_stats=False)
                        preds = logits_eval.argmax(dim=-1)
                        mask_eval = labels[:16] != IGNORE_LABEL_ID
                        acc = (preds == labels[:16])[mask_eval].float().mean().item()
                        print(f"Step {step:>6d} | Accuracy: {acc:.4f}")

                if step % cfg.save_every == 0 and step > 0:
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{step}.pt")
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved checkpoint to {save_path}")

            step += 1

    # Save final model
    if rank == 0:
        final_path = os.path.join(cfg.output_dir, "final.pt")
        torch.save(model.state_dict(), final_path)
        # Save training log
        with open(os.path.join(cfg.output_dir, "train_log.json"), "w") as f:
            json.dump(log_buffer, f)
        print(f"\nTraining complete! Final model saved to {final_path}")

    if world_size > 1:
        dist.destroy_process_group()


# ========================================================================
#  CLI
# ========================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Clifford Attractor on puzzle tasks")
    # Clifford params
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--q", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--num-rotors", type=int, default=4)
    parser.add_argument("--no-blade-selector", dest="use_blade_selector", action="store_false")
    parser.add_argument("--no-geometric-act", dest="use_geometric_activation", action="store_false")

    # Solver params
    parser.add_argument("--deq-max-iter", type=int, default=30)
    parser.add_argument("--deq-tol", type=float, default=1e-4)
    parser.add_argument("--deq-anderson-m", type=int, default=5)

    # Training params
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--jacobian-reg-lambda", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--dataset-paths", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/clifford-attractor-sudoku")

    args = parser.parse_args()

    cfg = TrainCliffordConfig(
        p=args.p,
        q=args.q,
        channels=args.channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_rotors=args.num_rotors,
        use_blade_selector=args.use_blade_selector,
        use_geometric_activation=args.use_geometric_activation,
        deq_max_iter=args.deq_max_iter,
        deq_tol=args.deq_tol,
        deq_anderson_m=args.deq_anderson_m,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        jacobian_reg_lambda=args.jacobian_reg_lambda,
        dataset_paths=args.dataset_paths,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    train(cfg)


if __name__ == "__main__":
    main()
