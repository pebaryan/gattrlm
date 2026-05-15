"""Iterable puzzle dataset (ported from TinyRecursiveModels with minor cleanups).

Each batch yields a dict with int32 numpy → torch tensors:
    inputs              : (B, seq_len)
    labels              : (B, seq_len)  with metadata.ignore_label_id remapped to -100
    puzzle_identifiers  : (B,)

Plus the set name (e.g. "all") and the pre-pad effective batch size.
"""
import json
import os
from typing import List, Optional

import numpy as np
import pydantic
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .common import PuzzleDatasetMetadata


IGNORE_LABEL_ID = -100


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
):
    batch = []
    batch_puzzle_indices = []
    cur = 0
    while start_index < group_order.size and cur < global_batch_size:
        gid = group_order[start_index]
        pid = rng.integers(group_indices[gid], group_indices[gid + 1])
        start_index += 1

        p_start = puzzle_indices[pid]
        p_size = int(puzzle_indices[pid + 1] - p_start)
        take = min(p_size, global_batch_size - cur)

        batch_puzzle_indices.append(np.full(take, pid, dtype=np.int32))
        batch.append(p_start + np.random.choice(p_size, take, replace=False))
        cur += take
    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int
    num_replicas: int


class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        prev = {}
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for path in config.dataset_paths:
            md = self._load_metadata(path)
            for k in ("seq_len", "vocab_size", "pad_id", "ignore_label_id",
                     "blank_identifier_id", "sets", "num_puzzle_identifiers"):
                v = getattr(md, k)
                if k not in prev:
                    prev[k] = v
                else:
                    assert prev[k] == v, f"metadata mismatch on {k}: {prev[k]} vs {v}"
            mean_puzzle_examples += md.mean_puzzle_examples * md.total_puzzles
            total_puzzles += md.total_puzzles
            total_groups += md.total_groups
            num_identifiers += md.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / max(total_puzzles, 1)

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev["seq_len"],
            vocab_size=prev["vocab_size"],
            pad_id=prev["pad_id"],
            ignore_label_id=prev["ignore_label_id"],
            blank_identifier_id=prev["blank_identifier_id"],
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev["sets"],
        )

        assert config.global_batch_size % config.num_replicas == 0, (
            f"global_batch_size {config.global_batch_size} must be divisible by num_replicas {config.num_replicas}")
        self.local_batch_size = config.global_batch_size // config.num_replicas
        self._data: Optional[dict] = None
        self._iters = 0

    def _load_metadata(self, dataset_path: str) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return
        mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }
        self._data = {}
        for set_name in self.metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                key = set_name if i == 0 else f"{set_name}{i}"
                self._data[key] = {
                    field: np.load(
                        os.path.join(dataset_path, self.split, f"{set_name}__{field}.npy"),
                        mmap_mode=mmap_mode,
                    )
                    for field, mmap_mode in mmap_modes.items()
                }

    def _collate_batch(self, batch):
        batch = {k: v.astype(np.int32) for k, v in batch.items()}
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            batch = {
                k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_values[k])
                for k, v in batch.items()
            }
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        for set_name, dataset in self._data.items():
            total = len(dataset["inputs"])
            start = 0
            while start < total:
                end = min(total, start + self.config.global_batch_size)
                local_start = start + self.config.rank * self.local_batch_size
                local_end = min(start + (self.config.rank + 1) * self.local_batch_size, end)

                puzzle_idx = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                puzzle_indices = []
                for i in range(local_start, local_end):
                    while puzzle_idx + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_idx + 1]:
                        puzzle_idx += 1
                    puzzle_indices.append(puzzle_idx)

                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start:local_end],
                    "labels": dataset["labels"][local_start:local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices],
                })
                yield set_name, batch, end - start
                start += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            group_order = np.concatenate(
                [rng.permutation(dataset["group_indices"].size - 1) for _ in range(self.config.epochs_per_iter)]
            )
            start = 0
            while start < group_order.size:
                start, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start,
                    global_batch_size=self.config.global_batch_size,
                )
                global_effective = batch_puzzle_indices.size
                if global_effective < self.config.global_batch_size:
                    break

                lo = self.config.rank * self.local_batch_size
                hi = (self.config.rank + 1) * self.local_batch_size
                batch_indices = batch_indices[lo:hi]
                batch_puzzle_indices = batch_puzzle_indices[lo:hi]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices],
                })
                yield set_name, batch, global_effective

    def __iter__(self):
        info = get_worker_info()
        assert info is None or info.num_workers == 1, "Multi-worker dataloading is not supported."
        self._lazy_load_dataset()
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
