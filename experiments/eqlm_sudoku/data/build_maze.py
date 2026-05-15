"""Build the Maze-Hard 30x30 dataset (ported from TinyRecursiveModels).

Vocab: PAD=0, '#'=1, ' '=2, 'S'=3, 'G'=4, 'o'=5
Grid: 30x30, seq_len=900.

Usage:
    python -m experiments.eqlm_sudoku.data.build_maze \\
        --output-dir /scratch1/feinashl/data/maze-30x30-hard-1k
"""
import argparse
import csv
import json
import math
import os

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .common import PuzzleDatasetMetadata, dihedral_transform

CHARSET = "# SGo"


def convert_subset(set_name: str, args):
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []

    csv_path = hf_hub_download(args.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for source, q, a, rating in reader:
            all_chars.update(q)
            all_chars.update(a)
            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    if set_name == "train" and args.subsample_size is not None:
        total = len(inputs)
        if args.subsample_size < total:
            idxs = np.random.choice(total, size=args.subsample_size, replace=False)
            inputs = [inputs[i] for i in idxs]
            labels = [labels[i] for i in idxs]

    out = {k: [] for k in ["inputs", "labels", "puzzle_identifiers",
                            "puzzle_indices", "group_indices"]}
    out["puzzle_indices"].append(0)
    out["group_indices"].append(0)
    pid = 0

    do_aug = set_name == "train" and args.aug
    for inp, lab in zip(tqdm(inputs, desc=f"build {set_name}"), labels):
        for aug_idx in range(8 if do_aug else 1):
            out["inputs"].append(dihedral_transform(inp, aug_idx))
            out["labels"].append(dihedral_transform(lab, aug_idx))
            pid += 1
            out["puzzle_indices"].append(pid)
            out["puzzle_identifiers"].append(0)
        out["group_indices"].append(pid)

    assert len(all_chars - set(CHARSET)) == 0
    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    def _to_np(seq):
        return np.vstack([char2id[s.reshape(-1)] for s in seq])

    results = {
        "inputs": _to_np(out["inputs"]),
        "labels": _to_np(out["labels"]),
        "group_indices": np.array(out["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(out["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(out["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),
        vocab_size=len(CHARSET) + 1,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    save_dir = os.path.join(args.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/scratch1/feinashl/data/maze-30x30-hard-1k")
    p.add_argument("--source-repo", default="sapientinc/maze-30x30-hard-1k")
    p.add_argument("--subsample-size", type=int, default=None)
    p.add_argument("--aug", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    convert_subset("train", args)
    convert_subset("test", args)


if __name__ == "__main__":
    main()
