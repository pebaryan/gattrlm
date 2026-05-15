"""Build the Sudoku-Extreme dataset (ported from TinyRecursiveModels).

Outputs the same numpy/json layout that puzzle_dataset.PuzzleDataset expects:

    <output_dir>/
        identifiers.json       (just ["<blank>"])
        train/
            dataset.json
            all__inputs.npy            (uint8: 1+pad_id .. 10, shape (N, 81))
            all__labels.npy            (uint8)
            all__puzzle_indices.npy    (int32, shape (N+1,))
            all__group_indices.npy     (int32, shape (G+1,))
            all__puzzle_identifiers.npy(int32, shape (N,))
        test/
            (same)

Vocab convention (matches TRM):
    0  -> PAD / ignore_label_id
    1  -> token "0" (blank cell)
    2..10 -> tokens "1".."9"
"""
import argparse
import csv
import json
import os
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .common import PuzzleDatasetMetadata


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[new_board]

    return apply(board), apply(solution)


def convert_subset(set_name: str, args, output_dir: str):
    inputs = []
    labels = []
    csv_path = hf_hub_download(args.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for source, q, a, rating in reader:
            if (args.min_difficulty is None) or (int(rating) >= args.min_difficulty):
                assert len(q) == 81 and len(a) == 81
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    if set_name == "train" and args.subsample_size is not None:
        total = len(inputs)
        if args.subsample_size < total:
            idxs = np.random.choice(total, size=args.subsample_size, replace=False)
            inputs = [inputs[i] for i in idxs]
            labels = [labels[i] for i in idxs]

    num_aug = args.num_aug if set_name == "train" else 0

    out = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    out["puzzle_indices"].append(0)
    out["group_indices"].append(0)
    pid = 0
    for orig_inp, orig_out in zip(tqdm(inputs, desc=f"build {set_name}"), labels):
        for aug_idx in range(1 + num_aug):
            if aug_idx == 0:
                inp, lab = orig_inp, orig_out
            else:
                inp, lab = shuffle_sudoku(orig_inp, orig_out)
            out["inputs"].append(inp)
            out["labels"].append(lab)
            pid += 1
            out["puzzle_indices"].append(pid)
            out["puzzle_identifiers"].append(0)
        out["group_indices"].append(pid)

    def _to_np(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    results = {
        "inputs": _to_np(out["inputs"]),
        "labels": _to_np(out["labels"]),
        "group_indices": np.array(out["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(out["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(out["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def main():
    p = argparse.ArgumentParser(description="Build Sudoku-Extreme dataset for EQLM/TRM training.")
    p.add_argument("--output-dir", default="/scratch1/feinashl/data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--source-repo", default="sapientinc/sudoku-extreme")
    p.add_argument("--subsample-size", type=int, default=1000)
    p.add_argument("--min-difficulty", type=int, default=None)
    p.add_argument("--num-aug", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    convert_subset("train", args, args.output_dir)
    convert_subset("test", args, args.output_dir)


if __name__ == "__main__":
    main()
