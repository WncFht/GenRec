#!/usr/bin/env python3
"""Prepare <category>.train/valid/test.inter from <category>.inter.json and run preprocess_data_sft_rl.py."""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Optional, Tuple, Union


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_category(category_dir: str, category: Optional[str]) -> str:
    if category:
        return category
    return os.path.basename(os.path.normpath(category_dir))


def find_index_candidates(category_dir: str, category: str) -> list[str]:
    patterns = [
        f"{category}.index.json",
        f"{category}.index_*.json",
        f"{category}.index*.json",
        "*.index*.json",
    ]
    found = set()
    for p in patterns:
        for path in glob.glob(os.path.join(category_dir, p)):
            if os.path.isfile(path):
                found.add(os.path.abspath(path))
    return sorted(found)


def resolve_index_path(category_dir: str, category: str, index_path: Optional[str]) -> str:
    if index_path:
        resolved = os.path.abspath(index_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Index file not found: {resolved}")
        return resolved

    exact = os.path.join(category_dir, f"{category}.index.json")
    if os.path.isfile(exact):
        return os.path.abspath(exact)

    candidates = find_index_candidates(category_dir, category)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple index candidates found. Please set --index-path explicitly:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )
    raise FileNotFoundError(f"No index candidates found under {category_dir}")


def user_sort_key(uid: str) -> Tuple[int, Union[int, str]]:
    """Sort numeric user IDs numerically, fallback to string order."""
    u = str(uid)
    if u.isdigit():
        return (0, int(u))
    return (1, u)


def normalize_seq(seq: Any) -> list[str]:
    if not isinstance(seq, list):
        return []
    return [str(x).strip() for x in seq if str(x).strip()]


def build_rows_mimionerec(inter_data: dict[str, Any], history_max: int) -> list[tuple[str, list[str], str]]:
    """MIMIOneRec-like: build all next-item rows, then global ratio split (8:1:1 by default)."""
    rows: list[tuple[str, list[str], str]] = []
    for user_id in sorted(inter_data.keys(), key=user_sort_key):
        seq_ids = normalize_seq(inter_data[user_id])
        if len(seq_ids) < 2:
            continue
        for i in range(1, len(seq_ids)):
            st = max(0, i - history_max)
            history = seq_ids[st:i]
            if not history:
                continue
            target = seq_ids[i]
            rows.append((str(user_id), history, target))
    return rows


def split_rows_by_ratio(rows: list[tuple[str, list[str], str]], train_ratio: float, valid_ratio: float) -> tuple[list, list, list]:
    total = len(rows)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))
    train_rows = rows[:train_end]
    valid_rows = rows[train_end:valid_end]
    test_rows = rows[valid_end:]
    return train_rows, valid_rows, test_rows


def build_rows_grec(inter_data: dict[str, Any], history_max: int) -> tuple[list, list, list]:
    """GRec-like split:
    - train: per-user training prefix (exclude last2) and generate next-item pairs
    - valid: one row per user (target is last-2, history is train prefix)
    - test : one row per user (target is last-1, history is train+valid prefix)
    """
    train_rows: list[tuple[str, list[str], str]] = []
    valid_rows: list[tuple[str, list[str], str]] = []
    test_rows: list[tuple[str, list[str], str]] = []

    for user_id in sorted(inter_data.keys(), key=user_sort_key):
        seq_ids = normalize_seq(inter_data[user_id])
        if len(seq_ids) < 2:
            continue

        # Match GRec data_process/amazon18_data_process.py:
        # train_data = seq[:-2], valid target=seq[-2], test target=seq[-1].
        train_seq = seq_ids[:-2]

        # Generate train rows in reverse target order, same as GRec's convert_to_atomic_files.
        for target_idx in range(1, len(train_seq)):
            target_item = train_seq[-target_idx]
            history = train_seq[:-target_idx][-history_max:]
            train_rows.append((str(user_id), history, target_item))

        valid_history = seq_ids[:-2][-history_max:]
        valid_target = seq_ids[-2]
        valid_rows.append((str(user_id), valid_history, valid_target))

        test_history = seq_ids[:-1][-history_max:]
        test_target = seq_ids[-1]
        test_rows.append((str(user_id), test_history, test_target))

    return train_rows, valid_rows, test_rows


def write_inter(path: str, rows: list[tuple[str, list[str], str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for user_id, history, target in rows:
            f.write(f"{user_id}\t{' '.join(history)}\t{target}\n")


def run(cmd: list[str], dry_run: bool) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build .train/.valid/.test.inter from .inter.json, then run preprocess_data_sft_rl.py."
    )
    parser.add_argument("--genrec-root", default=".", help="Path to GenRec repo root.")
    parser.add_argument("--category-dir", required=True, help="Path to raw category dir, e.g. /path/to/data/Instruments")
    parser.add_argument("--category", default=None, help="Category name. Defaults to basename(category-dir).")
    parser.add_argument("--index-path", default=None, help="Path to selected index json.")
    parser.add_argument("--output-dir", default=None, help="Where SFT/RL outputs are written. Default: <genrec-root>/data/<category>")
    parser.add_argument(
        "--staging-root",
        default=None,
        help="Temp input root for preprocess_data_sft_rl.py. Default: <genrec-root>/data/_preprocess_input",
    )
    parser.add_argument("--history-max", type=int, default=50, help="Keep last N history item IDs in .inter rows.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Valid split ratio.")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="mimionerec",
        choices=["mimionerec", "grec"],
        help="How to build train/valid/test rows from <category>.inter.json.",
    )
    parser.add_argument("--seq-sample", type=int, default=10000, help="Forwarded to preprocess_data_sft_rl.py --seq_sample")
    parser.add_argument("--seed", type=int, default=42, help="Forwarded to preprocess_data_sft_rl.py --seed")
    parser.add_argument("--sid-levels", type=int, default=-1, help="Forwarded to preprocess_data_sft_rl.py --sid_levels. <=0 means auto(all levels).")
    parser.add_argument("--data-source", default=None, help="Forwarded to preprocess_data_sft_rl.py --data_source")
    parser.add_argument("--python-bin", default=sys.executable or "python3", help="Python executable to run preprocess_data_sft_rl.py")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare staging files, skip preprocess_data_sft_rl.py")
    parser.add_argument("--dry-run", action="store_true", help="Print command only and skip executing preprocess.")
    args = parser.parse_args()

    genrec_root = os.path.abspath(args.genrec_root)
    category_dir = os.path.abspath(args.category_dir)
    category = resolve_category(category_dir, args.category)
    output_dir = os.path.abspath(args.output_dir or os.path.join(genrec_root, "data", category))
    staging_root = os.path.abspath(args.staging_root or os.path.join(genrec_root, "data", "_preprocess_input"))
    staging_category_dir = os.path.join(staging_root, category)

    if args.split_strategy == "mimionerec":
        if not (0.0 < args.train_ratio < 1.0 and 0.0 <= args.valid_ratio < 1.0 and args.train_ratio + args.valid_ratio < 1.0):
            raise ValueError(
                "Invalid split ratios. Require 0 < train_ratio < 1, 0 <= valid_ratio < 1, train_ratio + valid_ratio < 1."
            )

    item_src = os.path.join(category_dir, f"{category}.item.json")
    inter_src = os.path.join(category_dir, f"{category}.inter.json")
    if not os.path.isfile(item_src):
        raise FileNotFoundError(f"Missing item file: {item_src}")
    if not os.path.isfile(inter_src):
        raise FileNotFoundError(f"Missing inter file: {inter_src}")

    index_src = resolve_index_path(category_dir, category, args.index_path)

    inter_data = read_json(inter_src)
    if not isinstance(inter_data, dict):
        raise ValueError(f"Expected dict in inter json, got {type(inter_data)}")

    if args.split_strategy == "mimionerec":
        rows = build_rows_mimionerec(inter_data, args.history_max)
        if not rows:
            raise ValueError("No usable interaction rows generated from inter json.")
        train_rows, valid_rows, test_rows = split_rows_by_ratio(rows, args.train_ratio, args.valid_ratio)
    else:
        train_rows, valid_rows, test_rows = build_rows_grec(inter_data, args.history_max)
        if not (train_rows or valid_rows or test_rows):
            raise ValueError("No usable interaction rows generated from inter json.")

    os.makedirs(staging_category_dir, exist_ok=True)
    shutil.copy2(item_src, os.path.join(staging_category_dir, f"{category}.item.json"))
    shutil.copy2(index_src, os.path.join(staging_category_dir, f"{category}.index.json"))
    write_inter(os.path.join(staging_category_dir, f"{category}.train.inter"), train_rows)
    write_inter(os.path.join(staging_category_dir, f"{category}.valid.inter"), valid_rows)
    write_inter(os.path.join(staging_category_dir, f"{category}.test.inter"), test_rows)

    print(f"[INFO] category={category}")
    print(f"[INFO] item_src={item_src}")
    print(f"[INFO] inter_src={inter_src}")
    print(f"[INFO] index_src={index_src}")
    print(f"[INFO] split_strategy={args.split_strategy}")
    if args.split_strategy == "mimionerec":
        print(f"[INFO] split_ratio train/valid/test={args.train_ratio}/{args.valid_ratio}/{1.0 - args.train_ratio - args.valid_ratio}")
    print(f"[INFO] sid_levels={args.sid_levels}")
    print(f"[INFO] staging_category_dir={staging_category_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] generated rows: train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}")

    if args.prepare_only:
        print("[INFO] prepare-only mode, skip preprocess_data_sft_rl.py")
        return

    preprocess_script = os.path.join(genrec_root, "preprocess_data_sft_rl.py")
    if not os.path.isfile(preprocess_script):
        raise FileNotFoundError(f"Cannot find preprocess_data_sft_rl.py under genrec-root: {preprocess_script}")

    data_source = args.data_source or category
    cmd = [
        args.python_bin,
        preprocess_script,
        "--data_dir",
        staging_root,
        "--category",
        category,
        "--output_dir",
        output_dir,
        "--seq_sample",
        str(args.seq_sample),
        "--seed",
        str(args.seed),
        "--sid_levels",
        str(args.sid_levels),
        "--data_source",
        data_source,
    ]
    run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
