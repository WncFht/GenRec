#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SID_TOKEN_PATTERN = re.compile(r"<[^<>]+>")
TASK_FOCUS = ["sid", "hisTitle2sid"]
TRANSITIONS = [
    {
        "name": "base_to_need_hint",
        "frame_filter": lambda frame: frame.index == frame.index,
        "label": lambda frame: frame["effective_hint_depth"] >= 1,
        "depth": 1,
        "group_parent_col": None,
        "group_token_col": "token_d1",
    },
    {
        "name": "hint1_to_need_hint2_plus",
        "frame_filter": lambda frame: frame["effective_hint_depth"] >= 1,
        "label": lambda frame: frame["effective_hint_depth"] >= 2,
        "depth": 2,
        "group_parent_col": "parent_d2",
        "group_token_col": "token_d2",
    },
    {
        "name": "hint2_to_need_hint3_plus",
        "frame_filter": lambda frame: frame["effective_hint_depth"] >= 2,
        "label": lambda frame: frame["effective_hint_depth"] >= 3,
        "depth": 3,
        "group_parent_col": "parent_d3",
        "group_token_col": "token_d3",
    },
    {
        "name": "hint3_to_unsolved",
        "frame_filter": lambda frame: frame["effective_hint_depth"] >= 3,
        "label": lambda frame: frame["final_unsolved"].astype(bool),
        "depth": 4,
        "group_parent_col": "parent_d4",
        "group_token_col": "token_d4",
    },
]
FEATURE_BASES = [
    "global_count",
    "task_parent_share",
    "subtree",
    "child_share",
    "child_rank",
    "sibling",
    "parent_entropy",
    "parent_maxshare",
]
SEQUENCE_FEATURES = [
    "seq_history_len",
    "seq_target_pos_ratio",
    "seq_train_seq_len",
    "seq_target_from_end",
]
TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TITLE_STOPWORDS = {
    "walker",
    "williams",
    "guitar",
    "strap",
    "straps",
    "with",
    "and",
    "the",
    "for",
    "black",
    "brown",
    "leather",
    "padded",
    "glove",
    "back",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore a local GenRec hint research bundle.")
    parser.add_argument(
        "--bundle-root",
        default=(
            "/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle"
        ),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def prefix_key(tokens: list[str]) -> str:
    return "".join(tokens) if tokens else "ROOT"


def normalized_entropy(shares: list[float]) -> float:
    shares = [share for share in shares if share > 0]
    if len(shares) <= 1:
        return 0.0
    entropy = -sum(share * math.log(share) for share in shares)
    return entropy / math.log(len(shares))


def find_bundle_file(bundle_root: Path, relative_path: str) -> Path:
    candidate = bundle_root / relative_path
    if not candidate.exists():
        raise FileNotFoundError(f"Missing bundle file: {candidate}")
    return candidate


def best_threshold(frame: pd.DataFrame, feature: str, label_col: str) -> dict[str, Any]:
    subset = frame[[feature, label_col]].dropna().sort_values(feature)
    if subset.empty:
        return {
            "feature": feature,
            "balanced_acc": np.nan,
            "acc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "threshold": np.nan,
            "direction": "",
        }

    x = subset[feature].to_numpy(dtype=float)
    y = subset[label_col].to_numpy(dtype=bool)
    unique_values = np.unique(x)
    if len(unique_values) <= 200:
        candidates = unique_values
    else:
        candidates = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 199)))

    best: tuple[float, float, float, float, float, str] | None = None
    for threshold in candidates:
        for direction in ("<=", ">="):
            pred = x <= threshold if direction == "<=" else x >= threshold
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tpr = tp / max(tp + fn, 1)
            tnr = tn / max(tn + fp, 1)
            balanced_acc = (tpr + tnr) / 2.0
            acc = (tp + tn) / len(y)
            precision = tp / max(tp + fp, 1)
            recall = tpr
            candidate = (balanced_acc, acc, precision, recall, float(threshold), direction)
            if best is None or candidate[0] > best[0]:
                best = candidate

    assert best is not None
    return {
        "feature": feature,
        "balanced_acc": best[0],
        "acc": best[1],
        "precision": best[2],
        "recall": best[3],
        "threshold": best[4],
        "direction": best[5],
    }


def attach_per_depth_metrics(df: pd.DataFrame, id2sid: dict[str, list[str]]) -> pd.DataFrame:
    frame = df.copy()
    frame["sid_tokens"] = frame["ground_truth"].map(lambda text: SID_TOKEN_PATTERN.findall(text))

    child_counts_by_parent: dict[tuple[int, tuple[str, ...]], Counter[str]] = defaultdict(Counter)
    for raw_tokens in id2sid.values():
        tokens = [str(token) for token in raw_tokens]
        for depth, token in enumerate(tokens, start=1):
            parent = tuple(tokens[: depth - 1])
            child_counts_by_parent[(depth, parent)][token] += 1

    tree_lookup: dict[tuple[int, str, str], dict[str, float]] = {}
    for (depth, parent), child_counter in child_counts_by_parent.items():
        ordered_children = sorted(child_counter.items(), key=lambda item: (-item[1], item[0]))
        total = sum(child_counter.values())
        shares = [count / total for _, count in ordered_children]
        entropy = normalized_entropy(shares)
        max_share = max(shares) if shares else 0.0
        for rank, (token, count) in enumerate(ordered_children, start=1):
            tree_lookup[(depth, prefix_key(list(parent)), token)] = {
                "subtree": float(count),
                "child_share": float(count) / float(total),
                "child_rank": float(rank),
                "sibling": float(len(ordered_children)),
                "parent_entropy": entropy,
                "parent_maxshare": max_share,
            }

    task_parent_counts: Counter[tuple[str, int, str, str]] = Counter()
    task_parent_totals: Counter[tuple[str, int, str]] = Counter()
    train_global_counts: Counter[tuple[int, str]] = Counter()
    for task, tokens in zip(frame["task"], frame["sid_tokens"]):
        for depth, token in enumerate(tokens, start=1):
            parent = prefix_key(tokens[: depth - 1])
            train_global_counts[(depth, token)] += 1
            task_parent_counts[(task, depth, parent, token)] += 1
            task_parent_totals[(task, depth, parent)] += 1

    max_depth = max(len(tokens) for tokens in frame["sid_tokens"])
    for depth in range(1, max_depth + 1):
        columns: dict[str, list[Any]] = {
            "token": [],
            "parent": [],
            "global_count": [],
            "task_parent_share": [],
            "subtree": [],
            "child_share": [],
            "child_rank": [],
            "sibling": [],
            "parent_entropy": [],
            "parent_maxshare": [],
        }

        for task, tokens in zip(frame["task"], frame["sid_tokens"]):
            token = tokens[depth - 1] if len(tokens) >= depth else None
            parent = prefix_key(tokens[: depth - 1]) if token is not None else None
            columns["token"].append(token)
            columns["parent"].append(parent)
            columns["global_count"].append(
                train_global_counts.get((depth, token), np.nan) if token is not None else np.nan
            )

            denom = task_parent_totals.get((task, depth, parent), 0)
            if token is None or denom == 0:
                columns["task_parent_share"].append(np.nan)
            else:
                columns["task_parent_share"].append(
                    task_parent_counts.get((task, depth, parent, token), np.nan) / float(denom)
                )

            tree_metrics = tree_lookup.get((depth, parent, token), {}) if token is not None else {}
            for key in ("subtree", "child_share", "child_rank", "sibling", "parent_entropy", "parent_maxshare"):
                columns[key].append(tree_metrics.get(key, np.nan))

        for key, values in columns.items():
            frame[f"{key}_d{depth}"] = values

    return frame


def build_threshold_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS:
        transition_frame = frame[transition["frame_filter"](frame)].copy()
        transition_frame["label"] = transition["label"](transition_frame)
        depth = transition["depth"]

        for feature_base in FEATURE_BASES:
            feature = f"{feature_base}_d{depth}"
            rows.append(
                {
                    "transition": transition["name"],
                    "task_label": "all",
                    "sample_count": int(len(transition_frame)),
                    "positive_rate": float(transition_frame["label"].mean()),
                    **best_threshold(transition_frame, feature, "label"),
                }
            )

        for task_label in ("sid", "hisTitle2sid", "title_desc2sid"):
            task_frame = transition_frame[transition_frame["task_label"] == task_label].copy()
            if task_frame.empty:
                continue
            for feature_base in FEATURE_BASES:
                feature = f"{feature_base}_d{depth}"
                rows.append(
                    {
                        "transition": transition["name"],
                        "task_label": task_label,
                        "sample_count": int(len(task_frame)),
                        "positive_rate": float(task_frame["label"].mean()),
                        **best_threshold(task_frame, feature, "label"),
                    }
                )

    for transition in TRANSITIONS[:2]:
        transition_frame = frame[transition["frame_filter"](frame)].copy()
        transition_frame["label"] = transition["label"](transition_frame)
        transition_frame = transition_frame[transition_frame["task_label"].isin(TASK_FOCUS)].copy()
        depth = transition["depth"]
        for feature in SEQUENCE_FEATURES:
            rows.append(
                {
                    "transition": transition["name"],
                    "task_label": "sid+hisTitle2sid",
                    "sample_count": int(len(transition_frame)),
                    "positive_rate": float(transition_frame["label"].mean()),
                    **best_threshold(transition_frame, feature, "label"),
                }
            )
    return rows


def build_group_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS:
        transition_frame = frame[transition["frame_filter"](frame)].copy()
        transition_frame["label"] = transition["label"](transition_frame)
        parent_col = transition["group_parent_col"]
        token_col = transition["group_token_col"]
        group_columns = [token_col] if parent_col is None else [parent_col, token_col]

        grouped = (
            transition_frame.groupby(group_columns)
            .agg(sample_count=("sample_id", "size"), positive_count=("label", "sum"), positive_rate=("label", "mean"))
            .reset_index()
        )
        grouped["expected_rate"] = float(transition_frame["label"].mean())
        grouped["lift"] = grouped["positive_rate"] / grouped["expected_rate"].replace(0, np.nan)

        if transition["name"] == "base_to_need_hint":
            min_size = 100
        elif transition["name"] == "hint3_to_unsolved":
            min_size = 1
        else:
            min_size = 20
        grouped = grouped[grouped["sample_count"] >= min_size].copy()
        grouped = grouped.sort_values(["lift", "positive_count", "sample_count"], ascending=[False, False, False])
        grouped = grouped.head(50)

        for record in grouped.to_dict("records"):
            record["transition"] = transition["name"]
            rows.append(record)
    return rows


def build_task_gap_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS[:3]:
        transition_frame = frame[transition["frame_filter"](frame)].copy()
        transition_frame["label"] = transition["label"](transition_frame)
        transition_frame = transition_frame[transition_frame["task_label"].isin(TASK_FOCUS)].copy()
        depth = transition["depth"]
        transition_frame["share_bin"] = pd.qcut(
            transition_frame[f"task_parent_share_d{depth}"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
        transition_frame["subtree_bin"] = pd.qcut(
            transition_frame[f"subtree_d{depth}"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )

        grouped = (
            transition_frame.groupby(["share_bin", "subtree_bin", "task_label"])
            .agg(sample_count=("sample_id", "size"), positive_rate=("label", "mean"))
            .reset_index()
        )
        pivot_count = grouped.pivot_table(
            index=["share_bin", "subtree_bin"], columns="task_label", values="sample_count", fill_value=0
        )
        pivot_rate = grouped.pivot_table(
            index=["share_bin", "subtree_bin"], columns="task_label", values="positive_rate"
        )
        common_bins = pivot_count[(pivot_count["sid"] > 0) & (pivot_count["hisTitle2sid"] > 0)].index
        weights = pivot_count.loc[common_bins, "sid"] + pivot_count.loc[common_bins, "hisTitle2sid"]
        weights = weights / weights.sum()

        raw_rates = transition_frame.groupby("task_label")["label"].mean().to_dict()
        rows.append(
            {
                "transition": transition["name"],
                "raw_sid_rate": float(raw_rates["sid"]),
                "raw_hisTitle2sid_rate": float(raw_rates["hisTitle2sid"]),
                "standardized_sid_rate": float((pivot_rate.loc[common_bins, "sid"] * weights).sum()),
                "standardized_hisTitle2sid_rate": float((pivot_rate.loc[common_bins, "hisTitle2sid"] * weights).sum()),
                "common_bin_count": int(len(common_bins)),
            }
        )
    return rows


def build_residual_frame(frame: pd.DataFrame) -> pd.DataFrame:
    residual = frame[frame["final_unsolved"].astype(bool)].copy()
    keep_columns = [
        "sample_id",
        "source_index",
        "task_label",
        "ground_truth",
        "token_d1",
        "token_d2",
        "token_d3",
        "token_d4",
        "parent_d3",
        "parent_d4",
        "global_count_d3",
        "global_count_d4",
        "subtree_d3",
        "sibling_d3",
        "parent_entropy_d3",
        "task_parent_share_d4",
        "child_share_d4",
        "sibling_d4",
        "parent_maxshare_d4",
        "seq_history_len",
        "seq_target_pos_ratio",
    ]
    return residual[keep_columns].sort_values(["task_label", "ground_truth", "source_index"]).reset_index(drop=True)


def export_dominant_residual_parent_items(
    *,
    bundle_root: Path,
    id2sid: dict[str, list[str]],
    residual: pd.DataFrame,
    output_dir: Path,
) -> str | None:
    if residual.empty or "parent_d4" not in residual.columns:
        return None

    dominant_parent = residual["parent_d4"].value_counts().index[0]
    if not dominant_parent:
        return None

    item_path = bundle_root / "raw_data/Instruments/Instruments.item.json"
    if not item_path.exists():
        return None
    items = json.loads(item_path.read_text(encoding="utf-8"))
    target_tokens = SID_TOKEN_PATTERN.findall(dominant_parent)

    rows = []
    for item_id, tokens in id2sid.items():
        normalized_tokens = [str(token) for token in tokens]
        if normalized_tokens[: len(target_tokens)] != target_tokens:
            continue
        leaf_token = normalized_tokens[len(target_tokens)] if len(normalized_tokens) > len(target_tokens) else ""
        item = items.get(str(item_id), {})
        rows.append(
            {
                "item_id": str(item_id),
                "sid": "".join(normalized_tokens),
                "leaf_token": leaf_token,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
            }
        )

    if not rows:
        return None

    branch_frame = pd.DataFrame(rows).sort_values(["sid", "item_id"]).reset_index(drop=True)
    out_path = output_dir / "dominant_residual_parent_items.csv"
    branch_frame.to_csv(out_path, index=False)
    return str(out_path)


def build_shared_hotspot_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS:
        transition_frame = frame[transition["frame_filter"](frame)].copy()
        transition_frame["label"] = transition["label"](transition_frame)
        transition_frame = transition_frame[transition_frame["task_label"].isin(TASK_FOCUS)].copy()
        if transition_frame.empty:
            continue

        group_columns = [transition["group_token_col"]]
        if transition["group_parent_col"] is not None:
            group_columns = [transition["group_parent_col"], transition["group_token_col"]]

        grouped = (
            transition_frame.groupby(group_columns + ["task_label"])
            .agg(sample_count=("sample_id", "size"), positive_count=("label", "sum"), positive_rate=("label", "mean"))
            .reset_index()
        )
        baseline_rates = transition_frame.groupby("task_label")["label"].mean().to_dict()
        grouped["lift"] = grouped.apply(
            lambda row: (
                row["positive_rate"] / baseline_rates[row["task_label"]]
                if baseline_rates[row["task_label"]] > 0
                else np.nan
            ),
            axis=1,
        )

        pivot = grouped.pivot_table(
            index=group_columns,
            columns="task_label",
            values=["sample_count", "positive_count", "positive_rate", "lift"],
        )
        pivot.columns = [f"{metric}_{task}" for metric, task in pivot.columns]
        pivot = pivot.reset_index()

        for task in TASK_FOCUS:
            for metric in ("sample_count", "positive_count", "positive_rate", "lift"):
                column = f"{metric}_{task}"
                if column not in pivot.columns:
                    pivot[column] = np.nan

        pivot["shared_lift_min"] = pivot[[f"lift_{task}" for task in TASK_FOCUS]].min(axis=1)
        pivot["shared_lift_geo"] = np.sqrt(pivot["lift_sid"] * pivot["lift_hisTitle2sid"])
        pivot["rate_gap_his_minus_sid"] = pivot["positive_rate_hisTitle2sid"] - pivot["positive_rate_sid"]
        pivot["total_sample_count"] = pivot[[f"sample_count_{task}" for task in TASK_FOCUS]].sum(axis=1)
        pivot["transition"] = transition["name"]

        min_count = 100 if transition["name"] == "base_to_need_hint" else 20
        pivot = pivot[
            (pivot["sample_count_sid"] >= min_count) & (pivot["sample_count_hisTitle2sid"] >= min_count)
        ].copy()
        pivot = pivot.sort_values(
            ["shared_lift_min", "shared_lift_geo", "total_sample_count"],
            ascending=[False, False, False],
        )

        rows.extend(pivot.head(100).to_dict("records"))
    return rows


def build_transition_task_gap_group_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in TRANSITIONS[:3]:
        transition_frame = frame[transition["frame_filter"](frame)].copy()
        transition_frame["label"] = transition["label"](transition_frame)
        transition_frame = transition_frame[transition_frame["task_label"].isin(TASK_FOCUS)].copy()
        if transition_frame.empty:
            continue

        group_columns = [transition["group_token_col"]]
        if transition["group_parent_col"] is not None:
            group_columns = [transition["group_parent_col"], transition["group_token_col"]]

        grouped = (
            transition_frame.groupby(group_columns + ["task_label"])
            .agg(sample_count=("sample_id", "size"), positive_rate=("label", "mean"))
            .reset_index()
        )
        pivot = grouped.pivot_table(
            index=group_columns, columns="task_label", values=["sample_count", "positive_rate"]
        )
        pivot.columns = [f"{metric}_{task}" for metric, task in pivot.columns]
        pivot = pivot.reset_index()

        for task in TASK_FOCUS:
            for metric in ("sample_count", "positive_rate"):
                column = f"{metric}_{task}"
                if column not in pivot.columns:
                    pivot[column] = np.nan

        pivot = pivot[(pivot["sample_count_sid"] >= 20) & (pivot["sample_count_hisTitle2sid"] >= 20)].copy()
        pivot["rate_gap_his_minus_sid"] = pivot["positive_rate_hisTitle2sid"] - pivot["positive_rate_sid"]
        pivot["abs_rate_gap"] = pivot["rate_gap_his_minus_sid"].abs()
        pivot["transition"] = transition["name"]
        pivot = pivot.sort_values(
            ["abs_rate_gap", "sample_count_sid", "sample_count_hisTitle2sid"], ascending=[False, False, False]
        )
        rows.extend(pivot.head(100).to_dict("records"))
    return rows


def tokenize_title(text: str) -> list[str]:
    tokens = [token for token in TITLE_TOKEN_PATTERN.findall((text or "").lower()) if len(token) >= 3]
    return [token for token in tokens if token not in TITLE_STOPWORDS]


def build_dominant_residual_parent_leaf_stats(
    *,
    bundle_root: Path,
    id2sid: dict[str, list[str]],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    residual = frame[frame["final_unsolved"].astype(bool)].copy()
    if residual.empty or "parent_d4" not in residual.columns:
        return pd.DataFrame(), pd.DataFrame(), None

    dominant_parent = residual["parent_d4"].value_counts().index[0]
    if not dominant_parent:
        return pd.DataFrame(), pd.DataFrame(), None

    item_path = bundle_root / "raw_data/Instruments/Instruments.item.json"
    items = json.loads(item_path.read_text(encoding="utf-8"))
    target_tokens = SID_TOKEN_PATTERN.findall(dominant_parent)

    branch_items = []
    for item_id, tokens in id2sid.items():
        normalized_tokens = [str(token) for token in tokens]
        if normalized_tokens[: len(target_tokens)] != target_tokens:
            continue
        leaf_token = normalized_tokens[len(target_tokens)] if len(normalized_tokens) > len(target_tokens) else ""
        item = items.get(str(item_id), {})
        branch_items.append(
            {
                "item_id": str(item_id),
                "sid": "".join(normalized_tokens),
                "leaf_token": leaf_token,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
            }
        )

    branch_item_df = pd.DataFrame(branch_items)
    branch_cases = frame[frame["parent_d4"] == dominant_parent].copy()
    branch_case_summary = (
        branch_cases.groupby("token_d4")
        .agg(
            total_case_count=("sample_id", "size"),
            unsolved_case_count=("final_unsolved", "sum"),
            sid_case_count=("task_label", lambda series: int((series == "sid").sum())),
            hisTitle2sid_case_count=("task_label", lambda series: int((series == "hisTitle2sid").sum())),
            title_desc2sid_case_count=("task_label", lambda series: int((series == "title_desc2sid").sum())),
            mean_effective_hint_depth=("effective_hint_depth", "mean"),
            global_count_d4=("global_count_d4", "mean"),
            child_share_d4=("child_share_d4", "mean"),
        )
        .reset_index()
        .rename(columns={"token_d4": "leaf_token"})
    )

    leaf_stats = branch_item_df.merge(branch_case_summary, on="leaf_token", how="left")
    leaf_stats = leaf_stats.fillna(
        {
            "total_case_count": 0,
            "unsolved_case_count": 0,
            "sid_case_count": 0,
            "hisTitle2sid_case_count": 0,
            "title_desc2sid_case_count": 0,
        }
    )
    leaf_stats["unsolved_case_count"] = leaf_stats["unsolved_case_count"].astype(int)
    leaf_stats["total_case_count"] = leaf_stats["total_case_count"].astype(int)
    leaf_stats["unsolved_rate_within_leaf"] = leaf_stats["unsolved_case_count"] / leaf_stats[
        "total_case_count"
    ].replace(0, np.nan)
    leaf_stats = leaf_stats.sort_values(
        ["unsolved_case_count", "total_case_count", "global_count_d4"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    residual_leaf_tokens = set(leaf_stats.loc[leaf_stats["unsolved_case_count"] > 0, "leaf_token"])
    term_counts: Counter[tuple[str, str]] = Counter()
    for record in branch_items:
        bucket = "residual_leaf" if record["leaf_token"] in residual_leaf_tokens else "solved_leaf"
        for token in set(tokenize_title(record["title"])):
            term_counts[(bucket, token)] += 1

    all_terms = sorted({term for _, term in term_counts})
    term_rows = []
    residual_leaf_count = max(len(residual_leaf_tokens), 1)
    solved_leaf_count = max(len(set(branch_item_df["leaf_token"]) - residual_leaf_tokens), 1)
    for term in all_terms:
        residual_count = term_counts.get(("residual_leaf", term), 0)
        solved_count = term_counts.get(("solved_leaf", term), 0)
        residual_share = residual_count / residual_leaf_count
        solved_share = solved_count / solved_leaf_count
        term_rows.append(
            {
                "term": term,
                "residual_leaf_count": residual_count,
                "solved_leaf_count": solved_count,
                "residual_leaf_share": residual_share,
                "solved_leaf_share": solved_share,
                "share_gap": residual_share - solved_share,
            }
        )

    term_df = (
        pd.DataFrame(term_rows)
        .sort_values(
            ["share_gap", "residual_leaf_count", "solved_leaf_count"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    return leaf_stats, term_df, dominant_parent


def build_parent_pathology_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parent_frame = frame[frame["parent_d4"].notna()].copy()
    for parent_d4, parent_cases in parent_frame.groupby("parent_d4"):
        leaf_counts = parent_cases["token_d4"].value_counts()
        unsolved_cases = parent_cases[parent_cases["final_unsolved"].astype(bool)].copy()
        unsolved_leaf_counts = unsolved_cases["token_d4"].value_counts()

        dominant_leaf = unsolved_leaf_counts.index[0] if not unsolved_leaf_counts.empty else None
        dominant_unsolved_count = int(unsolved_leaf_counts.iloc[0]) if not unsolved_leaf_counts.empty else 0
        unsolved_total = int(len(unsolved_cases))
        hint3_plus_cases = parent_cases[parent_cases["effective_hint_depth"] >= 3].copy()
        hint3_plus_leaf_counts = hint3_plus_cases["token_d4"].value_counts()

        rows.append(
            {
                "parent_d4": parent_d4,
                "total_case_count": int(len(parent_cases)),
                "observed_leaf_count": int(parent_cases["token_d4"].nunique()),
                "unsolved_case_count": unsolved_total,
                "hint3_plus_case_count": int(len(hint3_plus_cases)),
                "dominant_leaf_by_case_count": leaf_counts.index[0] if not leaf_counts.empty else None,
                "dominant_leaf_case_count": int(leaf_counts.iloc[0]) if not leaf_counts.empty else 0,
                "dominant_unsolved_leaf": dominant_leaf,
                "dominant_unsolved_leaf_count": dominant_unsolved_count,
                "dominant_unsolved_leaf_share": dominant_unsolved_count / max(unsolved_total, 1),
                "dominant_hint3_plus_leaf": hint3_plus_leaf_counts.index[0]
                if not hint3_plus_leaf_counts.empty
                else None,
                "dominant_hint3_plus_leaf_count": int(hint3_plus_leaf_counts.iloc[0])
                if not hint3_plus_leaf_counts.empty
                else 0,
                "dominant_hint3_plus_leaf_share": (
                    int(hint3_plus_leaf_counts.iloc[0]) / max(len(hint3_plus_cases), 1)
                    if not hint3_plus_leaf_counts.empty
                    else 0.0
                ),
                "mean_effective_hint_depth": float(parent_cases["effective_hint_depth"].mean()),
                "sid_case_count": int((parent_cases["task_label"] == "sid").sum()),
                "hisTitle2sid_case_count": int((parent_cases["task_label"] == "hisTitle2sid").sum()),
                "title_desc2sid_case_count": int((parent_cases["task_label"] == "title_desc2sid").sum()),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["dominant_unsolved_leaf_share"],
            row["unsolved_case_count"],
            row["dominant_hint3_plus_leaf_share"],
            row["hint3_plus_case_count"],
        ),
        reverse=True,
    )


def build_leaf_distinctiveness_frame(branch_item_df: pd.DataFrame, leaf_stats: pd.DataFrame) -> pd.DataFrame:
    if branch_item_df.empty:
        return pd.DataFrame()

    item_frame = branch_item_df.copy()
    item_frame["title_terms"] = item_frame["title"].map(lambda title: set(tokenize_title(title)))

    sibling_doc_freq: Counter[str] = Counter()
    for terms in item_frame["title_terms"]:
        for term in terms:
            sibling_doc_freq[term] += 1

    rows: list[dict[str, Any]] = []
    total_leaf_count = len(item_frame)
    for index, record in item_frame.iterrows():
        terms = record["title_terms"]
        other_term_sets = [
            other_terms for row_idx, other_terms in enumerate(item_frame["title_terms"]) if row_idx != index
        ]
        jaccards = []
        for other_terms in other_term_sets:
            union = terms | other_terms
            jaccards.append((len(terms & other_terms) / len(union)) if union else 0.0)

        unique_terms = [term for term in terms if sibling_doc_freq[term] == 1]
        distinctiveness = sum(math.log((1 + total_leaf_count) / (1 + sibling_doc_freq[term])) for term in terms) / max(
            len(terms), 1
        )

        rows.append(
            {
                "leaf_token": record["leaf_token"],
                "title": record["title"],
                "term_count": len(terms),
                "unique_term_count": len(unique_terms),
                "unique_terms": "|".join(sorted(unique_terms)),
                "mean_jaccard_to_siblings": float(np.mean(jaccards)) if jaccards else 0.0,
                "max_jaccard_to_siblings": float(np.max(jaccards)) if jaccards else 0.0,
                "mean_idf_like_distinctiveness": distinctiveness,
            }
        )

    distinctiveness_frame = pd.DataFrame(rows)
    if not leaf_stats.empty:
        merge_columns = [
            column
            for column in [
                "leaf_token",
                "total_case_count",
                "unsolved_case_count",
                "unsolved_rate_within_leaf",
                "global_count_d4",
                "child_share_d4",
            ]
            if column in leaf_stats.columns
        ]
        distinctiveness_frame = distinctiveness_frame.merge(
            leaf_stats[merge_columns],
            on="leaf_token",
            how="left",
        )
    return distinctiveness_frame.sort_values(
        ["unsolved_case_count", "mean_jaccard_to_siblings", "mean_idf_like_distinctiveness"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_parent_title_family_rows(branch_items: pd.DataFrame) -> list[dict[str, Any]]:
    if branch_items.empty:
        return []

    frame = branch_items.copy()
    if "parent_d4" not in frame.columns:
        raise KeyError("branch_items must include parent_d4")
    frame["title_terms"] = frame["title"].map(lambda title: set(tokenize_title(title)))

    rows: list[dict[str, Any]] = []
    for parent_d4, parent_frame in frame.groupby("parent_d4"):
        term_sets = parent_frame["title_terms"].tolist()
        pairwise = []
        for i in range(len(term_sets)):
            for j in range(i + 1, len(term_sets)):
                union = term_sets[i] | term_sets[j]
                pairwise.append((len(term_sets[i] & term_sets[j]) / len(union)) if union else 0.0)

        term_counter: Counter[str] = Counter()
        for terms in term_sets:
            for term in terms:
                term_counter[term] += 1

        threshold = math.ceil(len(parent_frame) / 2)
        shared_terms = sorted(term for term, count in term_counter.items() if count >= threshold)
        rows.append(
            {
                "parent_d4": parent_d4,
                "leaf_count": int(len(parent_frame)),
                "mean_pairwise_jaccard": float(np.mean(pairwise)) if pairwise else 0.0,
                "max_pairwise_jaccard": float(np.max(pairwise)) if pairwise else 0.0,
                "shared_term_count": len(shared_terms),
                "shared_terms": "|".join(shared_terms),
            }
        )

    return sorted(rows, key=lambda row: (row["mean_pairwise_jaccard"], row["leaf_count"]), reverse=True)


def build_all_parent_branch_items(bundle_root: Path, id2sid: dict[str, list[str]]) -> pd.DataFrame:
    item_path = bundle_root / "raw_data/Instruments/Instruments.item.json"
    items = json.loads(item_path.read_text(encoding="utf-8"))

    rows = []
    for item_id, tokens in id2sid.items():
        normalized_tokens = [str(token) for token in tokens]
        if len(normalized_tokens) < 4:
            continue
        parent_d4 = "".join(normalized_tokens[:3])
        leaf_token = normalized_tokens[3]
        item = items.get(str(item_id), {})
        rows.append(
            {
                "parent_d4": parent_d4,
                "leaf_token": leaf_token,
                "item_id": str(item_id),
                "sid": "".join(normalized_tokens),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else bundle_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    difficulty_csv = find_bundle_file(
        bundle_root,
        (
            "GenRec/output/jupyter-notebook/genrec-hint-cascade-artifacts/"
            "instruments_grec_beam16_hint_difficulty_table.csv"
        ),
    )
    id2sid_path = find_bundle_file(
        bundle_root,
        (
            "GenRec/data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_"
            "dsInstruments_ridFeb-10-2026-05-40-47/id2sid.json"
        ),
    )

    df = pd.read_csv(difficulty_csv)
    id2sid = json.loads(id2sid_path.read_text(encoding="utf-8"))
    enriched = attach_per_depth_metrics(df, id2sid)

    thresholds = pd.DataFrame(build_threshold_rows(enriched))
    thresholds.to_csv(output_dir / "hint_transition_feature_thresholds.csv", index=False)

    group_lifts = pd.DataFrame(build_group_rows(enriched))
    group_lifts.to_csv(output_dir / "hint_transition_top_groups.csv", index=False)

    shared_hotspots = pd.DataFrame(build_shared_hotspot_rows(enriched))
    shared_hotspots.to_csv(output_dir / "shared_transition_hotspots.csv", index=False)

    task_gap_groups = pd.DataFrame(build_transition_task_gap_group_rows(enriched))
    task_gap_groups.to_csv(output_dir / "transition_task_gap_groups.csv", index=False)

    residual = build_residual_frame(enriched)
    residual.to_csv(output_dir / "hint_transition_residual_cases.csv", index=False)
    dominant_branch_items_path = export_dominant_residual_parent_items(
        bundle_root=bundle_root,
        id2sid=id2sid,
        residual=residual,
        output_dir=output_dir,
    )
    all_parent_branch_items = build_all_parent_branch_items(bundle_root, id2sid)
    parent_family_summary = pd.DataFrame(build_parent_title_family_rows(all_parent_branch_items))
    dominant_leaf_stats, dominant_title_terms, dominant_parent = build_dominant_residual_parent_leaf_stats(
        bundle_root=bundle_root,
        id2sid=id2sid,
        frame=enriched,
    )
    parent_pathology = pd.DataFrame(build_parent_pathology_rows(enriched))
    parent_pathology_path = output_dir / "parent_leaf_pathology_summary.csv"
    parent_pathology.to_csv(parent_pathology_path, index=False)
    repeated_pathology_candidates = parent_pathology.merge(parent_family_summary, on="parent_d4", how="left")
    repeated_pathology_candidates["pathology_score"] = (
        repeated_pathology_candidates["dominant_unsolved_leaf_share"].fillna(0.0) * 4.0
        + repeated_pathology_candidates["dominant_hint3_plus_leaf_share"].fillna(0.0) * 2.0
        + repeated_pathology_candidates["mean_pairwise_jaccard"].fillna(0.0)
    )
    repeated_pathology_candidates = repeated_pathology_candidates.sort_values(
        [
            "pathology_score",
            "unsolved_case_count",
            "dominant_hint3_plus_leaf_share",
            "hint3_plus_case_count",
            "mean_pairwise_jaccard",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    repeated_pathology_candidates_path = output_dir / "repeated_pathology_candidates.csv"
    repeated_pathology_candidates.to_csv(repeated_pathology_candidates_path, index=False)
    dominant_leaf_stats_path = None
    dominant_title_terms_path = None
    dominant_leaf_distinctiveness_path = None
    if not dominant_leaf_stats.empty:
        dominant_leaf_stats_path = output_dir / "dominant_residual_parent_leaf_stats.csv"
        dominant_leaf_stats.to_csv(dominant_leaf_stats_path, index=False)
        dominant_branch_items = (
            pd.read_csv(dominant_branch_items_path) if dominant_branch_items_path else pd.DataFrame()
        )
        dominant_leaf_distinctiveness = build_leaf_distinctiveness_frame(dominant_branch_items, dominant_leaf_stats)
        dominant_leaf_distinctiveness_path = output_dir / "dominant_residual_parent_leaf_distinctiveness.csv"
        dominant_leaf_distinctiveness.to_csv(dominant_leaf_distinctiveness_path, index=False)
    if not dominant_title_terms.empty:
        dominant_title_terms_path = output_dir / "dominant_residual_parent_title_terms.csv"
        dominant_title_terms.to_csv(dominant_title_terms_path, index=False)

    task_gaps = build_task_gap_rows(enriched)
    summary = {
        "bundle_root": str(bundle_root),
        "output_dir": str(output_dir),
        "num_samples": int(len(enriched)),
        "task_summary": (
            enriched.groupby("task_label")
            .agg(
                sample_count=("sample_id", "size"),
                base_hit_rate=("base_hit", "mean"),
                need_hint_rate=("effective_hint_depth", lambda series: (series >= 1).mean()),
                hint2_plus_rate=("effective_hint_depth", lambda series: (series >= 2).mean()),
                hint3_plus_rate=("effective_hint_depth", lambda series: (series >= 3).mean()),
                unsolved_rate=("final_unsolved", "mean"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "task_gap_standardized": task_gaps,
        "residual_count": int(len(residual)),
        "residual_parent_d4_counts": residual["parent_d4"].value_counts().to_dict(),
        "shared_transition_hotspots_path": str(output_dir / "shared_transition_hotspots.csv"),
        "transition_task_gap_groups_path": str(output_dir / "transition_task_gap_groups.csv"),
        "parent_leaf_pathology_summary_path": str(parent_pathology_path),
        "repeated_pathology_candidates_path": str(repeated_pathology_candidates_path),
        "dominant_residual_parent_items_path": dominant_branch_items_path,
        "dominant_residual_parent": dominant_parent,
        "dominant_residual_parent_leaf_stats_path": str(dominant_leaf_stats_path)
        if dominant_leaf_stats_path
        else None,
        "dominant_residual_parent_leaf_distinctiveness_path": (
            str(dominant_leaf_distinctiveness_path) if dominant_leaf_distinctiveness_path else None
        ),
        "dominant_residual_parent_title_terms_path": str(dominant_title_terms_path)
        if dominant_title_terms_path
        else None,
    }
    (output_dir / "hint_transition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved thresholds to {output_dir / 'hint_transition_feature_thresholds.csv'}")
    print(f"Saved top groups to {output_dir / 'hint_transition_top_groups.csv'}")
    print(f"Saved shared hotspots to {output_dir / 'shared_transition_hotspots.csv'}")
    print(f"Saved transition task gaps to {output_dir / 'transition_task_gap_groups.csv'}")
    print(f"Saved residual cases to {output_dir / 'hint_transition_residual_cases.csv'}")
    print(f"Saved parent pathology summary to {parent_pathology_path}")
    print(f"Saved repeated pathology candidates to {repeated_pathology_candidates_path}")
    if dominant_branch_items_path is not None:
        print(f"Saved dominant residual parent items to {dominant_branch_items_path}")
    if dominant_leaf_stats_path is not None:
        print(f"Saved dominant residual parent leaf stats to {dominant_leaf_stats_path}")
    if dominant_leaf_distinctiveness_path is not None:
        print(f"Saved dominant residual parent leaf distinctiveness to {dominant_leaf_distinctiveness_path}")
    if dominant_title_terms_path is not None:
        print(f"Saved dominant residual parent title terms to {dominant_title_terms_path}")
    print(f"Saved summary to {output_dir / 'hint_transition_summary.json'}")


if __name__ == "__main__":
    main()
