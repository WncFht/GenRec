#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


TRANSITIONS = [
    {
        "name": "base_to_need_hint",
        "frame_filter": lambda frame: frame.index == frame.index,
        "label": lambda frame: frame["effective_hint_depth"] >= 1,
        "depth": 1,
        "parent_col": None,
        "token_col": "token_d1",
    },
    {
        "name": "hint1_to_need_hint2_plus",
        "frame_filter": lambda frame: frame["effective_hint_depth"] >= 1,
        "label": lambda frame: frame["effective_hint_depth"] >= 2,
        "depth": 2,
        "parent_col": "parent_d2",
        "token_col": "token_d2",
    },
    {
        "name": "hint2_to_need_hint3_plus",
        "frame_filter": lambda frame: frame["effective_hint_depth"] >= 2,
        "label": lambda frame: frame["effective_hint_depth"] >= 3,
        "depth": 3,
        "parent_col": "parent_d3",
        "token_col": "token_d3",
    },
    {
        "name": "hint3_to_unsolved",
        "frame_filter": lambda frame: frame["effective_hint_depth"] >= 3,
        "label": lambda frame: frame["final_unsolved"].astype(bool),
        "depth": 4,
        "parent_col": "parent_d4",
        "token_col": "token_d4",
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

TASK_ORDER = ["all", "sid", "hisTitle2sid", "title_desc2sid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local rollout node difficulty for a hint bundle.")
    parser.add_argument(
        "--bundle-root",
        default=(
            "/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle"
        ),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def load_explore_module(repo_root: Path) -> Any:
    script_path = repo_root / "scripts/hint_research/explore_local_hint_bundle.py"
    spec = importlib.util.spec_from_file_location("explore_local_hint_bundle", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spearman_like(series_a: pd.Series, series_b: pd.Series) -> float:
    ranked_a = series_a.rank(method="average")
    ranked_b = series_b.rank(method="average")
    return float(ranked_a.corr(ranked_b))


def feature_columns(depth: int) -> list[str]:
    return [f"{feature}_d{depth}" for feature in FEATURE_BASES]


def build_transition_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for transition in TRANSITIONS:
        subset = frame[transition["frame_filter"](frame)].copy()
        subset["label"] = transition["label"](subset).astype(int)
        outputs[transition["name"]] = subset
    return outputs


def build_node_stats_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transition_frames = build_transition_frames(frame)
    for transition in TRANSITIONS:
        subset = transition_frames[transition["name"]]
        depth = transition["depth"]
        parent_col = transition["parent_col"]
        token_col = transition["token_col"]
        group_columns = [token_col] if parent_col is None else [parent_col, token_col]

        for task_label in TASK_ORDER:
            task_subset = subset if task_label == "all" else subset[subset["task_label"] == task_label]
            if task_subset.empty:
                continue

            aggregated = (
                task_subset.groupby(group_columns)
                .agg(
                    sample_count=("sample_id", "size"),
                    fail_count=("label", "sum"),
                    fail_rate=("label", "mean"),
                    **{feature: (feature, "mean") for feature in feature_columns(depth)},
                )
                .reset_index()
            )
            aggregated["task_label"] = task_label
            aggregated["transition"] = transition["name"]
            aggregated["depth"] = depth
            aggregated = aggregated.rename(columns=dict(zip(feature_columns(depth), FEATURE_BASES)))
            if parent_col is None:
                aggregated["parent_key"] = "ROOT"
            else:
                aggregated["parent_key"] = aggregated[parent_col]
            aggregated["token_key"] = aggregated[token_col]
            aggregated = aggregated.sort_values(["fail_rate", "sample_count"], ascending=[False, False]).reset_index(
                drop=True
            )
            aggregated["fail_rate_rank"] = aggregated["fail_rate"].rank(method="dense", ascending=False).astype(int)
            rows.extend(
                aggregated[
                    [
                        "transition",
                        "task_label",
                        "depth",
                        "parent_key",
                        "token_key",
                        "sample_count",
                        "fail_count",
                        "fail_rate",
                        "fail_rate_rank",
                    ]
                    + FEATURE_BASES
                ].to_dict("records")
            )
    return rows


def build_parent_stats_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transition_frames = build_transition_frames(frame)
    for transition in TRANSITIONS[1:]:
        subset = transition_frames[transition["name"]]
        depth = transition["depth"]
        parent_col = transition["parent_col"]
        assert parent_col is not None

        for task_label in TASK_ORDER:
            task_subset = subset if task_label == "all" else subset[subset["task_label"] == task_label]
            if task_subset.empty:
                continue

            aggregated = (
                task_subset.groupby(parent_col)
                .agg(
                    sample_count=("sample_id", "size"),
                    fail_count=("label", "sum"),
                    fail_rate=("label", "mean"),
                    sibling=(f"sibling_d{depth}", "mean"),
                    parent_entropy=(f"parent_entropy_d{depth}", "mean"),
                    parent_maxshare=(f"parent_maxshare_d{depth}", "mean"),
                    observed_token_count=(transition["token_col"], "nunique"),
                )
                .reset_index()
                .rename(columns={parent_col: "parent_key"})
            )
            aggregated["task_label"] = task_label
            aggregated["transition"] = transition["name"]
            aggregated["depth"] = depth
            aggregated = aggregated.sort_values(
                ["fail_count", "fail_rate", "sample_count"], ascending=[False, False, False]
            ).reset_index(drop=True)
            aggregated["fail_count_rank"] = aggregated["fail_count"].rank(method="dense", ascending=False).astype(int)
            rows.extend(aggregated.to_dict("records"))
    return rows


def build_feature_bin_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transition_frames = build_transition_frames(frame)
    for transition in TRANSITIONS:
        subset = transition_frames[transition["name"]]
        depth = transition["depth"]
        for feature in feature_columns(depth):
            bins = pd.qcut(subset[feature].rank(method="first"), 5, labels=False, duplicates="drop")
            grouped = (
                subset.groupby(bins)
                .agg(
                    sample_count=("sample_id", "size"),
                    fail_count=("label", "sum"),
                    fail_rate=("label", "mean"),
                    min_value=(feature, "min"),
                    max_value=(feature, "max"),
                )
                .reset_index()
                .rename(columns={feature: "bin_index"})
            )
            grouped["transition"] = transition["name"]
            grouped["depth"] = depth
            grouped["feature"] = feature
            grouped["bin_index"] = grouped["index"] + 1 if "index" in grouped.columns else grouped.iloc[:, 0] + 1
            if grouped.columns[0] != "bin_index":
                grouped = grouped.drop(columns=[grouped.columns[0]])
            rows.extend(grouped.to_dict("records"))
    return rows


def build_structure_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth in range(1, 5):
        token_col = f"token_d{depth}"
        parent_col = f"parent_d{depth}"
        sibling_col = f"sibling_d{depth}"
        parent_maxshare_col = f"parent_maxshare_d{depth}"
        parent_entropy_col = f"parent_entropy_d{depth}"
        dedup = frame[[parent_col, sibling_col, parent_maxshare_col, parent_entropy_col]].drop_duplicates().dropna()
        rows.append(
            {
                "depth": depth,
                "active_token_count": int(frame[token_col].nunique(dropna=True)),
                "active_parent_count": int(frame[parent_col].nunique(dropna=True)),
                "mean_sibling": float(dedup[sibling_col].mean()),
                "median_sibling": float(dedup[sibling_col].median()),
                "max_sibling": float(dedup[sibling_col].max()),
                "mean_parent_maxshare": float(dedup[parent_maxshare_col].mean()),
                "median_parent_maxshare": float(dedup[parent_maxshare_col].median()),
                "mean_parent_entropy": float(dedup[parent_entropy_col].mean()),
                "median_parent_entropy": float(dedup[parent_entropy_col].median()),
            }
        )
    return rows


def top_rows(frame: pd.DataFrame, sort_columns: list[str], ascending: list[bool], head: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.sort_values(sort_columns, ascending=ascending).head(head).to_dict("records")


def build_summary(
    *,
    frame: pd.DataFrame,
    node_stats: pd.DataFrame,
    parent_stats: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["tree_structure"] = build_structure_summary(frame)

    transition_frames = build_transition_frames(frame)
    summary["transition_rates"] = []
    for transition in TRANSITIONS:
        subset = transition_frames[transition["name"]]
        task_rates = (
            subset.groupby("task_label")
            .agg(sample_count=("sample_id", "size"), positive_rate=("label", "mean"))
            .reset_index()
            .to_dict("records")
        )
        summary["transition_rates"].append(
            {
                "transition": transition["name"],
                "depth": transition["depth"],
                "overall_positive_rate": float(subset["label"].mean()),
                "task_rates": task_rates,
            }
        )

    root_stats = node_stats[
        (node_stats["transition"] == "base_to_need_hint") & (node_stats["task_label"] == "all")
    ].copy()
    root_stats = root_stats[root_stats["sample_count"] >= 200].copy()
    summary["top_risky_roots"] = top_rows(root_stats, ["fail_rate", "sample_count"], [False, False], 15)
    summary["safest_roots"] = top_rows(root_stats, ["fail_rate", "sample_count"], [True, False], 15)

    depth2_parent_stats = parent_stats[
        (parent_stats["transition"] == "hint1_to_need_hint2_plus") & (parent_stats["task_label"] == "all")
    ].copy()
    summary["worst_depth2_parents"] = top_rows(
        depth2_parent_stats[depth2_parent_stats["sample_count"] >= 500],
        ["fail_rate", "sample_count"],
        [False, False],
        15,
    )

    depth3_parent_stats = parent_stats[
        (parent_stats["transition"] == "hint2_to_need_hint3_plus") & (parent_stats["task_label"] == "all")
    ].copy()
    summary["worst_depth3_parents_by_fail_count"] = top_rows(
        depth3_parent_stats, ["fail_count", "fail_rate"], [False, False], 15
    )

    concentration = []
    base_all = transition_frames["base_to_need_hint"]
    root_ranked = root_stats.sort_values(["fail_rate", "sample_count"], ascending=[False, False]).reset_index(
        drop=True
    )
    for top_k in [5, 10, 20]:
        top = root_ranked.head(top_k)
        concentration.append(
            {
                "transition": "base_to_need_hint",
                "unit": "node",
                "top_k": top_k,
                "sample_share": float(top["sample_count"].sum() / len(base_all)),
                "failure_share": float(top["fail_count"].sum() / max(base_all["label"].sum(), 1)),
            }
        )

    hint1_all = transition_frames["hint1_to_need_hint2_plus"]
    depth2_parent_ranked = depth2_parent_stats.sort_values(
        ["fail_count", "fail_rate"], ascending=[False, False]
    ).reset_index(drop=True)
    for top_k in [5, 10, 20]:
        top = depth2_parent_ranked.head(top_k)
        concentration.append(
            {
                "transition": "hint1_to_need_hint2_plus",
                "unit": "parent",
                "top_k": top_k,
                "sample_share": float(top["sample_count"].sum() / len(hint1_all)),
                "failure_share": float(top["fail_count"].sum() / max(hint1_all["label"].sum(), 1)),
            }
        )

    hint2_all = transition_frames["hint2_to_need_hint3_plus"]
    depth3_parent_ranked = depth3_parent_stats.sort_values(
        ["fail_count", "fail_rate"], ascending=[False, False]
    ).reset_index(drop=True)
    for top_k in [5, 10, 20]:
        top = depth3_parent_ranked.head(top_k)
        concentration.append(
            {
                "transition": "hint2_to_need_hint3_plus",
                "unit": "parent",
                "top_k": top_k,
                "sample_share": float(top["sample_count"].sum() / len(hint2_all)),
                "failure_share": float(top["fail_count"].sum() / max(hint2_all["label"].sum(), 1)),
            }
        )
    summary["concentration"] = concentration

    correlations: dict[str, Any] = {}
    if not root_stats.empty:
        correlations["root_level"] = {
            feature: {
                "pearson": float(root_stats[feature].corr(root_stats["fail_rate"])),
                "spearman_like": spearman_like(root_stats[feature], root_stats["fail_rate"]),
            }
            for feature in ["global_count", "task_parent_share", "subtree", "child_rank"]
        }

    if not depth2_parent_stats.empty:
        correlations["depth2_parent_level"] = {
            feature: {
                "pearson": float(depth2_parent_stats[feature].corr(depth2_parent_stats["fail_rate"])),
                "spearman_like": spearman_like(depth2_parent_stats[feature], depth2_parent_stats["fail_rate"]),
            }
            for feature in ["sibling", "parent_maxshare"]
        }

    depth2_node_stats = node_stats[
        (node_stats["transition"] == "hint1_to_need_hint2_plus")
        & (node_stats["task_label"] == "all")
        & (node_stats["sample_count"] >= 20)
    ].copy()
    if not depth2_node_stats.empty:
        correlations["depth2_node_level"] = {
            feature: {
                "pearson": float(depth2_node_stats[feature].corr(depth2_node_stats["fail_rate"])),
                "spearman_like": spearman_like(depth2_node_stats[feature], depth2_node_stats["fail_rate"]),
            }
            for feature in [
                "global_count",
                "task_parent_share",
                "subtree",
                "sibling",
                "parent_maxshare",
            ]
        }
    summary["correlations"] = correlations

    threshold_rows = []
    threshold_specs = [
        ("base_to_need_hint", "task_parent_share_d1"),
        ("base_to_need_hint", "global_count_d1"),
        ("hint1_to_need_hint2_plus", "task_parent_share_d2"),
        ("hint1_to_need_hint2_plus", "sibling_d2"),
        ("hint1_to_need_hint2_plus", "global_count_d2"),
        ("hint2_to_need_hint3_plus", "sibling_d3"),
        ("hint2_to_need_hint3_plus", "task_parent_share_d3"),
    ]
    for transition_name, feature in threshold_specs:
        threshold_row = thresholds[
            (thresholds["transition"] == transition_name)
            & (thresholds["task_label"] == "all")
            & (thresholds["feature"] == feature)
        ]
        if threshold_row.empty:
            continue
        threshold = threshold_row.iloc[0]
        subset = transition_frames[transition_name]
        if threshold["direction"] == "<=":
            mask = subset[feature] <= threshold["threshold"]
        else:
            mask = subset[feature] >= threshold["threshold"]
        threshold_rows.append(
            {
                "transition": transition_name,
                "feature": feature,
                "threshold": float(threshold["threshold"]),
                "direction": str(threshold["direction"]),
                "selected_share": float(mask.mean()),
                "selected_positive_rate": float(subset.loc[mask, "label"].mean()),
                "positive_recall": float(subset.loc[mask, "label"].sum() / max(subset["label"].sum(), 1)),
            }
        )
    summary["threshold_coverage"] = threshold_rows
    return summary


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    bundle_root = Path(args.bundle_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else bundle_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    explore = load_explore_module(repo_root)
    difficulty_csv = (
        bundle_root
        / "GenRec/output/jupyter-notebook/genrec-hint-cascade-artifacts/instruments_grec_beam16_hint_difficulty_table.csv"
    )
    id2sid_path = (
        bundle_root
        / "GenRec/data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47/id2sid.json"
    )
    thresholds_path = output_dir / "hint_transition_feature_thresholds.csv"

    frame = pd.read_csv(difficulty_csv)
    id2sid = json.loads(id2sid_path.read_text(encoding="utf-8"))
    enriched = explore.attach_per_depth_metrics(frame, id2sid)
    thresholds = pd.read_csv(thresholds_path)

    node_stats = pd.DataFrame(build_node_stats_rows(enriched))
    node_stats_path = output_dir / "rollout_node_stats.csv"
    node_stats.to_csv(node_stats_path, index=False)

    parent_stats = pd.DataFrame(build_parent_stats_rows(enriched))
    parent_stats_path = output_dir / "rollout_parent_stats.csv"
    parent_stats.to_csv(parent_stats_path, index=False)

    feature_bins = pd.DataFrame(build_feature_bin_rows(enriched))
    feature_bins_path = output_dir / "rollout_feature_bin_summary.csv"
    feature_bins.to_csv(feature_bins_path, index=False)

    summary = build_summary(frame=enriched, node_stats=node_stats, parent_stats=parent_stats, thresholds=thresholds)
    summary["node_stats_path"] = str(node_stats_path)
    summary["parent_stats_path"] = str(parent_stats_path)
    summary["feature_bin_summary_path"] = str(feature_bins_path)
    summary_path = output_dir / "rollout_node_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved node stats to {node_stats_path}")
    print(f"Saved parent stats to {parent_stats_path}")
    print(f"Saved feature bin summary to {feature_bins_path}")
    print(f"Saved rollout summary to {summary_path}")


if __name__ == "__main__":
    main()
