#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_ORDER = ["all", "sid", "hisTitle2sid", "title_desc2sid"]
HINT_METRICS = [
    "mean_effective_hint_depth",
    "need_hint_rate",
    "hint2_plus_rate",
    "hint3_plus_rate",
    "unsolved_rate",
    "stage_local_rate",
]
STAGE_BY_DEPTH = {
    1: "base_to_need_hint",
    2: "hint1_to_need_hint2_plus",
    3: "hint2_to_need_hint3_plus",
    4: "hint3_to_unsolved",
}
TASK_LABEL_MAP = {
    "all": "总体",
    "sid": "sid",
    "hisTitle2sid": "hisTitle2sid",
    "title_desc2sid": "title_desc2sid",
}
METRIC_LABEL_MAP = {
    "mean_effective_hint_depth": "平均有效 hint 深度",
    "need_hint_rate": "需要 hint 比例",
    "hint2_plus_rate": "需要 2+ hint 比例",
    "hint3_plus_rate": "需要 3+ hint 比例",
    "unsolved_rate": "最终未解比例",
    "stage_local_rate": "本层继续坍塌比例",
}


def zh_task_label(label: Any) -> str:
    return TASK_LABEL_MAP.get(str(label), str(label))


def zh_metric_label(label: Any) -> str:
    return METRIC_LABEL_MAP.get(str(label), str(label))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the relationship between node hint strength and subtree size."
    )
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


def normalize_task_labels(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "task_label" not in normalized.columns:
        if "task" not in normalized.columns:
            raise KeyError("Expected either 'task_label' or 'task' in the input frame.")
        normalized["task_label"] = normalized["task"].astype(str)
    normalized["task_label"] = normalized["task_label"].astype(str)
    return normalized


def load_enriched_frame(bundle_root: Path) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[2]
    explore = load_explore_module(repo_root)
    difficulty_csv = explore.find_bundle_file(
        bundle_root,
        (
            "GenRec/output/jupyter-notebook/genrec-hint-cascade-artifacts/"
            "instruments_grec_beam16_hint_difficulty_table.csv"
        ),
    )
    id2sid_path = explore.find_bundle_file(
        bundle_root,
        (
            "GenRec/data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_"
            "dsInstruments_ridFeb-10-2026-05-40-47/id2sid.json"
        ),
    )
    frame = pd.read_csv(difficulty_csv)
    frame = normalize_task_labels(frame)
    id2sid = json.loads(id2sid_path.read_text(encoding="utf-8"))
    return explore.attach_per_depth_metrics(frame, id2sid)


def stage_local_columns(frame: pd.DataFrame, depth: int) -> tuple[str, pd.Series, pd.Series]:
    if depth == 1:
        eligible = pd.Series(True, index=frame.index)
        positive = frame["effective_hint_depth"] >= 1
    elif depth == 2:
        eligible = frame["effective_hint_depth"] >= 1
        positive = frame["effective_hint_depth"] >= 2
    elif depth == 3:
        eligible = frame["effective_hint_depth"] >= 2
        positive = frame["effective_hint_depth"] >= 3
    elif depth == 4:
        eligible = frame["effective_hint_depth"] >= 3
        positive = frame["final_unsolved"].astype(bool)
    else:
        raise ValueError(f"Unsupported depth: {depth}")
    return STAGE_BY_DEPTH[depth], eligible.astype(bool), positive.astype(bool)


def max_depth(frame: pd.DataFrame) -> int:
    token_depths = [int(column.split("_d", 1)[1]) for column in frame.columns if column.startswith("token_d")]
    if not token_depths:
        raise ValueError("No token depth columns found.")
    return max(token_depths)


def task_subsets(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    subsets: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    for task_label in TASK_ORDER[1:]:
        task_frame = frame[frame["task_label"] == task_label].copy()
        if not task_frame.empty:
            subsets.append((task_label, task_frame))
    return subsets


def build_node_stats(frame: pd.DataFrame) -> pd.DataFrame:
    working = normalize_task_labels(frame)
    rows: list[dict[str, Any]] = []

    for depth in range(1, max_depth(working) + 1):
        token_col = f"token_d{depth}"
        parent_col = f"parent_d{depth}"
        subtree_col = f"subtree_d{depth}"
        depth_frame = working[working[token_col].notna()].copy()
        stage_local_name, stage_eligible, stage_positive = stage_local_columns(depth_frame, depth)
        depth_frame["_stage_local_eligible"] = stage_eligible.astype(int)
        depth_frame["_stage_local_positive"] = (stage_positive & stage_eligible).astype(int)

        for task_label, task_frame in task_subsets(depth_frame):
            if task_frame.empty:
                continue

            if depth == 1:
                grouped = task_frame.groupby([token_col], dropna=False)
            else:
                grouped = task_frame.groupby([parent_col, token_col], dropna=False)

            aggregated = grouped.agg(
                sample_count=("sample_id", "size"),
                subtree_size=(subtree_col, "mean"),
                mean_effective_hint_depth=("effective_hint_depth", "mean"),
                need_hint_rate=("effective_hint_depth", lambda series: float((series >= 1).mean())),
                hint2_plus_rate=("effective_hint_depth", lambda series: float((series >= 2).mean())),
                hint3_plus_rate=("effective_hint_depth", lambda series: float((series >= 3).mean())),
                unsolved_rate=("final_unsolved", lambda series: float(pd.Series(series).astype(bool).mean())),
                stage_local_eligible_count=("_stage_local_eligible", "sum"),
                stage_local_positive_count=("_stage_local_positive", "sum"),
            ).reset_index()

            if depth == 1:
                aggregated["parent_key"] = "ROOT"
            else:
                aggregated["parent_key"] = aggregated[parent_col].astype(str)
            aggregated["token_key"] = aggregated[token_col].astype(str)
            aggregated["node_key"] = aggregated["parent_key"] + aggregated["token_key"]
            aggregated.loc[aggregated["parent_key"] == "ROOT", "node_key"] = aggregated["token_key"]
            aggregated["stage_local_rate"] = aggregated["stage_local_positive_count"] / aggregated[
                "stage_local_eligible_count"
            ].replace(0, np.nan)
            aggregated["stage_local_name"] = stage_local_name
            aggregated["depth"] = depth
            aggregated["task_label"] = task_label
            rows.extend(
                aggregated[
                    [
                        "task_label",
                        "depth",
                        "stage_local_name",
                        "parent_key",
                        "token_key",
                        "node_key",
                        "sample_count",
                        "subtree_size",
                        "mean_effective_hint_depth",
                        "need_hint_rate",
                        "hint2_plus_rate",
                        "hint3_plus_rate",
                        "unsolved_rate",
                        "stage_local_eligible_count",
                        "stage_local_positive_count",
                        "stage_local_rate",
                    ]
                ].to_dict("records")
            )

    node_stats = pd.DataFrame(rows)
    if node_stats.empty:
        return node_stats
    numeric_columns = [
        "sample_count",
        "subtree_size",
        "mean_effective_hint_depth",
        "need_hint_rate",
        "hint2_plus_rate",
        "hint3_plus_rate",
        "unsolved_rate",
        "stage_local_eligible_count",
        "stage_local_positive_count",
        "stage_local_rate",
    ]
    for column in numeric_columns:
        node_stats[column] = pd.to_numeric(node_stats[column], errors="coerce")
    return node_stats.sort_values(
        ["task_label", "depth", "sample_count", "node_key"], ascending=[True, True, False, True]
    ).reset_index(drop=True)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def build_bin_summary(node_stats: pd.DataFrame, quantiles: int = 8) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if node_stats.empty:
        return pd.DataFrame(
            columns=[
                "task_label",
                "depth",
                "bin_index",
                "node_count",
                "sample_count",
                "subtree_size_min",
                "subtree_size_median",
                "subtree_size_max",
                "mean_effective_hint_depth",
                "need_hint_rate",
                "hint2_plus_rate",
                "hint3_plus_rate",
                "unsolved_rate",
                "stage_local_name",
                "stage_local_eligible_count",
                "stage_local_positive_count",
                "stage_local_rate",
            ]
        )
    for task_label in TASK_ORDER:
        for depth in sorted(node_stats["depth"].dropna().unique()):
            subset = node_stats[
                (node_stats["task_label"] == task_label)
                & node_stats["subtree_size"].notna()
                & (node_stats["subtree_size"] > 0)
            ].copy()
            subset = subset[subset["depth"] == depth].copy()
            if subset.empty:
                continue

            bin_count = min(quantiles, len(subset), subset["subtree_size"].nunique())
            if bin_count <= 1:
                subset["subtree_bin"] = 0
            else:
                subset["subtree_bin"] = pd.qcut(
                    subset["subtree_size"].rank(method="first"),
                    q=bin_count,
                    labels=False,
                    duplicates="drop",
                )

            for bin_index, bin_frame in subset.groupby("subtree_bin"):
                sample_weights = bin_frame["sample_count"].fillna(0)
                eligible_weights = bin_frame["stage_local_eligible_count"].fillna(0)
                positive_total = float(bin_frame["stage_local_positive_count"].fillna(0).sum())
                eligible_total = float(eligible_weights.sum())
                rows.append(
                    {
                        "task_label": task_label,
                        "depth": int(depth),
                        "bin_index": int(bin_index) + 1,
                        "node_count": int(len(bin_frame)),
                        "sample_count": float(sample_weights.sum()),
                        "subtree_size_min": float(bin_frame["subtree_size"].min()),
                        "subtree_size_median": float(bin_frame["subtree_size"].median()),
                        "subtree_size_max": float(bin_frame["subtree_size"].max()),
                        "mean_effective_hint_depth": weighted_mean(
                            bin_frame["mean_effective_hint_depth"], sample_weights
                        ),
                        "need_hint_rate": weighted_mean(bin_frame["need_hint_rate"], sample_weights),
                        "hint2_plus_rate": weighted_mean(bin_frame["hint2_plus_rate"], sample_weights),
                        "hint3_plus_rate": weighted_mean(bin_frame["hint3_plus_rate"], sample_weights),
                        "unsolved_rate": weighted_mean(bin_frame["unsolved_rate"], sample_weights),
                        "stage_local_name": str(bin_frame["stage_local_name"].iloc[0]),
                        "stage_local_eligible_count": eligible_total,
                        "stage_local_positive_count": positive_total,
                        "stage_local_rate": positive_total / eligible_total if eligible_total > 0 else np.nan,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "task_label",
                "depth",
                "bin_index",
                "node_count",
                "sample_count",
                "subtree_size_min",
                "subtree_size_median",
                "subtree_size_max",
                "mean_effective_hint_depth",
                "need_hint_rate",
                "hint2_plus_rate",
                "hint3_plus_rate",
                "unsolved_rate",
                "stage_local_name",
                "stage_local_eligible_count",
                "stage_local_positive_count",
                "stage_local_rate",
            ]
        )
    return pd.DataFrame(rows).sort_values(["task_label", "depth", "bin_index"]).reset_index(drop=True)


def weighted_pearson(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    valid = x.notna() & y.notna() & weights.notna() & (weights > 0)
    if valid.sum() < 2:
        return np.nan
    x_arr = x[valid].to_numpy(dtype=float)
    y_arr = y[valid].to_numpy(dtype=float)
    w_arr = weights[valid].to_numpy(dtype=float)
    x_mean = np.average(x_arr, weights=w_arr)
    y_mean = np.average(y_arr, weights=w_arr)
    covariance = np.average((x_arr - x_mean) * (y_arr - y_mean), weights=w_arr)
    x_var = np.average((x_arr - x_mean) ** 2, weights=w_arr)
    y_var = np.average((y_arr - y_mean) ** 2, weights=w_arr)
    if x_var <= 0 or y_var <= 0:
        return np.nan
    return float(covariance / np.sqrt(x_var * y_var))


def build_correlation_summary(node_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if node_stats.empty:
        return pd.DataFrame(
            columns=[
                "task_label",
                "depth",
                "stage_local_name",
                "metric",
                "node_count",
                "sample_count",
                "pearson",
                "spearman",
                "weighted_pearson",
            ]
        )
    for task_label in TASK_ORDER:
        for depth in sorted(node_stats["depth"].dropna().unique()):
            subset = node_stats[
                (node_stats["task_label"] == task_label)
                & (node_stats["depth"] == depth)
                & node_stats["subtree_size"].notna()
                & (node_stats["subtree_size"] > 0)
            ].copy()
            if subset.empty:
                continue
            x = np.log1p(subset["subtree_size"])
            for metric in HINT_METRICS:
                metric_frame = subset[[metric, "sample_count", "stage_local_eligible_count"]].copy()
                valid = x.notna() & metric_frame[metric].notna()
                if valid.sum() < 2:
                    continue
                x_valid = x[valid]
                y_valid = metric_frame.loc[valid, metric]
                if x_valid.nunique() < 2 or y_valid.nunique() < 2:
                    continue
                if metric == "stage_local_rate":
                    weight_col = metric_frame.loc[valid, "stage_local_eligible_count"].fillna(0)
                else:
                    weight_col = metric_frame.loc[valid, "sample_count"].fillna(0)

                pearson = float(x_valid.corr(y_valid))
                spearman = float(x_valid.rank(method="average").corr(y_valid.rank(method="average")))
                weighted = weighted_pearson(x_valid, y_valid, weight_col)
                rows.append(
                    {
                        "task_label": task_label,
                        "depth": int(depth),
                        "stage_local_name": str(subset["stage_local_name"].iloc[0]),
                        "metric": metric,
                        "node_count": int(valid.sum()),
                        "sample_count": float(metric_frame.loc[valid, "sample_count"].sum()),
                        "pearson": pearson,
                        "spearman": spearman,
                        "weighted_pearson": weighted,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "task_label",
                "depth",
                "stage_local_name",
                "metric",
                "node_count",
                "sample_count",
                "pearson",
                "spearman",
                "weighted_pearson",
            ]
        )
    return pd.DataFrame(rows).sort_values(["task_label", "depth", "metric"]).reset_index(drop=True)


def fit_weighted_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    try:
        slope, intercept = np.polyfit(x, y, deg=1, w=weights)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return float(slope), float(intercept)


def build_outlier_rows(node_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_label in TASK_ORDER:
        for depth in sorted(node_stats["depth"].dropna().unique()):
            subset = node_stats[
                (node_stats["task_label"] == task_label)
                & (node_stats["depth"] == depth)
                & node_stats["subtree_size"].notna()
                & (node_stats["subtree_size"] > 0)
            ].copy()
            if len(subset) < 3:
                continue
            x = np.log1p(subset["subtree_size"].to_numpy(dtype=float))
            y = subset["mean_effective_hint_depth"].to_numpy(dtype=float)
            weights = np.sqrt(np.maximum(subset["sample_count"].to_numpy(dtype=float), 1.0))
            slope, intercept = fit_weighted_line(x, y, weights)
            if np.isnan(slope):
                continue
            subset["predicted_mean_effective_hint_depth"] = slope * x + intercept
            subset["residual_hint_depth"] = (
                subset["mean_effective_hint_depth"] - subset["predicted_mean_effective_hint_depth"]
            )
            residual_std = float(subset["residual_hint_depth"].std(ddof=0))
            if residual_std > 0:
                subset["residual_zscore"] = subset["residual_hint_depth"] / residual_std
            else:
                subset["residual_zscore"] = 0.0
            subset["slope"] = slope
            subset["intercept"] = intercept
            rows.extend(subset.to_dict("records"))
    outliers = pd.DataFrame(rows)
    if outliers.empty:
        return outliers
    return outliers.sort_values(
        ["task_label", "depth", "residual_hint_depth", "sample_count"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def safe_slug(text: str) -> str:
    return text.replace("+", "_plus_").replace("/", "_").replace(" ", "_")


def plot_scatter_panels(node_stats: pd.DataFrame, plot_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if node_stats.empty:
        return paths
    for task_label in TASK_ORDER:
        for depth in sorted(node_stats["depth"].dropna().unique()):
            subset = node_stats[
                (node_stats["task_label"] == task_label)
                & (node_stats["depth"] == depth)
                & node_stats["subtree_size"].notna()
                & (node_stats["subtree_size"] > 0)
            ].copy()
            if subset.empty:
                continue
            x = np.log1p(subset["subtree_size"].to_numpy(dtype=float))
            y = subset["mean_effective_hint_depth"].to_numpy(dtype=float)
            sample_count = subset["sample_count"].to_numpy(dtype=float)
            stage_values = subset["stage_local_rate"].fillna(0).to_numpy(dtype=float)
            marker_sizes = 18.0 + 120.0 * np.sqrt(sample_count / max(sample_count.max(), 1.0))

            fig, ax = plt.subplots(figsize=(7, 4.8))
            scatter = ax.scatter(
                x,
                y,
                s=marker_sizes,
                c=stage_values,
                cmap="viridis",
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
            )
            ax.set_title(f"{zh_task_label(task_label)} 第 {depth} 层散点图")
            ax.set_xlabel("log1p(子树规模)")
            ax.set_ylabel("平均有效 hint 深度")
            ax.grid(alpha=0.25, linewidth=0.5)
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label("本层继续坍塌比例")

            highlight = subset.sort_values(
                ["sample_count", "mean_effective_hint_depth"], ascending=[False, False]
            ).head(5)
            for _, row in highlight.iterrows():
                ax.annotate(
                    row["token_key"],
                    (np.log1p(float(row["subtree_size"])), float(row["mean_effective_hint_depth"])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.8,
                )

            out_path = plot_dir / f"scatter_{safe_slug(task_label)}_depth{int(depth)}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            paths.append(out_path)
    return paths


def plot_bin_lines(bin_summary: pd.DataFrame, plot_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if bin_summary.empty:
        return paths
    for task_label in TASK_ORDER:
        for depth in sorted(bin_summary["depth"].dropna().unique()):
            subset = bin_summary[(bin_summary["task_label"] == task_label) & (bin_summary["depth"] == depth)].copy()
            if subset.empty:
                continue
            subset = subset.sort_values("bin_index")
            x = subset["bin_index"].to_numpy(dtype=float)

            fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
            ax_top.plot(x, subset["mean_effective_hint_depth"], marker="o", color="#d1495b", linewidth=1.8)
            ax_top.set_ylabel("平均有效 hint 深度")
            ax_top.set_title(f"{zh_task_label(task_label)} 第 {depth} 层分箱趋势")
            ax_top.grid(alpha=0.25, linewidth=0.5)

            rate_specs = [
                ("need_hint_rate", "需要 hint"),
                ("hint2_plus_rate", "需要 2+ hint"),
                ("hint3_plus_rate", "需要 3+ hint"),
                ("unsolved_rate", "最终未解"),
                ("stage_local_rate", "本层继续坍塌"),
            ]
            for column, label in rate_specs:
                if column not in subset.columns:
                    continue
                ax_bottom.plot(x, subset[column], marker="o", linewidth=1.4, label=label)
            ax_bottom.set_ylim(-0.02, 1.02)
            ax_bottom.set_xlabel("子树规模分位箱")
            ax_bottom.set_ylabel("比例")
            ax_bottom.grid(alpha=0.25, linewidth=0.5)
            ax_bottom.legend(frameon=False, ncol=3, fontsize=8)
            ax_bottom.set_xticks(x)
            ax_bottom.set_xticklabels([str(int(v)) for v in x])

            out_path = plot_dir / f"bin_lines_{safe_slug(task_label)}_depth{int(depth)}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            paths.append(out_path)
    return paths


def draw_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    row_labels = []
    for index in matrix.index:
        text = str(index)
        if " d" in text:
            task_label, depth = text.rsplit(" d", 1)
            row_labels.append(f"{zh_task_label(task_label)} 第{depth}层")
        else:
            row_labels.append(zh_task_label(text))
    col_labels = [zh_metric_label(str(column)) for column in matrix.columns]
    values = matrix.to_numpy(dtype=float)

    fig_width = max(7.0, 1.1 * len(col_labels))
    fig_height = max(5.0, 0.55 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isnan(value):
                label = "nan"
            else:
                label = f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_correlation_heatmaps(correlations: pd.DataFrame, plot_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if correlations.empty:
        return paths
    working = correlations.copy()
    working["row_label"] = working["task_label"] + " d" + working["depth"].astype(str)

    pearson_matrix = working.pivot(index="row_label", columns="metric", values="pearson").reindex(columns=HINT_METRICS)
    pearson_path = plot_dir / "correlation_heatmap_pearson.png"
    draw_heatmap(
        pearson_matrix,
        title="与 log1p(子树规模) 的 Pearson 相关系数",
        out_path=pearson_path,
    )
    paths.append(pearson_path)

    spearman_matrix = working.pivot(index="row_label", columns="metric", values="spearman").reindex(
        columns=HINT_METRICS
    )
    spearman_path = plot_dir / "correlation_heatmap_spearman.png"
    draw_heatmap(
        spearman_matrix,
        title="与 log1p(子树规模) 的 Spearman 相关系数",
        out_path=spearman_path,
    )
    paths.append(spearman_path)
    return paths


def plot_outliers(outliers: pd.DataFrame, plot_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if outliers.empty:
        return paths
    for task_label in TASK_ORDER:
        for depth in sorted(outliers["depth"].dropna().unique()):
            subset = outliers[(outliers["task_label"] == task_label) & (outliers["depth"] == depth)].copy()
            if subset.empty:
                continue
            subset = subset.sort_values("residual_hint_depth", ascending=False)
            x = np.log1p(subset["subtree_size"].to_numpy(dtype=float))
            y = subset["residual_hint_depth"].to_numpy(dtype=float)
            sample_count = subset["sample_count"].to_numpy(dtype=float)
            marker_sizes = 20.0 + 130.0 * np.sqrt(sample_count / max(sample_count.max(), 1.0))

            fig, ax = plt.subplots(figsize=(7, 4.8))
            ax.scatter(
                x,
                y,
                s=marker_sizes,
                c=subset["mean_effective_hint_depth"],
                cmap="plasma",
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
            )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_title(f"{zh_task_label(task_label)} 第 {depth} 层异常点")
            ax.set_xlabel("log1p(子树规模)")
            ax.set_ylabel("相对趋势残差")
            ax.grid(alpha=0.25, linewidth=0.5)

            label_rows = pd.concat([subset.head(4), subset.tail(4)]).drop_duplicates(subset=["node_key"])
            for _, row in label_rows.iterrows():
                ax.annotate(
                    row["token_key"],
                    (np.log1p(float(row["subtree_size"])), float(row["residual_hint_depth"])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.8,
                )

            out_path = plot_dir / f"outliers_{safe_slug(task_label)}_depth{int(depth)}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            paths.append(out_path)
    return paths


def build_plot_paths(
    *,
    node_stats: pd.DataFrame,
    bin_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    outliers: pd.DataFrame,
    plot_dir: Path,
) -> dict[str, list[Path]]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    return {
        "scatter": plot_scatter_panels(node_stats, plot_dir),
        "bins": plot_bin_lines(bin_summary, plot_dir),
        "heatmap": plot_correlation_heatmaps(correlations, plot_dir),
        "outliers": plot_outliers(outliers, plot_dir),
    }


def top_correlation_row(correlations: pd.DataFrame, column: str, *, largest: bool) -> dict[str, Any] | None:
    if correlations.empty or column not in correlations.columns:
        return None
    working = correlations.dropna(subset=[column]).copy()
    if working.empty:
        return None
    row = working.nlargest(1, column).iloc[0] if largest else working.nsmallest(1, column).iloc[0]
    return row.to_dict()


def build_summary(
    *,
    enriched: pd.DataFrame,
    node_stats: pd.DataFrame,
    bin_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    outliers: pd.DataFrame,
    plot_paths: dict[str, list[Path]],
    output_dir: Path,
) -> dict[str, Any]:
    summary = {
        "num_samples": int(len(enriched)),
        "task_summary": (
            enriched.groupby("task_label")
            .agg(
                sample_count=("sample_id", "size"),
                need_hint_rate=("effective_hint_depth", lambda series: float((series >= 1).mean())),
                hint2_plus_rate=("effective_hint_depth", lambda series: float((series >= 2).mean())),
                hint3_plus_rate=("effective_hint_depth", lambda series: float((series >= 3).mean())),
                unsolved_rate=("final_unsolved", lambda series: float(pd.Series(series).astype(bool).mean())),
            )
            .reset_index()
            .to_dict("records")
        ),
        "artifacts": {
            "node_stats_path": str(output_dir / "hint_subtree_relation_node_stats.csv"),
            "bin_summary_path": str(output_dir / "hint_subtree_relation_bin_summary.csv"),
            "outliers_path": str(output_dir / "hint_subtree_relation_outliers.csv"),
            "correlations_path": str(output_dir / "hint_subtree_relation_correlations.csv"),
            "plots_dir": str(output_dir / "hint_subtree_relation_plots"),
            "html_report_path": str(output_dir / "hint_subtree_relation_report.html"),
        },
        "overall": {
            "strongest_positive_correlation": top_correlation_row(correlations, "pearson", largest=True),
            "strongest_negative_correlation": top_correlation_row(correlations, "pearson", largest=False),
            "strongest_positive_spearman": top_correlation_row(correlations, "spearman", largest=True),
            "strongest_negative_spearman": top_correlation_row(correlations, "spearman", largest=False),
        },
        "largest_subtree_nodes": (
            node_stats.sort_values(["subtree_size", "sample_count"], ascending=[False, False])
            .head(20)
            .to_dict("records")
        ),
        "hardest_nodes_by_mean_hint_depth": (
            node_stats.sort_values(["mean_effective_hint_depth", "sample_count"], ascending=[False, False])
            .head(20)
            .to_dict("records")
        ),
        "most_positive_outliers": (
            outliers.sort_values(["residual_hint_depth", "sample_count"], ascending=[False, False])
            .head(20)
            .to_dict("records")
            if not outliers.empty
            else []
        ),
        "most_negative_outliers": (
            outliers.sort_values(["residual_hint_depth", "sample_count"], ascending=[True, False])
            .head(20)
            .to_dict("records")
            if not outliers.empty
            else []
        ),
        "plot_paths": {
            key: [str(path.relative_to(output_dir)) for path in paths] for key, paths in plot_paths.items()
        },
        "correlations": correlations.to_dict("records"),
        "bin_summary_rows": len(bin_summary),
        "node_count_rows": len(node_stats),
    }
    return summary


def build_report_html(summary: dict[str, Any], plot_paths: dict[str, list[Path]]) -> str:
    kind_label_map = {
        "scatter": "散点图册",
        "bins": "分箱趋势图",
        "heatmap": "相关性热图",
        "outliers": "异常点图册",
    }
    kind_kicker_map = {
        "scatter": "原始节点云",
        "bins": "按子树规模分箱",
        "heatmap": "相关性总览",
        "outliers": "残差结构",
    }
    kind_prefix_map = {
        "scatter": "scatter_",
        "bins": "bin_lines_",
        "outliers": "outliers_",
    }

    def format_number(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "n/a"
            if abs(value) >= 100:
                return f"{value:,.0f}"
            if abs(value) >= 10:
                return f"{value:,.1f}"
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)

    def format_percent(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if np.isnan(numeric):
            return "n/a"
        return f"{numeric * 100:.1f}%"

    def format_corr(record: dict[str, Any] | None) -> str:
        if not record:
            return "暂无相关性摘要"
        task = zh_task_label(record.get("task_label", "n/a"))
        depth = record.get("depth", "n/a")
        metric = zh_metric_label(record.get("metric", "n/a"))
        pearson = format_number(record.get("pearson"))
        return f"{task}，第 {depth} 层，{metric}，Pearson {pearson}"

    def summary_card(label: str, value: str, tone: str = "") -> str:
        tone_class = f" tone-{tone}" if tone else ""
        return f"""
        <article class="stat-card{tone_class}">
          <span class="stat-label">{html.escape(label)}</span>
          <strong class="stat-value">{html.escape(value)}</strong>
        </article>
        """

    def task_cards() -> str:
        cards: list[str] = []
        for row in summary.get("task_summary", []):
            label = zh_task_label(row.get("task_label", "unknown"))
            cards.append(
                f"""
                <article class="task-card">
                  <div class="task-card-head">
                    <h3>{html.escape(label)}</h3>
                    <span>{format_number(row.get("sample_count"))} 个样本</span>
                  </div>
                  <dl class="task-metrics">
                    <div><dt>需要 hint</dt><dd>{format_percent(row.get("need_hint_rate"))}</dd></div>
                    <div><dt>需要 2+ hint</dt><dd>{format_percent(row.get("hint2_plus_rate"))}</dd></div>
                    <div><dt>需要 3+ hint</dt><dd>{format_percent(row.get("hint3_plus_rate"))}</dd></div>
                    <div><dt>最终未解</dt><dd>{format_percent(row.get("unsolved_rate"))}</dd></div>
                  </dl>
                </article>
                """
            )
        return "".join(cards) if cards else '<p class="empty-state">暂无任务摘要。</p>'

    def extract_plot_meta(kind: str, path: Path) -> dict[str, str]:
        rel = path.as_posix()
        stem = path.stem
        task = "all"
        depth = "all"
        if kind in kind_prefix_map and stem.startswith(kind_prefix_map[kind]):
            payload = stem[len(kind_prefix_map[kind]) :]
            if "_depth" in payload:
                task, depth = payload.rsplit("_depth", 1)
        title = kind_label_map.get(kind, kind)
        if depth == "all":
            subtitle = zh_task_label(task)
        else:
            subtitle = f"{zh_task_label(task)} · 第 {depth} 层"
        if kind == "heatmap":
            if "pearson" in stem:
                subtitle = "Pearson 相关性"
            elif "spearman" in stem:
                subtitle = "Spearman 相关性"
        return {
            "rel": rel,
            "name": path.name,
            "title": title,
            "subtitle": subtitle,
            "task": task,
            "task_display": zh_task_label(task),
            "depth": depth,
            "depth_display": "全层" if depth == "all" else f"第 {depth} 层",
            "kind": kind,
            "kind_display": kind_label_map.get(kind, kind),
        }

    def plot_question(meta: dict[str, str]) -> str:
        if meta["kind"] == "scatter":
            return f"这张图在问：{meta['task_display']} 的 {meta['depth_display']} 节点里，子树越大时，平均 hint 深度会不会更高？"
        if meta["kind"] == "bins":
            return f"这张图在问：把 {meta['task_display']} 的 {meta['depth_display']} 节点按子树规模分组后，hint 强度会不会随组别系统性变化？"
        if meta["kind"] == "heatmap":
            return "这张图在问：不同 task-depth 与不同 hint 指标，和子树规模之间的相关性到底谁更强、方向如何？"
        if meta["kind"] == "outliers":
            return f"这张图在问：{meta['task_display']} 的 {meta['depth_display']} 节点里，哪些点比 subtree size 本身所能预测的更难或更容易？"
        return "这张图展示当前分组下的节点行为。"

    def plot_how_to_read(meta: dict[str, str]) -> str:
        if meta["kind"] == "scatter":
            return "每个点代表一个节点；横轴是 log1p(子树规模)，纵轴是平均有效 hint 深度，颜色是本层继续坍塌比例，点越大表示该节点样本越多。"
        if meta["kind"] == "bins":
            return "上半图看平均有效 hint 深度，下半图看需要 hint / 需要 2+ hint / 需要 3+ hint / 最终未解 / 本层继续坍塌比例；横轴是子树规模分位箱。"
        if meta["kind"] == "heatmap":
            return "每一行是一个 task-depth，每一列是一个指标；颜色和格内数字都是相关系数。越接近 1 表示子树越大越难，越接近 -1 表示越大反而越容易。"
        if meta["kind"] == "outliers":
            return "横轴是 log1p(子树规模)，纵轴是相对线性趋势的残差。残差越高表示该节点比 subtree size 预测得更难，越低表示比预测得更容易。"
        return "阅读方式：先看整体趋势，再看被标注出来的异常节点。"

    def section_guide(kind: str) -> str:
        section_questions = {
            "scatter": "这一类图用来直接看节点云的原始分布，判断“大子树”和“高 hint 深度”是否会同时出现。",
            "bins": "这一类图用来判断，当我们把相近 subtree size 的节点放在一起时，hint 强度是否还会稳定上升或下降。",
            "heatmap": "这一类图用来总览所有 task-depth 组合和所有指标，快速找出最强正相关和最强负相关。",
            "outliers": "这一类图专门寻找“树不算大但特别难”或“树很大却意外容易”的反常节点。",
        }
        section_how = {
            "scatter": "如果点整体往右上角堆，说明子树越大越可能伴随更高的 hint 深度；如果只是右侧颜色变深，说明更像是第一跳没中后继续坍塌。",
            "bins": "优先看上半图是否随 bin 明显抬升，再看下半图里到底是哪一种 rate 被拉高，这样能分清是整体变难还是某个阶段特别容易崩。",
            "heatmap": "先看颜色深浅，再看格子里的数字；正值越大表示大子树更难，负值越大表示大子树反而更容易。",
            "outliers": "纵轴越高越值得排查，因为这些点比 subtree size 所预测的更难；纵轴很低的点则是“树大但不难”的反例。",
        }
        return f"""
        <div class="chart-guide">
          <div>
            <h3>这类图在回答什么</h3>
            <p>{html.escape(section_questions.get(kind, "这类图用于展示当前分组下的总体关系。"))}</p>
          </div>
          <div>
            <h3>怎么看这些图</h3>
            <p>{html.escape(section_how.get(kind, "先看总体趋势，再看标注节点。"))}</p>
          </div>
        </div>
        """

    def filter_buttons(options: list[tuple[str, str]], group: str) -> str:
        buttons = []
        for index, (value, label) in enumerate(options):
            pressed = "true" if index == 0 else "false"
            active_class = " is-active" if index == 0 else ""
            buttons.append(
                f"""
                <button
                  type="button"
                  class="filter-button{active_class}"
                  data-filter-group="{group}"
                  data-filter-value="{html.escape(value)}"
                  aria-pressed="{pressed}"
                >{html.escape(label)}</button>
                """
            )
        return "".join(buttons)

    def artifact_links() -> str:
        artifacts = summary.get("artifacts", {})
        if not artifacts:
            return '<p class="empty-state">暂无附属产物链接。</p>'
        ordered = [
            ("node_stats_path", "节点级统计表"),
            ("bin_summary_path", "分箱统计表"),
            ("correlations_path", "相关性表"),
            ("outliers_path", "异常点表"),
            ("html_report_path", "当前 HTML 报告"),
        ]
        cards: list[str] = []
        for key, label in ordered:
            target = artifacts.get(key)
            if not target:
                continue
            href = Path(str(target)).name
            cards.append(
                f"""
                <a class="artifact-link" href="{html.escape(href)}">
                  <span>{html.escape(label)}</span>
                  <strong>{html.escape(href)}</strong>
                </a>
                """
            )
        return "".join(cards) if cards else '<p class="empty-state">暂无附属产物链接。</p>'

    def glossary_cards() -> str:
        entries = [
            ("subtree_size", "这个 prefix 下面挂着多少个完整 SID leaf。数值越大，说明这个节点下面的候选空间越宽。"),
            (
                "mean_effective_hint_depth",
                "平均需要多少层 hint 才能把这个节点相关的样本带到正确路径上。数值越大，整体越难。",
            ),
            ("need_hint_rate", "从 base 直接没猜中的比例，也可以理解成至少需要 1 层 hint 的比例。"),
            ("hint2_plus_rate", "至少需要 2 层 hint 的比例。它往往更能反映“第一跳没中后，后面会不会继续崩”。"),
            ("hint3_plus_rate", "至少需要 3 层 hint 的比例，已经是比较深的持续坍塌信号。"),
            ("unsolved_rate", "最终即使给到最深 hint 仍然没解出来的比例。"),
            ("stage_local_rate", "到了这一层之后，是否还会继续向下一层 hint 滚落的条件概率。"),
        ]
        cards = []
        for term, text in entries:
            cards.append(
                f"""
                <article class="glossary-card">
                  <h3>{html.escape(term)}</h3>
                  <p>{html.escape(text)}</p>
                </article>
                """
            )
        return "".join(cards)

    def gallery_section(kind: str, paths: list[Path]) -> str:
        if not paths:
            return ""
        cards = []
        for path in paths:
            meta = extract_plot_meta(kind, path)
            cards.append(
                f"""
                <figure
                  class="plot-card"
                  data-task="{html.escape(meta["task"])}"
                  data-depth="{html.escape(meta["depth"])}"
                  data-kind="{html.escape(meta["kind"])}"
                >
                  <div class="plot-meta">
                    <span class="plot-chip">{html.escape(meta["kind_display"])}</span>
                    <span class="plot-chip plot-chip-soft">{html.escape(meta["task_display"])}</span>
                    <span class="plot-chip plot-chip-soft">{html.escape(meta["depth_display"])}</span>
                  </div>
                  <a class="plot-link" href="{html.escape(meta["rel"])}" target="_blank" rel="noreferrer">
                    <img src="{html.escape(meta["rel"])}" alt="{html.escape(meta["name"])}" loading="lazy">
                  </a>
                  <figcaption>
                    <strong>{html.escape(meta["subtitle"])}</strong>
                    <span>{html.escape(plot_question(meta))}</span>
                    <span>{html.escape(plot_how_to_read(meta))}</span>
                    <span class="plot-file">{html.escape(meta["name"])}</span>
                  </figcaption>
                </figure>
                """
            )
        return f"""
        <section class="section-block gallery-section" data-gallery-kind="{html.escape(kind)}">
          <div class="section-head">
            <span class="eyebrow">{html.escape(kind_kicker_map.get(kind, "图册"))}</span>
            <h2>{html.escape(kind_label_map.get(kind, kind))}</h2>
            <p>点击图片可以直接打开原图；上方筛选条会在整页范围内同步过滤这些图卡。</p>
          </div>
          {section_guide(kind)}
          <div class="plot-grid">
            {"".join(cards)}
          </div>
        </section>
        """

    total_plot_count = sum(len(paths) for paths in plot_paths.values())
    strongest_positive = summary.get("overall", {}).get("strongest_positive_correlation")
    strongest_negative = summary.get("overall", {}).get("strongest_negative_correlation")
    raw_json = html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
    task_filter_buttons = filter_buttons(
        [
            ("all", "全部任务"),
            ("sid", "sid"),
            ("hisTitle2sid", "hisTitle2sid"),
            ("title_desc2sid", "title_desc2sid"),
        ],
        "task",
    )
    depth_filter_buttons = filter_buttons(
        [("all", "全部层级"), ("1", "第 1 层"), ("2", "第 2 层"), ("3", "第 3 层"), ("4", "第 4 层")],
        "depth",
    )
    kind_filter_buttons = filter_buttons(
        [("all", "全部图类"), ("scatter", "散点"), ("bins", "分箱"), ("heatmap", "热图"), ("outliers", "异常点")],
        "kind",
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>提示强度 vs 子树规模</title>
  <style>
    :root {{
      color-scheme: light;
      --color-bg: #f6f1e8;
      --color-surface: rgba(255, 251, 244, 0.92);
      --color-surface-strong: #fffdf8;
      --color-surface-muted: rgba(244, 236, 223, 0.9);
      --color-text: #152a31;
      --color-text-muted: #617076;
      --color-border: rgba(36, 59, 63, 0.14);
      --color-border-strong: rgba(36, 59, 63, 0.24);
      --color-primary: #bd5a2f;
      --color-primary-strong: #7a3420;
      --color-primary-soft: rgba(189, 90, 47, 0.12);
      --color-secondary-soft: rgba(118, 166, 175, 0.16);
      --space-1: 4px;
      --space-2: 8px;
      --space-3: 12px;
      --space-4: 16px;
      --space-5: 20px;
      --space-6: 24px;
      --space-7: 28px;
      --space-8: 32px;
      --space-10: 40px;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 18px;
      --radius-xl: 24px;
      --shadow-sm: 0 8px 18px rgba(62, 48, 29, 0.06);
      --shadow-md: 0 16px 36px rgba(62, 48, 29, 0.08);
      --shadow-lg: 0 22px 60px rgba(62, 48, 29, 0.10);
      --duration-fast: 150ms;
      --duration-base: 220ms;
      --ease-standard: ease-in-out;
      --font-display: "Songti SC", "STSong", "Noto Serif CJK SC", serif;
      --font-body: "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      --font-size-body: clamp(0.98rem, 0.94rem + 0.16vw, 1.04rem);
      --font-size-h2: clamp(1.45rem, 1.2rem + 0.8vw, 2.2rem);
      --font-size-h1: clamp(2.3rem, 1.9rem + 2vw, 4.4rem);
    }}
    * {{
      box-sizing: border-box;
    }}
    html {{
      scroll-behavior: smooth;
    }}
    body {{
      margin: 0;
      font-family: var(--font-body);
      font-size: var(--font-size-body);
      color: var(--color-text);
      background:
        radial-gradient(circle at top left, rgba(255, 210, 138, 0.55), transparent 32%),
        radial-gradient(circle at top right, rgba(156, 210, 217, 0.42), transparent 28%),
        linear-gradient(180deg, #faf6ef 0%, #eee4d3 100%);
    }}
    a {{
      color: inherit;
    }}
    button {{
      font: inherit;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: var(--space-7) var(--space-5) 72px;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-family: var(--font-display);
      letter-spacing: -0.03em;
    }}
    h1 {{
      font-size: var(--font-size-h1);
      line-height: 0.94;
    }}
    h2 {{
      font-size: var(--font-size-h2);
      line-height: 1.08;
    }}
    p {{
      margin: 0;
      line-height: 1.7;
      color: var(--color-text-muted);
    }}
    [hidden] {{
      display: none !important;
    }}
    .hero,
    .task-strip,
    .filter-panel,
    .section-block,
    .json-section {{
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow-lg);
    }}
    .hero {{
      position: relative;
      overflow: hidden;
      padding: var(--space-8);
      margin-bottom: var(--space-6);
      background:
        linear-gradient(145deg, rgba(255, 253, 248, 0.96), rgba(247, 238, 224, 0.88));
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: auto -60px -110px auto;
      width: 280px;
      height: 280px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(189, 90, 47, 0.14), transparent 70%);
      pointer-events: none;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.65fr) minmax(280px, 0.95fr);
      gap: var(--space-6);
      align-items: start;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: var(--space-2);
      color: var(--color-primary-strong);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.74rem;
      margin-bottom: var(--space-4);
    }}
    .eyebrow::before {{
      content: "";
      width: 24px;
      height: 1px;
      background: currentColor;
    }}
    .hero-copy p {{
      max-width: 64ch;
      margin-top: var(--space-5);
    }}
    .hero-rail {{
      display: grid;
      gap: var(--space-4);
    }}
    .rail-card {{
      background: linear-gradient(180deg, rgba(255, 251, 244, 0.92), rgba(244, 235, 219, 0.90));
      border: 1px solid var(--color-border);
      border-radius: var(--radius-lg);
      padding: var(--space-5);
      box-shadow: var(--shadow-sm);
    }}
    .rail-card p + p {{
      margin-top: var(--space-3);
    }}
    .stat-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: var(--space-3);
      margin-top: var(--space-6);
    }}
    .stat-card {{
      padding: var(--space-4);
      border-radius: var(--radius-lg);
      border: 1px solid rgba(58, 49, 36, 0.08);
      background: linear-gradient(180deg, rgba(255, 251, 244, 0.94), rgba(248, 239, 226, 0.90));
    }}
    .tone-warm {{
      background: linear-gradient(180deg, rgba(250, 224, 201, 0.95), rgba(255, 246, 236, 0.98));
    }}
    .tone-cool {{
      background: linear-gradient(180deg, rgba(218, 237, 240, 0.94), rgba(246, 250, 252, 0.98));
    }}
    .stat-label {{
      display: block;
      margin-bottom: var(--space-2);
      color: var(--color-text-muted);
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .stat-value {{
      display: block;
      font-size: 1.18rem;
      line-height: 1.3;
      color: var(--color-text);
    }}
    .quick-links {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-3);
      margin: var(--space-5) 0 var(--space-6);
    }}
    .quick-links a {{
      text-decoration: none;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 252, 246, 0.88);
      border: 1px solid var(--color-border);
      box-shadow: var(--shadow-sm);
      transition: transform var(--duration-fast) var(--ease-standard), border-color var(--duration-fast) var(--ease-standard);
    }}
    .quick-links a:hover,
    .quick-links a:focus-visible {{
      transform: translateY(-1px);
      border-color: var(--color-border-strong);
      outline: none;
    }}
    .task-strip,
    .filter-panel,
    .section-block,
    .json-section {{
      padding: var(--space-6);
      margin-bottom: var(--space-6);
    }}
    .section-head {{
      display: flex;
      flex-direction: column;
      gap: var(--space-3);
      margin-bottom: var(--space-5);
    }}
    .section-head p {{
      max-width: 74ch;
    }}
    .chart-guide {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-4);
      margin-bottom: var(--space-5);
    }}
    .chart-guide > div {{
      padding: var(--space-4);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(255, 253, 248, 0.94), rgba(244, 237, 225, 0.92));
      border: 1px solid var(--color-border);
    }}
    .chart-guide h3 {{
      font-size: 1.02rem;
      margin-bottom: var(--space-2);
    }}
    .task-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: var(--space-4);
      margin-top: var(--space-4);
    }}
    .glossary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: var(--space-4);
      margin-top: var(--space-4);
    }}
    .glossary-card {{
      padding: var(--space-5);
      border-radius: var(--radius-lg);
      border: 1px solid var(--color-border);
      background: linear-gradient(180deg, rgba(255, 254, 250, 0.94), rgba(243, 238, 229, 0.96));
      box-shadow: var(--shadow-sm);
    }}
    .glossary-card h3 {{
      font-size: 1rem;
      margin-bottom: var(--space-2);
    }}
    .task-card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(251,245,234,0.92));
      border: 1px solid var(--color-border);
      border-radius: var(--radius-lg);
      padding: var(--space-5);
      box-shadow: var(--shadow-sm);
    }}
    .task-card-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: var(--space-3);
      margin-bottom: var(--space-4);
    }}
    .task-card-head span {{
      color: var(--color-text-muted);
      font-size: 0.9rem;
    }}
    .task-metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-3);
      margin: 0;
    }}
    .task-metrics div {{
      background: rgba(250, 244, 234, 0.95);
      border-radius: var(--radius-md);
      padding: 10px 12px;
    }}
    .task-metrics dt {{
      margin-bottom: var(--space-2);
      color: var(--color-text-muted);
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .task-metrics dd {{
      margin: 0;
      color: var(--color-text);
      font-size: 1.15rem;
      font-weight: 600;
    }}
    .filter-panel {{
      background:
        linear-gradient(180deg, rgba(255, 252, 247, 0.96), rgba(247, 239, 226, 0.94));
    }}
    .filter-grid {{
      display: grid;
      grid-template-columns: 1.2fr minmax(0, 1fr);
      gap: var(--space-5);
      align-items: start;
    }}
    .filter-stack {{
      display: grid;
      gap: var(--space-4);
    }}
    .filter-row {{
      display: grid;
      gap: var(--space-3);
    }}
    .filter-label {{
      font-size: 0.84rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      color: var(--color-text-muted);
      text-transform: uppercase;
    }}
    .filter-group {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
    }}
    .filter-button {{
      min-height: 44px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--color-border);
      background: rgba(255, 253, 248, 0.92);
      color: var(--color-text);
      cursor: pointer;
      transition:
        transform var(--duration-fast) var(--ease-standard),
        background var(--duration-fast) var(--ease-standard),
        border-color var(--duration-fast) var(--ease-standard),
        box-shadow var(--duration-fast) var(--ease-standard);
    }}
    .filter-button:hover,
    .filter-button:focus-visible {{
      outline: none;
      transform: translateY(-1px);
      border-color: var(--color-border-strong);
      box-shadow: 0 0 0 3px rgba(189, 90, 47, 0.14);
    }}
    .filter-button.is-active,
    .filter-button[aria-pressed="true"] {{
      background: var(--color-primary);
      color: #fffaf3;
      border-color: rgba(122, 52, 32, 0.45);
      box-shadow: var(--shadow-sm);
    }}
    .filter-status {{
      display: grid;
      gap: var(--space-3);
      padding: var(--space-4);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(220, 238, 241, 0.94), rgba(245, 251, 252, 0.98));
      border: 1px solid rgba(90, 130, 136, 0.18);
    }}
    .filter-status strong {{
      font-size: 1.2rem;
      color: var(--color-text);
    }}
    .artifact-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: var(--space-3);
      margin-top: var(--space-4);
    }}
    .artifact-link {{
      display: grid;
      gap: var(--space-2);
      padding: var(--space-4);
      text-decoration: none;
      border-radius: var(--radius-lg);
      background: rgba(255, 254, 250, 0.92);
      border: 1px solid var(--color-border);
      box-shadow: var(--shadow-sm);
      transition: transform var(--duration-fast) var(--ease-standard), border-color var(--duration-fast) var(--ease-standard);
    }}
    .artifact-link:hover,
    .artifact-link:focus-visible {{
      outline: none;
      transform: translateY(-1px);
      border-color: var(--color-border-strong);
    }}
    .artifact-link span {{
      color: var(--color-text-muted);
      font-size: 0.9rem;
    }}
    .artifact-link strong {{
      color: var(--color-text);
      word-break: break-all;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: var(--space-4);
    }}
    .plot-card {{
      display: grid;
      gap: var(--space-3);
      margin: 0;
      padding: var(--space-4);
      border-radius: var(--radius-lg);
      border: 1px solid var(--color-border);
      background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(250,245,236,0.96));
      box-shadow: var(--shadow-md);
      transition:
        transform var(--duration-fast) var(--ease-standard),
        box-shadow var(--duration-fast) var(--ease-standard),
        border-color var(--duration-fast) var(--ease-standard);
    }}
    .plot-card:hover {{
      transform: translateY(-2px);
      border-color: var(--color-border-strong);
      box-shadow: 0 18px 36px rgba(56, 44, 28, 0.12);
    }}
    .plot-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
    }}
    .plot-chip {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--color-primary-soft);
      color: var(--color-primary-strong);
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .plot-chip-soft {{
      background: rgba(21, 42, 49, 0.06);
      color: var(--color-text-muted);
    }}
    .plot-link {{
      display: block;
      border-radius: var(--radius-md);
    }}
    .plot-link:focus-visible {{
      outline: none;
      box-shadow: 0 0 0 3px rgba(189, 90, 47, 0.16);
    }}
    .plot-card img {{
      width: 100%;
      display: block;
      border-radius: var(--radius-md);
      background: white;
      border: 1px solid rgba(47, 69, 74, 0.08);
    }}
    .plot-card figcaption {{
      display: grid;
      gap: var(--space-1);
      font-size: 0.88rem;
      color: var(--color-text-muted);
      word-break: break-word;
    }}
    .plot-card figcaption strong {{
      color: var(--color-text);
      font-size: 1rem;
    }}
    .plot-file {{
      font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 0.78rem;
      color: rgba(21, 42, 49, 0.62);
    }}
    .empty-state {{
      color: var(--color-text-muted);
      font-style: italic;
    }}
    .filter-empty {{
      padding: var(--space-6);
      border-radius: var(--radius-lg);
      border: 1px dashed rgba(122, 52, 32, 0.28);
      background: rgba(255, 251, 244, 0.88);
      color: var(--color-text-muted);
      text-align: center;
    }}
    pre {{
      overflow-x: auto;
      white-space: pre-wrap;
      margin: var(--space-4) 0 0;
      padding: var(--space-4);
      border-radius: var(--radius-lg);
      border: 1px solid rgba(47, 69, 74, 0.10);
      background: #fbf8f2;
      line-height: 1.55;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--color-text);
    }}
    summary:focus-visible {{
      outline: none;
      box-shadow: 0 0 0 3px rgba(189, 90, 47, 0.16);
      border-radius: var(--radius-sm);
    }}
    @media (max-width: 980px) {{
      .hero-grid,
      .filter-grid,
      .chart-guide {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      main {{
        padding: var(--space-5) var(--space-4) 64px;
      }}
      .hero,
      .task-strip,
      .filter-panel,
      .section-block,
      .json-section {{
        padding: var(--space-5);
      }}
      .task-card-head {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .task-metrics {{
        grid-template-columns: 1fr;
      }}
      .plot-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <span class="eyebrow">本地 Bundle 报告</span>
          <h1>提示强度 vs 子树规模</h1>
          <p>
            这不是一份“把图堆上去”的导出页，而是一个专门为了读懂 hint 行为而整理出来的静态前端页面。
            它要回答的是：某个 SID 节点更难，到底是因为它下面的子树更大，还是因为前缀本身已经变得歧义。
          </p>
          <div class="stat-row">
            {summary_card("样本数", format_number(summary.get("num_samples")), "warm")}
            {summary_card("图表总数", format_number(total_plot_count), "cool")}
            {summary_card("最强正相关", format_corr(strongest_positive))}
            {summary_card("最强负相关", format_corr(strongest_negative))}
          </div>
        </div>
        <aside class="hero-rail">
          <section class="rail-card">
            <span class="eyebrow">关键结论</span>
            <p>root 层的大子树，经常会把“第一次没猜中之后继续往下要 hint”的风险抬高。</p>
            <p>但更深层的崩溃，往往已经不是 subtree size 能单独解释的，而更像 prefix 歧义或局部路径病灶。</p>
          </section>
          <section class="rail-card">
            <span class="eyebrow">阅读方式</span>
            <p>散点图先看原始节点云，分箱图再看平均趋势，热图用来快速定位 strongest signal，异常点图专门找“树大之外”的反常节点。</p>
          </section>
        </aside>
      </div>
    </section>

    <nav class="quick-links">
      <a href="#task-snapshot">任务概览</a>
      <a href="#metric-glossary">指标词典</a>
      <a href="#plot-filters">图册筛选</a>
      <a href="#scatter-panels">散点图册</a>
      <a href="#binned-trends">分箱趋势</a>
      <a href="#heatmaps">相关性热图</a>
      <a href="#outliers">异常点图册</a>
      <a href="#raw-json">原始摘要 JSON</a>
    </nav>

    <section class="task-strip" id="task-snapshot">
      <div class="section-head">
        <span class="eyebrow">任务概览</span>
        <h2>各任务的基线难度</h2>
        <p>先看 task 级别的底色，再去读节点级图表，能避免把 task 本身的难度误判成 subtree effect。</p>
      </div>
      <div class="task-grid">
        {task_cards()}
      </div>
    </section>

    <section class="task-strip" id="metric-glossary">
      <div class="section-head">
        <span class="eyebrow">指标词典</span>
        <h2>先把几个关键指标说清楚</h2>
        <p>如果不先统一这些词的含义，后面的图很容易越看越混。这里保留的都是你在图里会反复看到的核心指标。</p>
      </div>
      <div class="glossary-grid">
        {glossary_cards()}
      </div>
    </section>

    <section class="filter-panel" id="plot-filters">
      <div class="section-head">
        <span class="eyebrow">图册筛选</span>
        <h2>按任务、层级和图类浏览</h2>
        <p>这是一个静态网页，但仍然提供了前端筛选交互。你可以只看某个 task、某一层，或者只看异常点图。</p>
      </div>
      <div class="filter-grid">
        <div class="filter-stack">
          <div class="filter-row">
            <span class="filter-label">按任务筛选</span>
            <div class="filter-group" role="toolbar" aria-label="按任务筛选">
              {task_filter_buttons}
            </div>
          </div>
          <div class="filter-row">
            <span class="filter-label">按层级筛选</span>
            <div class="filter-group" role="toolbar" aria-label="按层级筛选">
              {depth_filter_buttons}
            </div>
          </div>
          <div class="filter-row">
            <span class="filter-label">按图类筛选</span>
            <div class="filter-group" role="toolbar" aria-label="按图类筛选">
              {kind_filter_buttons}
            </div>
          </div>
        </div>
        <aside class="filter-status">
          <span class="filter-label">当前状态</span>
          <strong id="plot-counter">当前显示 {total_plot_count} / {total_plot_count} 张图</strong>
          <p>如果筛选结果为空，页面会直接显示空状态提示，而不是把整个布局弄乱。</p>
        </aside>
      </div>
      <div class="artifact-grid">
        {artifact_links()}
      </div>
    </section>

    <div id="scatter-panels">
      {gallery_section("scatter", plot_paths.get("scatter", []))}
    </div>
    <div id="binned-trends">
      {gallery_section("bins", plot_paths.get("bins", []))}
    </div>
    <div id="heatmaps">
      {gallery_section("heatmap", plot_paths.get("heatmap", []))}
    </div>
    <div id="outliers">
      {gallery_section("outliers", plot_paths.get("outliers", []))}
    </div>

    <div id="filter-empty-state" class="filter-empty" hidden>
      当前筛选条件下没有可显示的图表。你可以切回“全部任务”或“全部层级”。
    </div>

    <section class="json-section" id="raw-json">
      <div class="section-head">
        <span class="eyebrow">原始摘要</span>
        <h2>Summary JSON</h2>
        <p>机器可读的完整摘要仍然保留在这里，方便你对照 CSV 和图表继续深挖。</p>
      </div>
      <details>
        <summary>展开原始 summary JSON</summary>
        <pre>{raw_json}</pre>
      </details>
    </section>
  </main>
  <script>
    (() => {{
      const state = {{ task: "all", depth: "all", kind: "all" }};
      const buttons = Array.from(document.querySelectorAll(".filter-button"));
      const cards = Array.from(document.querySelectorAll(".plot-card"));
      const sections = Array.from(document.querySelectorAll(".gallery-section"));
      const emptyState = document.getElementById("filter-empty-state");
      const counter = document.getElementById("plot-counter");

      function matches(card) {{
        const taskOk = state.task === "all" || card.dataset.task === state.task;
        const depthOk = state.depth === "all" || card.dataset.depth === state.depth;
        const kindOk = state.kind === "all" || card.dataset.kind === state.kind;
        return taskOk && depthOk && kindOk;
      }}

      function applyFilters() {{
        let visible = 0;
        sections.forEach((section) => {{
          let sectionVisible = 0;
          section.querySelectorAll(".plot-card").forEach((card) => {{
            const keep = matches(card);
            card.hidden = !keep;
            if (keep) {{
              sectionVisible += 1;
              visible += 1;
            }}
          }});
          section.hidden = sectionVisible === 0;
        }});
        if (counter) {{
          counter.textContent = `当前显示 ${{visible}} / {total_plot_count} 张图`;
        }}
        if (emptyState) {{
          emptyState.hidden = visible !== 0;
        }}
      }}

      buttons.forEach((button) => {{
        button.addEventListener("click", () => {{
          const group = button.dataset.filterGroup;
          const value = button.dataset.filterValue;
          if (!group || !value) return;
          state[group] = value;
          buttons
            .filter((candidate) => candidate.dataset.filterGroup === group)
            .forEach((candidate) => {{
              const isActive = candidate === button;
              candidate.classList.toggle("is-active", isActive);
              candidate.setAttribute("aria-pressed", isActive ? "true" : "false");
            }});
          applyFilters();
        }});
      }});

      applyFilters();
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else bundle_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "hint_subtree_relation_plots"

    enriched = load_enriched_frame(bundle_root)
    node_stats = build_node_stats(enriched)
    bin_summary = build_bin_summary(node_stats)
    correlations = build_correlation_summary(node_stats)
    outliers = build_outlier_rows(node_stats)
    plot_paths = build_plot_paths(
        node_stats=node_stats,
        bin_summary=bin_summary,
        correlations=correlations,
        outliers=outliers,
        plot_dir=plot_dir,
    )

    node_stats_path = output_dir / "hint_subtree_relation_node_stats.csv"
    bin_summary_path = output_dir / "hint_subtree_relation_bin_summary.csv"
    outliers_path = output_dir / "hint_subtree_relation_outliers.csv"
    correlations_path = output_dir / "hint_subtree_relation_correlations.csv"
    summary_path = output_dir / "hint_subtree_relation_summary.json"
    html_path = output_dir / "hint_subtree_relation_report.html"

    node_stats.to_csv(node_stats_path, index=False)
    bin_summary.to_csv(bin_summary_path, index=False)
    outliers.to_csv(outliers_path, index=False)
    correlations.to_csv(correlations_path, index=False)

    summary = build_summary(
        enriched=enriched,
        node_stats=node_stats,
        bin_summary=bin_summary,
        correlations=correlations,
        outliers=outliers,
        plot_paths=plot_paths,
        output_dir=output_dir,
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    html = build_report_html(summary=summary, plot_paths=plot_paths)
    html_path.write_text(html, encoding="utf-8")

    print(f"Saved node stats to {node_stats_path}")
    print(f"Saved bin summary to {bin_summary_path}")
    print(f"Saved outliers to {outliers_path}")
    print(f"Saved correlations to {correlations_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved HTML report to {html_path}")


if __name__ == "__main__":
    main()
