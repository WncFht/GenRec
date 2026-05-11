#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "docs" / "research" / "assets" / "lc4023-fixed"
HISTORICAL_DISTRIBUTION_CSV = (
    REPO_ROOT / "docs" / "deepresearch" / "genrec_rl_study_2026-03-28" / "data" / "fixed_hint_bug_depth_distribution.csv"
)
CURRENT_HINT_MAP_PATH = (
    REPO_ROOT / "temp" / "rl_beam_hint" / "instruments_grec_lc4023_fixed_20260508_022709_beam16_hint_map.json"
)

TABLEAU_10 = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "red": "#E15759",
    "teal": "#76B7B2",
    "green": "#59A14F",
    "yellow": "#EDC948",
    "purple": "#B07AA1",
    "pink": "#FF9DA7",
    "brown": "#9C755F",
    "gray": "#BAB0AB",
}

DEPTH_COLUMNS = [
    ("depth_0_ratio", "depth-0", TABLEAU_10["blue"]),
    ("depth_1_ratio", "depth-1", TABLEAU_10["green"]),
    ("depth_2_ratio", "depth-2", TABLEAU_10["orange"]),
    ("depth_3_ratio", "depth-3", TABLEAU_10["red"]),
]

SCOPE_ORDER = ["overall", "sid", "hisTitle2sid", "title_desc2sid"]
SCOPE_LABELS = {
    "overall": "Overall",
    "sid": "SID",
    "hisTitle2sid": "HisTitle2SID",
    "title_desc2sid": "TitleDesc2SID",
}
SCOPE_TO_TASK = {
    "sid": "task1_sid_sft",
    "hisTitle2sid": "task4_hisTitle2sid",
    "title_desc2sid": "task5_title_desc2sid",
}
VERSION_ORDER = ["correct_task_index", "legacy_index_only", "current_compact_index"]
VERSION_LABELS = {
    "correct_task_index": "historical\ntask+index",
    "legacy_index_only": "historical\nlegacy-index",
    "current_compact_index": "current\ncompact-index",
}


def _scope_depth_row(scope: str, version: str, counts: Counter[int], total: int, source_path: str) -> dict[str, object]:
    row: dict[str, object] = {
        "scope": scope,
        "version": version,
        "version_label": VERSION_LABELS[version],
        "total": total,
        "avg_depth": sum(depth * counts.get(depth, 0) for depth in range(4)) / total if total else 0.0,
        "summary_path": source_path,
    }
    for depth in range(4):
        count = int(counts.get(depth, 0))
        row[f"depth_{depth}_count"] = count
        row[f"depth_{depth}_ratio"] = count / total if total else 0.0
    return row


def build_current_rows() -> list[dict[str, object]]:
    payload = json.loads(CURRENT_HINT_MAP_PATH.read_text(encoding="utf-8"))
    hint_depth_by_sample_key = payload["hint_depth_by_sample_key"]

    overall_counts: Counter[int] = Counter()
    task_depth_counts: dict[str, Counter[int]] = {task: Counter() for task in SCOPE_TO_TASK.values()}

    for sample_key, depth in hint_depth_by_sample_key.items():
        task_name, _, _ = sample_key.partition("::")
        depth_int = int(depth)
        overall_counts[depth_int] += 1
        if task_name in task_depth_counts:
            task_depth_counts[task_name][depth_int] += 1

    rows = [
        _scope_depth_row(
            scope="overall",
            version="current_compact_index",
            counts=overall_counts,
            total=sum(overall_counts.values()),
            source_path=str(CURRENT_HINT_MAP_PATH),
        )
    ]
    for scope, task_name in SCOPE_TO_TASK.items():
        counts = task_depth_counts[task_name]
        rows.append(
            _scope_depth_row(
                scope=scope,
                version="current_compact_index",
                counts=counts,
                total=sum(counts.values()),
                source_path=str(CURRENT_HINT_MAP_PATH),
            )
        )
    return rows


def build_plot_dataframe() -> pd.DataFrame:
    historical_df = pd.read_csv(HISTORICAL_DISTRIBUTION_CSV)
    historical_df["version_label"] = historical_df["version"].map(VERSION_LABELS)
    historical_df["avg_depth"] = (
        historical_df["depth_1_ratio"] + 2 * historical_df["depth_2_ratio"] + 3 * historical_df["depth_3_ratio"]
    )
    historical_df["summary_path"] = ""

    current_df = pd.DataFrame(build_current_rows())
    plot_df = pd.concat([historical_df, current_df], ignore_index=True, sort=False)
    plot_df["version_order"] = plot_df["version"].map({name: idx for idx, name in enumerate(VERSION_ORDER)})
    plot_df["scope_order"] = plot_df["scope"].map({name: idx for idx, name in enumerate(SCOPE_ORDER)})
    plot_df = plot_df.sort_values(["scope_order", "version_order"]).reset_index(drop=True)
    return plot_df


def plot_distribution_grid(plot_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), sharey=True)
    bar_positions = list(range(len(VERSION_ORDER)))

    for ax, scope in zip(axes.flat, SCOPE_ORDER, strict=True):
        scope_df = plot_df[plot_df["scope"] == scope].set_index("version").loc[VERSION_ORDER]
        bottoms = [0.0] * len(VERSION_ORDER)
        for column, label, color in DEPTH_COLUMNS:
            values = (scope_df[column].astype(float) * 100.0).tolist()
            ax.bar(
                bar_positions,
                values,
                bottom=bottoms,
                width=0.72,
                color=color,
                label=label,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]

        ax.set_title(SCOPE_LABELS[scope], pad=16, fontsize=16)
        ax.set_xticks(bar_positions, [VERSION_LABELS[name] for name in VERSION_ORDER])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", labelsize=9)

        for idx, version in enumerate(VERSION_ORDER):
            total = int(scope_df.loc[version, "total"])
            avg_depth = float(scope_df.loc[version, "avg_depth"])
            ax.text(
                bar_positions[idx],
                101.1,
                f"n={total:,}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#6B7280",
                clip_on=False,
            )
            ax.text(
                bar_positions[idx],
                91.8,
                f"avg={avg_depth:.3f}",
                ha="center",
                va="top",
                fontsize=8.5,
                color="#374151",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
            )

    axes[0, 0].set_ylabel("Share of samples (%)")
    axes[1, 0].set_ylabel("Share of samples (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False)
    fig.suptitle("LC4023 Fixed-Hint Depth Distribution\nHistorical vs Current Compact Index", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = build_plot_dataframe()
    plot_df.to_csv(ASSET_DIR / "lc4023_fixed_hint_depth_distribution_comparison.csv", index=False)
    plot_distribution_grid(plot_df, ASSET_DIR / "lc4023_fixed_hint_depth_distribution_comparison.png")


if __name__ == "__main__":
    main()
