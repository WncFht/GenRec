#!/usr/bin/env python3
from __future__ import annotations

import json
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
CURRENT_SUMMARY_PATH = REPO_ROOT / "temp" / "rl_beam_hint" / "instruments_grec_lc4023_fixed_analysis_summary.json"

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


def build_current_overall_row() -> dict[str, object]:
    summary = json.loads(CURRENT_SUMMARY_PATH.read_text(encoding="utf-8"))
    beam_payload = summary["results"]["16"]
    stages = beam_payload["stages"]

    depth_counts = {
        0: int(stages["base"]["stage_rule_hit_sample_count"]),
        1: int(stages["hint_1"]["stage_rule_hit_sample_count"]),
        2: int(stages["hint_2"]["stage_rule_hit_sample_count"]),
        3: int(stages["hint_3"]["stage_rule_hit_sample_count"]) + int(beam_payload["cumulative"]["final_remaining_subset_size"]),
    }
    total = int(summary["num_samples"])

    row: dict[str, object] = {
        "scope": "overall",
        "version": "current_compact_index",
        "version_label": "current compact index",
        "total": total,
        "avg_depth": sum(depth * count for depth, count in depth_counts.items()) / total,
        "summary_path": str(CURRENT_SUMMARY_PATH),
    }
    for depth, count in depth_counts.items():
        row[f"depth_{depth}_count"] = count
        row[f"depth_{depth}_ratio"] = count / total
    return row


def build_plot_dataframe() -> pd.DataFrame:
    historical_df = pd.read_csv(HISTORICAL_DISTRIBUTION_CSV)
    overall_df = historical_df[historical_df["scope"] == "overall"].copy()
    overall_df["version_label"] = overall_df["version"].map(
        {
            "correct_task_index": "historical task+index",
            "legacy_index_only": "historical legacy index",
        }
    )
    overall_df["avg_depth"] = (
        overall_df["depth_1_ratio"] + 2 * overall_df["depth_2_ratio"] + 3 * overall_df["depth_3_ratio"]
    )
    overall_df["summary_path"] = ""

    current_df = pd.DataFrame([build_current_overall_row()])
    plot_df = pd.concat([overall_df, current_df], ignore_index=True, sort=False)
    plot_df["version_order"] = plot_df["version"].map(
        {
            "legacy_index_only": 0,
            "correct_task_index": 1,
            "current_compact_index": 2,
        }
    )
    plot_df = plot_df.sort_values("version_order").reset_index(drop=True)
    return plot_df


def plot_overall_distribution(plot_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    bar_positions = list(range(len(plot_df)))
    bottoms = [0.0] * len(plot_df)

    for column, label, color in DEPTH_COLUMNS:
        values = (plot_df[column].astype(float) * 100.0).tolist()
        ax.bar(
            bar_positions,
            values,
            bottom=bottoms,
            width=0.72,
            color=color,
            label=label,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]

    ax.set_xticks(bar_positions, plot_df["version_label"].tolist())
    ax.set_ylabel("Share of samples (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.22)
    ax.set_title("LC4023 Fixed-Hint Training Depth Distribution: Historical Figure 7 vs Current Compact Index")

    for idx, row in plot_df.iterrows():
        total = int(row["total"])
        avg_depth = float(row["avg_depth"])
        ax.text(idx, 101.0, f"n={total:,}", ha="center", va="bottom", fontsize=9)
        ax.text(
            idx,
            96.0,
            f"avg={avg_depth:.3f}",
            ha="center",
            va="top",
            fontsize=9,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
        )

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = build_plot_dataframe()
    plot_df.to_csv(ASSET_DIR / "lc4023_fixed_hint_depth_distribution_comparison.csv", index=False)
    plot_overall_distribution(plot_df, ASSET_DIR / "lc4023_fixed_hint_depth_distribution_comparison.png")


if __name__ == "__main__":
    main()
