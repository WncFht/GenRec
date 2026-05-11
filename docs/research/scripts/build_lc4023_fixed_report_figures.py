#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
ASSET_DIR = REPO_ROOT / "docs" / "research" / "assets" / "lc4023-fixed"
SFT_PATH = RESULTS_ROOT / "Instruments-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu" / "checkpoint-4023" / "metrics.json"

SERIES = [
    {
        "key": "fixed",
        "label": "RL fixed-hint",
        "root": RESULTS_ROOT / "Instruments-grec-lc4023-fixed",
        "max_step": 5006,
        "color": "#F28E2B",
        "marker": "o",
    },
    {
        "key": "rule",
        "label": "RL rule-only",
        "root": RESULTS_ROOT / "ins-lc4023-rule",
        "max_step": 5006,
        "color": "#9C755F",
        "marker": "s",
    },
    {
        "key": "fixed_ce",
        "label": "RL fixed-hint + CE",
        "root": RESULTS_ROOT / "Instruments-grec-lc4023-fixed-ce",
        "max_step": 5006,
        "color": "#4E79A7",
        "marker": "^",
    },
]

METRICS = [
    ("HR@1", "HR@1"),
    ("HR@5", "HR@5"),
    ("NDCG@5", "NDCG@5"),
    ("NDCG@10", "NDCG@10"),
    ("HR@10", "HR@10"),
    ("NDCG@50", "NDCG@50"),
    ("HR@50", "HR@50"),
]


def load_series_points(root: Path, max_step: int) -> dict[str, list[float]]:
    data = {"epoch": []}
    for metric, _ in METRICS:
        data[metric] = []

    metrics_files = sorted(root.glob("checkpoint-*/metrics.json"), key=lambda path: int(path.parent.name.split("-")[-1]))
    for path in metrics_files:
        step = int(path.parent.name.split("-")[-1])
        metrics = json.loads(path.read_text())
        data["epoch"].append(step / max_step * 2.0)
        for metric, _ in METRICS:
            data[metric].append(float(metrics[metric]))
    return data


def build_checkpoint_table() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for series in SERIES:
        metrics_files = sorted(
            series["root"].glob("checkpoint-*/metrics.json"),
            key=lambda path: int(path.parent.name.split("-")[-1]),
        )
        for path in metrics_files:
            step = int(path.parent.name.split("-")[-1])
            metrics = json.loads(path.read_text())
            row: dict[str, float | int | str] = {
                "variant": str(series["key"]),
                "variant_label": str(series["label"]),
                "step": step,
                "epoch": step / int(series["max_step"]) * 2.0,
            }
            for metric, _ in METRICS:
                row[metric] = float(metrics[metric])
            for metric in ["HR@1", "HR@5", "NDCG@5"]:
                row[metric] = float(metrics[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def load_sft_reference() -> dict[str, float]:
    metrics = json.loads(SFT_PATH.read_text())
    return {metric: float(metrics[metric]) for metric, _ in METRICS}


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sft = load_sft_reference()
    series_points = [(series, load_series_points(series["root"], int(series["max_step"]))) for series in SERIES]
    table_df = build_checkpoint_table()
    table_df.to_csv(ASSET_DIR / "lc4023_checkpoint_metrics.csv", index=False)

    fig, axes = plt.subplots(4, 2, figsize=(11.2, 13.6), sharex=True)
    axes_flat = list(axes.flat)
    for ax, (metric, title) in zip(axes_flat, METRICS):
        for series, points in series_points:
            ax.plot(
                points["epoch"],
                points[metric],
                label=series["label"],
                color=series["color"],
                marker=series["marker"],
                linewidth=2.2,
                markersize=5,
            )
        ax.axhline(sft[metric], linestyle="--", linewidth=1.4, color="#6B7280", label="SFT ckpt-4023")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_xlim(0.0, 2.0)
        ax.grid(alpha=0.22)

    unused_axes = axes_flat[len(METRICS):]
    for ax in unused_axes:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.suptitle("Instruments LC4023 RL: Fixed Hint vs Rule-only", y=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ASSET_DIR / "lc4023_fixed_vs_rule_epoch_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
