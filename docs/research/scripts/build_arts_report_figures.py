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
ASSET_DIR = REPO_ROOT / "docs" / "research" / "assets" / "arts-report"

SERIES = {
    "label": "Arts SFT",
    "root": RESULTS_ROOT / "Arts-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu",
    "max_step": 17268,
    "num_train_epochs": 4.0,
    "color": "#4E79A7",
    "marker": "o",
}

METRICS = [
    ("NDCG@10", "NDCG@10"),
    ("HR@10", "HR@10"),
    ("NDCG@50", "NDCG@50"),
    ("HR@50", "HR@50"),
]


def load_series_points() -> dict[str, list[float]]:
    data = {"epoch": []}
    for metric, _ in METRICS:
        data[metric] = []

    metrics_files = sorted(SERIES["root"].glob("checkpoint-*/metrics.json"), key=lambda path: int(path.parent.name.split("-")[-1]))
    for path in metrics_files:
        step = int(path.parent.name.split("-")[-1])
        metrics = json.loads(path.read_text())
        data["epoch"].append(step / int(SERIES["max_step"]) * float(SERIES["num_train_epochs"]))
        for metric, _ in METRICS:
            data[metric].append(float(metrics[metric]))
    return data


def build_checkpoint_table() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics_files = sorted(SERIES["root"].glob("checkpoint-*/metrics.json"), key=lambda path: int(path.parent.name.split("-")[-1]))
    for path in metrics_files:
        step = int(path.parent.name.split("-")[-1])
        metrics = json.loads(path.read_text())
        row: dict[str, float | int | str] = {
            "variant": "arts_sft",
            "variant_label": str(SERIES["label"]),
            "step": step,
            "epoch": step / int(SERIES["max_step"]) * float(SERIES["num_train_epochs"]),
        }
        for metric, _ in METRICS:
            row[metric] = float(metrics[metric])
        for metric in ["HR@1", "HR@5", "NDCG@5"]:
            row[metric] = float(metrics[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    table_df = build_checkpoint_table()
    table_df.to_csv(ASSET_DIR / "arts_checkpoint_metrics.csv", index=False)
    points = load_series_points()

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), sharex=True)
    for ax, (metric, title) in zip(axes.flat, METRICS, strict=True):
        ax.plot(
            points["epoch"],
            points[metric],
            label=str(SERIES["label"]),
            color=str(SERIES["color"]),
            marker=str(SERIES["marker"]),
            linewidth=2.2,
            markersize=5,
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_xlim(0.0, float(SERIES["num_train_epochs"]))
        ax.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=1, frameon=False)
    fig.suptitle("Arts LC-Rec SFT Checkpoint Curves", y=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ASSET_DIR / "arts_sft_epoch_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
