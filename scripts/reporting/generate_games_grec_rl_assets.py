#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
ASSET_DIR = REPO_ROOT / "docs" / "assets" / "2026-04-01-games-grec-qwen4b-4-256"

VARIANTS = [
    {
        "key": "sft",
        "label": "SFT",
        "model_dir": "Games-grec-sft-qwen4B-4-256-dsz0",
        "num_train_epochs": 10.0,
        "color": "#4d4d4d",
    },
    {
        "key": "rule_only",
        "label": "RL rule_only",
        "model_dir": "Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896",
        "num_train_epochs": 2.0,
        "color": "#2a9d8f",
    },
    {
        "key": "fixed_hint",
        "label": "RL fixed_hint",
        "model_dir": "Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896",
        "num_train_epochs": 2.0,
        "color": "#f4a261",
    },
    {
        "key": "dynamic_hint",
        "label": "RL dynamic-hint",
        "model_dir": "Games-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft896",
        "num_train_epochs": 2.0,
        "color": "#457b9d",
    },
]

METRIC_COLUMNS = ["NDCG@10", "HR@10", "NDCG@50", "HR@50"]
LEGEND_BBOX_Y = 0.995
SUPTITLE_Y = 0.945
TIGHT_LAYOUT_TOP = 0.88


def _discover_variant_steps(model_dir: str) -> list[int]:
    root = RESULTS_ROOT / model_dir
    metrics_files = sorted(root.glob("checkpoint-*/metrics.json"))
    return [int(path.parent.name.split("-")[-1]) for path in metrics_files]


def load_variant_rows(
    model_dir: str, num_train_epochs: float, max_step_override: int | None = None
) -> list[dict[str, object]]:
    root = RESULTS_ROOT / model_dir
    metrics_files = sorted(root.glob("checkpoint-*/metrics.json"))
    rows: list[dict[str, object]] = []
    if not metrics_files:
        return rows

    discovered_steps = [int(path.parent.name.split("-")[-1]) for path in metrics_files]
    max_step = max_step_override if max_step_override is not None else max(discovered_steps)
    for path in metrics_files:
        step = int(path.parent.name.split("-")[-1])
        metrics = json.loads(path.read_text())
        row = {
            "model_dir": model_dir,
            "checkpoint": path.parent.name,
            "step": step,
            "max_step": max_step,
            "epoch_progress": step / max_step * num_train_epochs if max_step else 0.0,
        }
        for key in METRIC_COLUMNS + ["NDCG@5", "HR@5"]:
            row[key] = metrics.get(key)
        rows.append(row)
    rows.sort(key=lambda item: item["step"])
    return rows


def build_dataframe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rl_shared_max_step = max(
        (
            max(_discover_variant_steps(variant["model_dir"]))
            for variant in VARIANTS
            if variant["key"] != "sft" and _discover_variant_steps(variant["model_dir"])
        ),
        default=None,
    )
    for variant in VARIANTS:
        max_step_override = rl_shared_max_step if variant["key"] != "sft" else None
        variant_rows = load_variant_rows(
            variant["model_dir"],
            variant["num_train_epochs"],
            max_step_override=max_step_override,
        )
        for row in variant_rows:
            row.update(
                {
                    "variant_key": variant["key"],
                    "variant_label": variant["label"],
                    "color": variant["color"],
                }
            )
        rows.extend(variant_rows)
    return pd.DataFrame(rows)


def build_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for variant in VARIANTS:
        variant_df = df[df["variant_key"] == variant["key"]]
        if variant_df.empty:
            continue
        best_row = variant_df.sort_values(["NDCG@10", "HR@10", "NDCG@50", "HR@50"], ascending=False).iloc[0]
        best_rows.append(best_row.to_dict())
    return pd.DataFrame(best_rows)


def save_csvs(df: pd.DataFrame, best_df: pd.DataFrame) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ASSET_DIR / "games_grec_rl_variant_checkpoint_metrics.csv", index=False)
    best_df.to_csv(ASSET_DIR / "games_grec_rl_variant_best_summary.csv", index=False)


def plot_rl_epoch_curves(df: pd.DataFrame, best_df: pd.DataFrame) -> Path:
    rl_df = df[df["variant_key"] != "sft"].copy()
    sft_best = best_df[best_df["variant_key"] == "sft"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = [
        ("NDCG@10", "NDCG@10"),
        ("HR@10", "HR@10"),
        ("NDCG@50", "NDCG@50"),
        ("HR@50", "HR@50"),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        for variant in VARIANTS:
            if variant["key"] == "sft":
                continue
            variant_df = rl_df[rl_df["variant_key"] == variant["key"]]
            ax.plot(
                variant_df["epoch_progress"],
                variant_df[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                color=variant["color"],
                label=variant["label"],
            )
        ax.axhline(float(sft_best[metric]), linestyle="--", linewidth=1.5, color="#666666", label="SFT best")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Games-grec RL Metric Curves by Epoch", y=SUPTITLE_Y)
    fig.tight_layout(rect=(0, 0, 1, TIGHT_LAYOUT_TOP))

    out_path = ASSET_DIR / "games_grec_rl_epoch_curves.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_best_scatter(best_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for variant in VARIANTS:
        row = best_df[best_df["variant_key"] == variant["key"]].iloc[0]
        ax.scatter(
            float(row["NDCG@10"]),
            float(row["HR@50"]),
            s=120,
            color=variant["color"],
            label=variant["label"],
        )
        ax.text(
            float(row["NDCG@10"]) + 0.00015,
            float(row["HR@50"]) + 0.0006,
            str(row["checkpoint"]),
            fontsize=9,
        )

    ax.set_xlabel("Best NDCG@10")
    ax.set_ylabel("HR@50 at Best NDCG@10 Checkpoint")
    ax.set_title("Games-grec Best-Point Trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = ASSET_DIR / "games_grec_best_ndcg10_vs_hr50_scatter.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    df = build_dataframe()
    best_df = build_best_summary(df)
    save_csvs(df, best_df)
    curve_path = plot_rl_epoch_curves(df, best_df)
    scatter_path = plot_best_scatter(best_df)
    print(f"checkpoint_metrics_csv={ASSET_DIR / 'games_grec_rl_variant_checkpoint_metrics.csv'}")
    print(f"best_summary_csv={ASSET_DIR / 'games_grec_rl_variant_best_summary.csv'}")
    print(f"epoch_curve_png={curve_path}")
    print(f"best_scatter_png={scatter_path}")


if __name__ == "__main__":
    main()
