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
ASSET_DIR = REPO_ROOT / "docs" / "assets" / "2026-04-16-instruments-dynamic-hint-max1-ablation"
SFT_METRICS_PATH = RESULTS_ROOT / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"

VARIANTS = [
    {
        "key": "max1",
        "label": "RL dynamic max1",
        "model_dir": "Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#457b9d",
        "max_step": 3326,
        "num_train_epochs": 2.0,
        "include_in_main": True,
    },
    {
        "key": "dynamic_gather_fix",
        "label": "RL dynamic gather-fix",
        "model_dir": "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#1d3557",
        "max_step": 3326,
        "num_train_epochs": 2.0,
        "include_in_main": True,
    },
    {
        "key": "dynamic_sid_only",
        "label": "RL dynamic sid-only",
        "model_dir": "Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#4c78a8",
        "max_step": 2652,
        "num_train_epochs": 2.0,
        "include_in_main": True,
    },
    {
        "key": "rule_only",
        "label": "RL rule_only",
        "model_dir": "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#2a9d8f",
        "max_step": 3326,
        "num_train_epochs": 2.0,
        "include_in_main": True,
    },
    {
        "key": "fixed_taskfix_sid_only",
        "label": "RL fixed taskfix sid-only",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495",
        "color": "#e76f51",
        "max_step": 2652,
        "num_train_epochs": 2.0,
        "include_in_main": True,
    },
    {
        "key": "fixed_taskfix",
        "label": "RL fixed taskfix",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        "color": "#f4a261",
        "max_step": 3326,
        "num_train_epochs": 2.0,
        "include_in_main": False,
    },
    {
        "key": "fixed_old",
        "label": "RL fixed old",
        "model_dir": "Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#6d597a",
        "max_step": 3326,
        "num_train_epochs": 2.0,
        "include_in_main": False,
    },
]

METRIC_COLUMNS = ["NDCG@10", "HR@10", "NDCG@50", "HR@50"]
MAIN_VARIANT_KEYS = ["max1", "dynamic_gather_fix", "dynamic_sid_only", "rule_only", "fixed_taskfix_sid_only"]
FOCUS_VARIANT_KEYS = ["fixed_taskfix", "dynamic_gather_fix", "max1", "fixed_old"]


def load_sft_reference() -> dict[str, float | str]:
    metrics = json.loads(SFT_METRICS_PATH.read_text())
    row: dict[str, float | str] = {
        "variant_key": "sft495",
        "variant_label": "SFT495",
        "checkpoint": "checkpoint-495",
        "step": 495,
    }
    for key in METRIC_COLUMNS + ["NDCG@5", "HR@5"]:
        row[key] = metrics.get(key)
    return row


def load_variant_rows(variant: dict[str, str]) -> list[dict[str, object]]:
    root = RESULTS_ROOT / variant["model_dir"]
    metrics_files = list(root.glob("checkpoint-*/metrics.json"))
    rows: list[dict[str, object]] = []
    observed_max_step = max((int(path.parent.name.split("-")[-1]) for path in metrics_files), default=0)
    for path in sorted(metrics_files, key=lambda p: int(p.parent.name.split("-")[-1])):
        step = int(path.parent.name.split("-")[-1])
        metrics = json.loads(path.read_text())
        row: dict[str, object] = {
            "variant_key": variant["key"],
            "variant_label": variant["label"],
            "model_dir": variant["model_dir"],
            "checkpoint": f"checkpoint-{step}",
            "step": step,
            "max_step": variant["max_step"],
            "observed_max_step": observed_max_step,
            "epoch_progress": step / variant["max_step"] * variant["num_train_epochs"] if variant["max_step"] else 0.0,
            "color": variant["color"],
        }
        for key in METRIC_COLUMNS + ["NDCG@5", "HR@5"]:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def build_dataframe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        rows.extend(load_variant_rows(variant))
    return pd.DataFrame(rows)


def build_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        variant_df = df[df["variant_key"] == variant["key"]]
        best_row = variant_df.sort_values(["NDCG@10", "HR@10", "NDCG@50", "HR@50"], ascending=False).iloc[0]
        peak_row = variant_df.sort_values(["HR@50", "NDCG@10", "HR@10", "NDCG@50"], ascending=False).iloc[0]
        row = best_row.to_dict()
        row["best_epoch_progress"] = best_row["epoch_progress"]
        row["peak_hr50_checkpoint"] = peak_row["checkpoint"]
        row["peak_hr50_step"] = peak_row["step"]
        row["peak_hr50_epoch_progress"] = peak_row["epoch_progress"]
        row["peak_hr50"] = peak_row["HR@50"]
        row["peak_hr50_ndcg10"] = peak_row["NDCG@10"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_epoch_curves(df: pd.DataFrame, sft_reference: dict[str, float | str], out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = [
        ("NDCG@10", "NDCG@10"),
        ("HR@10", "HR@10"),
        ("NDCG@50", "NDCG@50"),
        ("HR@50", "HR@50"),
    ]
    for ax, (metric, metric_title) in zip(axes.flat, metrics, strict=True):
        for variant in VARIANTS:
            if not variant["include_in_main"]:
                continue
            variant_df = df[df["variant_key"] == variant["key"]]
            ax.plot(
                variant_df["epoch_progress"],
                variant_df[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                color=variant["color"],
                label=variant["label"],
            )
        ax.axhline(float(sft_reference[metric]), linestyle="--", linewidth=1.5, color="#666666", label="SFT495")
        ax.set_title(metric_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Instruments-grec Dynamic-hint Max1 Ablation", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_focus_epoch_curves(df: pd.DataFrame, sft_reference: dict[str, float | str], out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = [
        ("NDCG@10", "NDCG@10"),
        ("HR@10", "HR@10"),
        ("NDCG@50", "NDCG@50"),
        ("HR@50", "HR@50"),
    ]
    variant_map = {variant["key"]: variant for variant in VARIANTS}

    for ax, (metric, metric_title) in zip(axes.flat, metrics, strict=True):
        for variant_key in FOCUS_VARIANT_KEYS:
            variant = variant_map[variant_key]
            variant_df = df[df["variant_key"] == variant_key]
            ax.plot(
                variant_df["epoch_progress"],
                variant_df[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                color=variant["color"],
                label=variant["label"],
            )
        ax.axhline(float(sft_reference[metric]), linestyle="--", linewidth=1.5, color="#666666", label="SFT495")
        ax.set_title(metric_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Instruments-grec Focused Comparison: Fixed vs Dynamic Max1", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    sft_reference = load_sft_reference()
    df = build_dataframe()
    best_df = build_best_summary(df)

    save_csv(ASSET_DIR / "max1_ablation_checkpoint_metrics.csv", df)
    save_csv(ASSET_DIR / "max1_ablation_best_summary.csv", best_df)
    save_csv(ASSET_DIR / "sft495_reference_metrics.csv", pd.DataFrame([sft_reference]))
    plot_epoch_curves(df, sft_reference, ASSET_DIR / "max1-ablation-epoch-curves.png")
    plot_focus_epoch_curves(df, sft_reference, ASSET_DIR / "max1-vs-fixed-epoch-curves.png")

    print(f"checkpoint_metrics_csv={ASSET_DIR / 'max1_ablation_checkpoint_metrics.csv'}")
    print(f"best_summary_csv={ASSET_DIR / 'max1_ablation_best_summary.csv'}")
    print(f"epoch_curve_png={ASSET_DIR / 'max1-ablation-epoch-curves.png'}")
    print(f"focus_curve_png={ASSET_DIR / 'max1-vs-fixed-epoch-curves.png'}")


if __name__ == "__main__":
    main()
