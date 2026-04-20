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
ASSET_DIR = REPO_ROOT / "docs" / "assets" / "2026-04-19-instruments-dual-task-single-hint-tracking"
SFT_METRICS_PATH = RESULTS_ROOT / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"

VARIANTS = [
    {
        "key": "rule_only",
        "label": "RL rule_only",
        "model_dir": "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#2a9d8f",
    },
    {
        "key": "dynamic_gather_fix",
        "label": "RL dynamic gather-fix",
        "model_dir": "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#1d3557",
    },
    {
        "key": "dynamic_dual_task",
        "label": "RL dynamic dual-task",
        "model_dir": "Instruments-grec-grpo-rule-only-dynamic-hint-sid-title-desc-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#457b9d",
    },
    {
        "key": "fixed_old",
        "label": "RL fixed old",
        "model_dir": "Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "color": "#6d597a",
    },
    {
        "key": "fixed_taskfix",
        "label": "RL fixed taskfix",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        "color": "#f4a261",
    },
    {
        "key": "fixed_taskfix_sid_only",
        "label": "RL fixed taskfix sid-only",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495",
        "color": "#e76f51",
        "epoch_max_step": 2652,
    },
    {
        "key": "single_hint_mixed",
        "label": "RL single-hint mixed",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-hint-only-mixed-sft495",
        "color": "#b5179e",
        "epoch_max_step": 3326,
    },
]

METRIC_COLUMNS = ["NDCG@10", "HR@10", "NDCG@50", "HR@50"]
SUPTITLE_Y = 0.94
LEGEND_BBOX_Y = 0.995
TIGHT_LAYOUT_TOP = 0.88
DYNAMIC_FAMILY_KEYS = ["dynamic_gather_fix", "dynamic_dual_task"]
FIXED_FAMILY_KEYS = ["fixed_old", "fixed_taskfix", "fixed_taskfix_sid_only", "single_hint_mixed"]
LATE_ALIGNMENT_REFERENCE_KEY = "dynamic_dual_task"
LATE_ALIGNMENT_EPOCH_START = 1.75
LATE_ALIGNMENT_EPOCH_END = 2.0


def _variant_map() -> dict[str, dict[str, object]]:
    return {variant["key"]: variant for variant in VARIANTS}


def load_sft_reference() -> dict[str, float | str]:
    metrics = json.loads(SFT_METRICS_PATH.read_text())
    row: dict[str, float | str] = {
        "variant_key": "sft495",
        "variant_label": "SFT495",
        "checkpoint": "checkpoint-495",
    }
    for key in METRIC_COLUMNS + ["NDCG@5", "HR@5"]:
        row[key] = metrics.get(key)
    return row


def load_variant_rows(variant: dict[str, object]) -> list[dict[str, object]]:
    root = RESULTS_ROOT / str(variant["model_dir"])
    metrics_files = list(root.glob("checkpoint-*/metrics.json"))
    if not metrics_files:
        return []

    rows: list[dict[str, object]] = []
    steps = sorted(int(path.parent.name.split("-")[-1]) for path in metrics_files)
    observed_max_step = max(steps)
    max_step = int(variant.get("epoch_max_step", observed_max_step))
    for step in steps:
        path = root / f"checkpoint-{step}" / "metrics.json"
        metrics = json.loads(path.read_text())
        row: dict[str, object] = {
            "variant_key": str(variant["key"]),
            "variant_label": str(variant["label"]),
            "model_dir": str(variant["model_dir"]),
            "checkpoint": f"checkpoint-{step}",
            "step": step,
            "max_step": max_step,
            "observed_max_step": observed_max_step,
            "epoch_progress": step / max_step * 2.0 if max_step else 0.0,
            "color": str(variant["color"]),
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


def build_best_summary(df: pd.DataFrame, epoch_column: str = "epoch_progress") -> pd.DataFrame:
    best_rows = []
    for variant_key in df["variant_key"].unique():
        variant_df = df[df["variant_key"] == variant_key]
        best_row = variant_df.sort_values(["NDCG@10", "HR@10", "NDCG@50", "HR@50"], ascending=False).iloc[0]
        peak_row = variant_df.sort_values(["HR@50", "NDCG@10", "HR@10", "NDCG@50"], ascending=False).iloc[0]
        row = best_row.to_dict()
        row["peak_hr50_checkpoint"] = peak_row["checkpoint"]
        row["peak_hr50_epoch"] = peak_row.get(epoch_column)
        row["peak_hr50"] = peak_row["HR@50"]
        row["peak_hr50_ndcg10"] = peak_row["NDCG@10"]
        best_rows.append(row)
    return pd.DataFrame(best_rows)


def _build_epoch_grid(point_count: int, epoch_start: float, epoch_end: float) -> list[float]:
    if point_count <= 0:
        return []
    if point_count == 1:
        return [epoch_end]
    interval = (epoch_end - epoch_start) / (point_count - 1)
    return [epoch_start + interval * idx for idx in range(point_count)]


def build_late_aligned_dataframe(
    df: pd.DataFrame,
    reference_key: str = LATE_ALIGNMENT_REFERENCE_KEY,
    epoch_start: float = LATE_ALIGNMENT_EPOCH_START,
    epoch_end: float = LATE_ALIGNMENT_EPOCH_END,
) -> pd.DataFrame:
    reference_df = df[df["variant_key"] == reference_key].sort_values("step")
    if reference_df.empty:
        raise ValueError(f"Missing reference variant for late alignment: {reference_key}")

    point_count = len(reference_df)
    aligned_epochs = _build_epoch_grid(point_count, epoch_start, epoch_end)
    rows: list[dict[str, object]] = []

    for variant_key in df["variant_key"].unique():
        variant_df = df[df["variant_key"] == variant_key].sort_values("step").tail(point_count).reset_index(drop=True)
        if variant_df.empty:
            continue

        variant_epochs = aligned_epochs[-len(variant_df) :]
        for idx, (_, row) in enumerate(variant_df.iterrows(), start=1):
            out_row = row.to_dict()
            out_row["aligned_epoch"] = variant_epochs[idx - 1]
            out_row["aligned_tail_index"] = idx
            out_row["aligned_tail_point_count"] = len(variant_df)
            out_row["aligned_reference_key"] = reference_key
            rows.append(out_row)

    return pd.DataFrame(rows)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_metric_panels(
    df: pd.DataFrame,
    variant_keys: list[str],
    title: str,
    out_path: Path,
    sft_reference: dict[str, float | str],
    x_column: str = "epoch_progress",
    x_label: str = "Epoch",
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = [
        ("NDCG@10", "NDCG@10"),
        ("HR@10", "HR@10"),
        ("NDCG@50", "NDCG@50"),
        ("HR@50", "HR@50"),
    ]
    variants = _variant_map()

    for ax, (metric, metric_title) in zip(axes.flat, metrics, strict=True):
        for variant_key in variant_keys:
            variant_df = df[df["variant_key"] == variant_key]
            if variant_df.empty:
                continue
            variant = variants[variant_key]
            ax.plot(
                variant_df[x_column],
                variant_df[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                color=str(variant["color"]),
                label=str(variant["label"]),
            )
        ax.axhline(float(sft_reference[metric]), linestyle="--", linewidth=1.5, color="#666666", label="SFT495")
        ax.set_title(metric_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        ncol=min(3, len(uniq)),
        frameon=False,
    )
    fig.suptitle(title, y=SUPTITLE_Y)
    fig.tight_layout(rect=(0, 0, 1, TIGHT_LAYOUT_TOP))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_best_scatter(best_df: pd.DataFrame, variant_keys: list[str], title: str, out_path: Path) -> Path:
    variants = _variant_map()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for variant_key in variant_keys:
        variant_rows = best_df[best_df["variant_key"] == variant_key]
        if variant_rows.empty:
            continue
        row = variant_rows.iloc[0]
        variant = variants[variant_key]
        ax.scatter(
            float(row["NDCG@10"]),
            float(row["HR@50"]),
            s=120,
            color=str(variant["color"]),
            label=str(variant["label"]),
        )
        ax.text(float(row["NDCG@10"]) + 0.00015, float(row["HR@50"]) + 0.0005, str(row["checkpoint"]), fontsize=9)

    ax.set_xlabel("Best NDCG@10")
    ax.set_ylabel("HR@50 at Best NDCG@10 Checkpoint")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.margins(x=0.08, y=0.08)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    sft_reference = load_sft_reference()
    df = build_dataframe()
    best_df = build_best_summary(df)
    late_aligned_df = build_late_aligned_dataframe(df)
    late_aligned_best_df = build_best_summary(late_aligned_df, epoch_column="aligned_epoch")

    save_csv(ASSET_DIR / "single_hint_tracking_checkpoint_metrics.csv", df)
    save_csv(ASSET_DIR / "single_hint_tracking_best_summary.csv", best_df)
    save_csv(ASSET_DIR / "late_epoch_aligned_checkpoint_metrics.csv", late_aligned_df)
    save_csv(ASSET_DIR / "late_epoch_aligned_best_summary.csv", late_aligned_best_df)
    save_csv(ASSET_DIR / "sft495_reference_metrics.csv", pd.DataFrame([sft_reference]))

    variant_keys = [variant["key"] for variant in VARIANTS]
    plot_metric_panels(
        df,
        variant_keys,
        "Instruments-grec Single-hint Mixed vs Baselines",
        ASSET_DIR / "single_hint_vs_baselines_epoch_curves.png",
        sft_reference,
    )
    plot_metric_panels(
        df,
        DYNAMIC_FAMILY_KEYS,
        "Instruments-grec Dynamic Family Curves",
        ASSET_DIR / "single_hint_vs_dynamic_family_epoch_curves.png",
        sft_reference,
    )
    plot_metric_panels(
        df,
        FIXED_FAMILY_KEYS,
        "Instruments-grec Fixed Family Curves",
        ASSET_DIR / "single_hint_vs_fixed_family_epoch_curves.png",
        sft_reference,
    )
    plot_metric_panels(
        late_aligned_df,
        variant_keys,
        "Instruments-grec Single-hint Mixed vs Baselines (Late Epoch Aligned)",
        ASSET_DIR / "single_hint_vs_baselines_late_epoch_aligned_curves.png",
        sft_reference,
        x_column="aligned_epoch",
        x_label="Aligned Epoch",
    )
    plot_metric_panels(
        late_aligned_df,
        DYNAMIC_FAMILY_KEYS,
        "Instruments-grec Dynamic Family Curves (Late Epoch Aligned)",
        ASSET_DIR / "single_hint_vs_dynamic_family_late_epoch_aligned_curves.png",
        sft_reference,
        x_column="aligned_epoch",
        x_label="Aligned Epoch",
    )

    print(f"checkpoint_metrics_csv={ASSET_DIR / 'single_hint_tracking_checkpoint_metrics.csv'}")
    print(f"best_summary_csv={ASSET_DIR / 'single_hint_tracking_best_summary.csv'}")
    print(f"late_aligned_checkpoint_metrics_csv={ASSET_DIR / 'late_epoch_aligned_checkpoint_metrics.csv'}")
    print(f"late_aligned_best_summary_csv={ASSET_DIR / 'late_epoch_aligned_best_summary.csv'}")
    print(f"full_curve_png={ASSET_DIR / 'single_hint_vs_baselines_epoch_curves.png'}")
    print(f"dynamic_family_png={ASSET_DIR / 'single_hint_vs_dynamic_family_epoch_curves.png'}")
    print(f"fixed_family_png={ASSET_DIR / 'single_hint_vs_fixed_family_epoch_curves.png'}")
    print(f"late_aligned_full_curve_png={ASSET_DIR / 'single_hint_vs_baselines_late_epoch_aligned_curves.png'}")
    print(f"late_aligned_dynamic_family_png={ASSET_DIR / 'single_hint_vs_dynamic_family_late_epoch_aligned_curves.png'}")


if __name__ == "__main__":
    main()
