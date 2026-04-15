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
ASSET_DIR = REPO_ROOT / "docs" / "assets" / "2026-04-11-genrec-instruments-rl-variant-comparison"
SFT_METRICS_PATH = RESULTS_ROOT / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"

VARIANTS = [
    {
        "key": "rule_only",
        "label": "RL rule_only",
        "model_dir": "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "family": "reward",
        "color": "#2a9d8f",
        "include_in_main": True,
    },
    {
        "key": "dynamic_sid_only",
        "label": "RL dynamic sid-only",
        "model_dir": "Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "family": "dynamic",
        "color": "#4c78a8",
        "include_in_main": True,
    },
    {
        "key": "dynamic_gather_fix",
        "label": "RL dynamic gather-fix",
        "model_dir": "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "family": "dynamic",
        "color": "#1d3557",
        "include_in_main": True,
    },
    {
        "key": "ranking_dynamic",
        "label": "RL ranking dynamic",
        "model_dir": "Instruments-grec-grpo-ranking-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "family": "dynamic",
        "color": "#8d99ae",
        "include_in_main": True,
    },
    {
        "key": "fixed_old",
        "label": "RL fixed old",
        "model_dir": "Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495",
        "family": "fixed",
        "color": "#6d597a",
        "include_in_main": True,
    },
    {
        "key": "fixed_taskfix",
        "label": "RL fixed taskfix",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        "family": "fixed",
        "color": "#f4a261",
        "include_in_main": True,
    },
    {
        "key": "fixed_taskfix_sid_only",
        "label": "RL fixed taskfix sid-only",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495",
        "family": "fixed",
        "color": "#e76f51",
        "include_in_main": True,
    },
    {
        "key": "fixed_taskfix_hintce",
        "label": "RL fixed taskfix + CE",
        "model_dir": "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-sft495",
        "family": "fixed_ce",
        "color": "#d62828",
        "include_in_main": False,
    },
]

METRIC_COLUMNS = ["NDCG@10", "HR@10", "NDCG@50", "HR@50"]
SUPTITLE_Y = 0.94
LEGEND_BBOX_Y = 0.995
TIGHT_LAYOUT_TOP = 0.88


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
    max_step = max(steps)
    for step in steps:
        path = root / f"checkpoint-{step}" / "metrics.json"
        metrics = json.loads(path.read_text())
        row: dict[str, object] = {
            "variant_key": str(variant["key"]),
            "variant_label": str(variant["label"]),
            "model_dir": str(variant["model_dir"]),
            "family": str(variant["family"]),
            "checkpoint": f"checkpoint-{step}",
            "step": step,
            "max_step": max_step,
            "epoch_progress": step / max_step * 2.0 if max_step else 0.0,
            "color": str(variant["color"]),
        }
        for key in METRIC_COLUMNS + ["NDCG@5", "HR@5"]:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def build_dataframe(include_main_only: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        if include_main_only and not bool(variant["include_in_main"]):
            continue
        rows.extend(load_variant_rows(variant))
    return pd.DataFrame(rows)


def build_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for variant_key in df["variant_key"].unique():
        variant_df = df[df["variant_key"] == variant_key]
        best_row = variant_df.sort_values(["NDCG@10", "HR@10", "NDCG@50", "HR@50"], ascending=False).iloc[0]
        peak_row = variant_df.sort_values(["HR@50", "NDCG@10", "HR@10", "NDCG@50"], ascending=False).iloc[0]
        row = best_row.to_dict()
        row["peak_hr50_checkpoint"] = peak_row["checkpoint"]
        row["peak_hr50_epoch"] = peak_row["epoch_progress"]
        row["peak_hr50"] = peak_row["HR@50"]
        row["peak_hr50_ndcg10"] = peak_row["NDCG@10"]
        best_rows.append(row)
    return pd.DataFrame(best_rows)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_metric_panels(
    df: pd.DataFrame,
    variant_keys: list[str],
    title: str,
    out_path: Path,
    sft_reference: dict[str, float | str],
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
                variant_df["epoch_progress"],
                variant_df[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                color=str(variant["color"]),
                label=str(variant["label"]),
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
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        ncol=min(4, len(uniq)),
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
        row = best_df[best_df["variant_key"] == variant_key].iloc[0]
        variant = variants[variant_key]
        ax.scatter(float(row["NDCG@10"]), float(row["HR@50"]), s=120, color=str(variant["color"]), label=str(variant["label"]))
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
    main_df = build_dataframe(include_main_only=True)
    main_best_df = build_best_summary(main_df)
    ce_df = build_dataframe(include_main_only=False)
    ce_best_df = build_best_summary(ce_df)

    save_csv(ASSET_DIR / "rl_variant_checkpoint_metrics.csv", main_df)
    save_csv(ASSET_DIR / "rl_variant_best_summary.csv", main_best_df)
    save_csv(ASSET_DIR / "sft495_reference_metrics.csv", pd.DataFrame([sft_reference]))
    save_csv(
        ASSET_DIR / "fixed_hint_ce_checkpoint_metrics.csv",
        ce_df[ce_df["variant_key"].isin(["fixed_taskfix", "fixed_taskfix_hintce"])],
    )
    save_csv(
        ASSET_DIR / "fixed_hint_ce_best_summary.csv",
        ce_best_df[ce_best_df["variant_key"].isin(["fixed_taskfix", "fixed_taskfix_hintce"])],
    )

    plot_metric_panels(
        main_df,
        [
            "rule_only",
            "dynamic_sid_only",
            "dynamic_gather_fix",
            "ranking_dynamic",
            "fixed_old",
            "fixed_taskfix",
            "fixed_taskfix_sid_only",
        ],
        "Instruments-grec RL Seven-way Curves",
        ASSET_DIR / "rl-seven-way-main-curves.png",
        sft_reference,
    )
    plot_metric_panels(
        main_df,
        ["dynamic_sid_only", "dynamic_gather_fix", "ranking_dynamic"],
        "Instruments-grec Dynamic-family Curves",
        ASSET_DIR / "rl-dynamic-family-curves.png",
        sft_reference,
    )
    plot_metric_panels(
        main_df,
        ["fixed_old", "fixed_taskfix", "fixed_taskfix_sid_only"],
        "Instruments-grec Fixed-family Curves",
        ASSET_DIR / "rl-fixed-family-curves.png",
        sft_reference,
    )
    plot_best_scatter(
        main_best_df,
        [
            "rule_only",
            "dynamic_sid_only",
            "dynamic_gather_fix",
            "ranking_dynamic",
            "fixed_old",
            "fixed_taskfix",
            "fixed_taskfix_sid_only",
        ],
        "Instruments-grec Best NDCG@10 vs HR@50",
        ASSET_DIR / "rl-best-ndcg10-vs-hr50-scatter.png",
    )
    plot_metric_panels(
        ce_df,
        ["fixed_taskfix", "fixed_taskfix_hintce"],
        "Instruments-grec Fixed-hint CE vs Non-CE",
        ASSET_DIR / "rl-fixed-taskfix-ce-vs-nonce-curves.png",
        sft_reference,
    )
    plot_best_scatter(
        ce_best_df,
        ["fixed_taskfix", "fixed_taskfix_hintce"],
        "Instruments-grec Fixed-hint CE vs Non-CE",
        ASSET_DIR / "rl-fixed-taskfix-ce-vs-nonce-scatter.png",
    )

    print(f"main_checkpoint_metrics_csv={ASSET_DIR / 'rl_variant_checkpoint_metrics.csv'}")
    print(f"main_best_summary_csv={ASSET_DIR / 'rl_variant_best_summary.csv'}")
    print(f"ce_checkpoint_metrics_csv={ASSET_DIR / 'fixed_hint_ce_checkpoint_metrics.csv'}")
    print(f"ce_best_summary_csv={ASSET_DIR / 'fixed_hint_ce_best_summary.csv'}")
    print(f"seven_way_png={ASSET_DIR / 'rl-seven-way-main-curves.png'}")
    print(f"ce_curve_png={ASSET_DIR / 'rl-fixed-taskfix-ce-vs-nonce-curves.png'}")


if __name__ == "__main__":
    main()
