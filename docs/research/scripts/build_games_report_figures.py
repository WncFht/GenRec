#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from instruments_plot_lib import (
    VariantSpec,
    build_best_summary,
    build_dataframe,
    export_metadata,
    plot_best_scatter,
    plot_metric_grid,
    save_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "games-report"
ACTIVE_PALETTE = "tableau-10"

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

SFT_MODEL_DIR = "Games-grec-sft-qwen4B-4-256-dsz0"
SFT_RL_INIT_STEP = 896
METRIC_COLUMNS = ["NDCG@10", "HR@10", "NDCG@50", "HR@50"]

RL_SPECS = [
    VariantSpec(
        key="rule_only",
        label="RL rule-only",
        model_dir="Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896",
        color=TABLEAU_10["brown"],
        marker="o",
    ),
    VariantSpec(
        key="dynamic_hint",
        label="RL dynamic-hint",
        model_dir="Games-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft896",
        color=TABLEAU_10["blue"],
        marker="s",
    ),
    VariantSpec(
        key="fixed_hint",
        label="RL fixed-hint",
        model_dir="Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896",
        color=TABLEAU_10["orange"],
        marker="D",
    ),
]

SCATTER_SPECS = [
    VariantSpec(
        key="sft_best",
        label="SFT best",
        model_dir=SFT_MODEL_DIR,
        color=TABLEAU_10["gray"],
        marker="P",
    ),
    *RL_SPECS,
]


def load_games_sft_dataframe() -> pd.DataFrame:
    root = RESULTS_ROOT / SFT_MODEL_DIR
    metrics_files = sorted(root.glob("checkpoint-*/metrics.json"), key=lambda path: int(path.parent.name.split("-")[-1]))
    rows: list[dict[str, object]] = []
    for path in metrics_files:
        step = int(path.parent.name.split("-")[-1])
        metrics = pd.read_json(path, typ="series")
        row: dict[str, object] = {
            "variant_key": "sft",
            "variant_label": "SFT checkpoints",
            "model_dir": SFT_MODEL_DIR,
            "checkpoint": f"checkpoint-{step}",
            "step": step,
        }
        for key in METRIC_COLUMNS + ["NDCG@5", "HR@5"]:
            row[key] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def build_games_sft_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    best_row = df.sort_values(["NDCG@10", "HR@10", "NDCG@50", "HR@50"], ascending=False).iloc[0].to_dict()
    best_row["variant_key"] = "sft_best"
    best_row["variant_label"] = "SFT best"
    return pd.DataFrame([best_row])


def plot_games_sft_checkpoint_curves(df: pd.DataFrame, out_path: Path) -> Path:
    best_step = int(df.sort_values(["NDCG@10", "HR@10", "NDCG@50", "HR@50"], ascending=False).iloc[0]["step"])
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), sharex=True)
    metrics = [("NDCG@10", "NDCG@10"), ("HR@10", "HR@10"), ("NDCG@50", "NDCG@50"), ("HR@50", "HR@50")]

    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        ax.plot(
            df["step"],
            df[metric],
            color=TABLEAU_10["blue"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label="SFT checkpoints",
        )
        ax.axvline(best_step, color=TABLEAU_10["orange"], linestyle="--", linewidth=1.6, label=f"Best NDCG@10: {best_step}")
        ax.axvline(
            SFT_RL_INIT_STEP,
            color=TABLEAU_10["red"],
            linestyle=":",
            linewidth=1.8,
            label=f"RL init: {SFT_RL_INIT_STEP}",
        )
        ax.set_title(title)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.suptitle("Games SFT Checkpoint Selection", y=0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    export_metadata(ASSET_DIR / "games_rl_variants_metadata.json", RL_SPECS)

    print(f"active_palette={ACTIVE_PALETTE}")

    sft_df = load_games_sft_dataframe()
    sft_best_df = build_games_sft_best_summary(sft_df)
    rl_df = build_dataframe(RL_SPECS)
    rl_best_df = build_best_summary(rl_df)
    combined_best_df = pd.concat([sft_best_df, rl_best_df], ignore_index=True)
    sft_reference = sft_best_df.iloc[0].to_dict()

    save_csv(ASSET_DIR / "games_sft_checkpoint_metrics.csv", sft_df)
    save_csv(ASSET_DIR / "games_sft_best_summary.csv", sft_best_df)
    save_csv(ASSET_DIR / "games_rl_checkpoint_metrics.csv", rl_df)
    save_csv(ASSET_DIR / "games_rl_best_summary.csv", rl_best_df)

    plot_games_sft_checkpoint_curves(sft_df, ASSET_DIR / "games_sft_checkpoint_curves.png")
    plot_metric_grid(
        rl_df,
        RL_SPECS,
        ["rule_only", "dynamic_hint", "fixed_hint"],
        "Games RL Metric Curves by Epoch",
        ASSET_DIR / "games_rl_epoch_curves.png",
        sft_reference,
        legend_cols=4,
        reference_label=f"SFT best ({sft_reference['checkpoint']})",
    )
    plot_best_scatter(
        combined_best_df,
        SCATTER_SPECS,
        ["sft_best", "rule_only", "dynamic_hint", "fixed_hint"],
        "Games Best NDCG@10 vs HR@50",
        ASSET_DIR / "games_best_ndcg10_vs_hr50_scatter.png",
        y_label="HR@50 at Best NDCG@10 Checkpoint",
        text_dx=0.00003,
        x_margin=0.16,
    )

    print(f"asset_dir={ASSET_DIR}")
    print(f"sft_metrics_csv={ASSET_DIR / 'games_sft_checkpoint_metrics.csv'}")
    print(f"rl_metrics_csv={ASSET_DIR / 'games_rl_checkpoint_metrics.csv'}")


if __name__ == "__main__":
    main()
