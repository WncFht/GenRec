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
    load_sft_reference,
    plot_best_scatter,
    plot_metric_grid,
    plot_single_metric,
    save_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs"
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "instruments-report"
ACTIVE_PALETTE = "tableau-10"
ACTIVE_CE_PROFILE = "CE-A"
BUG_DEPTH_DISTRIBUTION_PATH = (
    DOCS_ROOT / "deepresearch" / "genrec_rl_study_2026-03-28" / "data" / "fixed_hint_bug_depth_distribution.csv"
)
BUG_TASK_SUMMARY_PATH = (
    DOCS_ROOT / "deepresearch" / "genrec_rl_study_2026-03-28" / "data" / "fixed_hint_bug_task_summary.csv"
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

SEVEN_WAY_MAIN_KEYS = [
    "rule_only",
    "dynamic_sid_only",
    "dynamic_gather_fix",
    "ranking_dynamic",
    "fixed_old",
    "fixed_taskfix",
    "fixed_taskfix_sid_only",
]
MAX1_ABLATION_KEYS = ["max1", "dynamic_gather_fix", "dynamic_sid_only", "rule_only", "fixed_taskfix_sid_only"]
MAX1_FOCUS_KEYS = ["fixed_taskfix", "dynamic_gather_fix", "max1", "fixed_old"]
SINGLE_HINT_FIXED_KEYS = ["fixed_old", "fixed_taskfix", "fixed_taskfix_sid_only", "single_hint_mixed"]
SINGLE_HINT_BASELINE_KEYS = [
    "rule_only",
    "dynamic_gather_fix",
    "dynamic_dual_task",
    "dynamic_single_hint_mixed",
    "single_hint_mixed",
]
CE_FIXED_KEYS = ["fixed_taskfix", "hintce_batch_mean", "hintce_token_mean", "hintce_coef_005", "hintce_coef_01"]
CE_DYNAMIC_FIRSTLOOK_KEYS = ["dynamic_gather_fix", "dynamic_hint_ce005", "fixed_taskfix", "hintce_coef_01"]
PROMISING_CANDIDATE_KEYS = [
    "rule_only",
    "single_hint_mixed",
    "fixed_old",
    "fixed_taskfix_sid_only",
    "fixed_taskfix",
    "hintce_coef_005",
    "hintce_coef_01",
]
DEPTH_COLUMNS = [
    ("depth_0_ratio", "depth-0", TABLEAU_10["blue"]),
    ("depth_1_ratio", "depth-1", TABLEAU_10["green"]),
    ("depth_2_ratio", "depth-2", TABLEAU_10["orange"]),
    ("depth_3_ratio", "depth-3", TABLEAU_10["red"]),
]
BUG_SCOPE_ORDER = ["overall", "sid", "hisTitle2sid", "title_desc2sid"]
BUG_SCOPE_LABELS = {
    "overall": "overall",
    "sid": "sid",
    "hisTitle2sid": "hisTitle2sid",
    "title_desc2sid": "title_desc2sid",
}
BUG_VERSION_ORDER = ["correct_task_index", "legacy_index_only"]
BUG_VERSION_LABELS = {
    "correct_task_index": "task+index",
    "legacy_index_only": "legacy index",
}
BUG_SCOPE_TO_TASK = {
    "sid": "task1_sid_sft",
    "hisTitle2sid": "task4_hisTitle2sid",
    "title_desc2sid": "task5_title_desc2sid",
}

SPECS = [
    VariantSpec(
        key="rule_only",
        label="RL rule-only",
        model_dir="Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["brown"],
        marker="o",
    ),
    VariantSpec(
        key="dynamic_sid_only",
        label="RL dynamic sid-only",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["teal"],
        marker="s",
        epoch_max_step=2652,
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-only.sh",
    ),
    VariantSpec(
        key="dynamic_gather_fix",
        label="RL dynamic gather-fix",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["blue"],
        marker="o",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh",
    ),
    VariantSpec(
        key="dynamic_hint_ce005",
        label="RL dynamic CE coef=0.005",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-hintce005-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["yellow"],
        marker="X",
        linestyle="--",
        epoch_max_step_ref_key="dynamic_gather_fix",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-ce.sh",
    ),
    VariantSpec(
        key="ranking_dynamic",
        label="RL ranking dynamic",
        model_dir="Instruments-grec-grpo-ranking-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["gray"],
        marker="^",
        linestyle="--",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-ranking-dynamic-hint.sh",
    ),
    VariantSpec(
        key="fixed_taskfix",
        label="RL fixed taskfix",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        color=TABLEAU_10["orange"],
        marker="o",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint.sh",
    ),
    VariantSpec(
        key="fixed_taskfix_sid_only",
        label="RL fixed taskfix sid-only",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495",
        color=TABLEAU_10["red"],
        marker="s",
        epoch_max_step=2652,
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-only.sh",
    ),
    VariantSpec(
        key="fixed_old",
        label="RL fixed old",
        model_dir="Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["purple"],
        marker="D",
        linestyle="-.",
    ),
    VariantSpec(
        key="max1",
        label="RL dynamic max1",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["green"],
        marker="P",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-max1.sh",
    ),
    VariantSpec(
        key="hintce_batch_mean",
        label="RL fixed CE batch-mean",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-sft495",
        color=TABLEAU_10["blue"],
        marker="s",
        linestyle="--",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh",
    ),
    VariantSpec(
        key="hintce_token_mean",
        label="RL fixed CE token-mean",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-2-sft495",
        color=TABLEAU_10["green"],
        marker="^",
        linestyle="-.",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh",
    ),
    VariantSpec(
        key="hintce_coef_005",
        label="RL fixed CE coef=0.005",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
        color=TABLEAU_10["red"],
        marker="D",
        linestyle=":",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh",
    ),
    VariantSpec(
        key="hintce_coef_01",
        label="RL fixed CE coef=0.01",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-4-sft495",
        color=TABLEAU_10["purple"],
        marker="X",
        epoch_max_step_ref_key="fixed_taskfix",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh",
    ),
    VariantSpec(
        key="single_hint_mixed",
        label="RL single-hint mixed",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-hint-only-mixed-sft495",
        color=TABLEAU_10["pink"],
        marker="P",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-hint-only-mixed.sh",
    ),
    VariantSpec(
        key="dynamic_single_hint_mixed",
        label="RL dynamic single-hint mixed",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-sid-hint-only-mixed-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#4361EE",
        marker="^",
        linestyle="--",
        epoch_max_step_ref_key="single_hint_mixed",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-hint-only-mixed.sh",
    ),
    VariantSpec(
        key="dynamic_dual_task",
        label="RL dynamic dual-task",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-sid-title-desc-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color=TABLEAU_10["teal"],
        marker="o",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-title-desc.sh",
    ),
    VariantSpec(
        key="fixed_dual_task",
        label="RL fixed dual-task",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-title-desc-sft495",
        color=TABLEAU_10["gray"],
        marker="s",
        epoch_max_step_ref_key="dynamic_dual_task",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh",
    ),
]


def plot_fixed_hint_bug_depth_distribution(out_path: Path) -> Path:
    depth_df = pd.read_csv(BUG_DEPTH_DISTRIBUTION_PATH)
    task_df = pd.read_csv(BUG_TASK_SUMMARY_PATH)
    changed_rate_labels = {
        scope: f"changed {task_df.loc[task_df['task'] == task_name, 'changed_rate'].iloc[0] * 100:.1f}%"
        for scope, task_name in BUG_SCOPE_TO_TASK.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), sharey=True)
    bar_positions = [0, 1]

    for ax, scope in zip(axes.flat, BUG_SCOPE_ORDER, strict=True):
        scope_df = depth_df[depth_df["scope"] == scope].set_index("version").loc[BUG_VERSION_ORDER]
        bottoms = [0.0, 0.0]
        for column, label, color in DEPTH_COLUMNS:
            values = (scope_df[column].astype(float) * 100.0).tolist()
            ax.bar(
                bar_positions,
                values,
                bottom=bottoms,
                width=0.8,
                color=color,
                label=label,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]

        ax.set_title(BUG_SCOPE_LABELS[scope])
        ax.set_xticks(bar_positions, [BUG_VERSION_LABELS[key] for key in BUG_VERSION_ORDER])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.22)
        for idx, version in enumerate(BUG_VERSION_ORDER):
            total = int(scope_df.loc[version, "total"])
            ax.text(bar_positions[idx], 101.2, f"n={total}", ha="center", va="bottom", fontsize=9)
        if scope in changed_rate_labels:
            ax.text(
                0.5,
                96.5,
                changed_rate_labels[scope],
                ha="center",
                va="top",
                fontsize=9,
                color="#374151",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
            )

    axes[0, 0].set_ylabel("Share of samples (%)")
    axes[1, 0].set_ylabel("Share of samples (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle("Fixed-hint Training Depth Distribution: Task+Index vs Legacy Index-only Map", y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    export_metadata(ASSET_DIR / "instrument_variants_metadata.json", SPECS)

    print(f"active_palette={ACTIVE_PALETTE}")
    print(f"active_ce_profile={ACTIVE_CE_PROFILE}")

    df = build_dataframe(SPECS)
    best_df = build_best_summary(df)
    sft = load_sft_reference()

    save_csv(ASSET_DIR / "all_variant_checkpoint_metrics.csv", df)
    save_csv(ASSET_DIR / "all_variant_best_summary.csv", best_df)
    save_csv(
        ASSET_DIR / "promising_candidates_checkpoint_metrics.csv",
        df[df["variant_key"].isin(PROMISING_CANDIDATE_KEYS)],
    )
    save_csv(
        ASSET_DIR / "promising_candidates_best_summary.csv",
        best_df[best_df["variant_key"].isin(PROMISING_CANDIDATE_KEYS)],
    )

    plot_metric_grid(
        df,
        SPECS,
        SEVEN_WAY_MAIN_KEYS,
        "Instruments RL Seven-Way Main Curves",
        ASSET_DIR / "rl-seven-way-main-curves.png",
        sft,
        legend_cols=4,
    )
    plot_best_scatter(
        best_df,
        SPECS,
        SEVEN_WAY_MAIN_KEYS,
        "Instruments Best NDCG@10 vs HR@50",
        ASSET_DIR / "rl-best-ndcg10-vs-hr50-scatter.png",
    )
    plot_metric_grid(
        df,
        SPECS,
        ["dynamic_sid_only", "dynamic_gather_fix"],
        "Instruments Dynamic Sid-Only vs Dynamic Gather-Fix",
        ASSET_DIR / "dynamic_sid_only_vs_dynamic_gather_fix_curves.png",
        sft,
        legend_cols=3,
    )
    plot_metric_grid(
        df,
        SPECS,
        ["fixed_taskfix", "fixed_taskfix_sid_only", "fixed_old"],
        "Instruments Fixed Sid-Only vs Fixed Taskfix",
        ASSET_DIR / "fixed_sid_only_vs_fixed_taskfix_curves.png",
        sft,
        legend_cols=4,
    )
    plot_metric_grid(
        df,
        SPECS,
        ["ranking_dynamic", "dynamic_gather_fix"],
        "Instruments Ranking Dynamic vs Canonical Dynamic",
        ASSET_DIR / "ranking_dynamic_vs_canonical_dynamic_curves.png",
        sft,
        legend_cols=3,
    )
    plot_metric_grid(
        df,
        SPECS,
        ["fixed_old", "fixed_taskfix", "fixed_taskfix_sid_only"],
        "Instruments Old Fixed vs Corrected Fixed",
        ASSET_DIR / "old_fixed_vs_corrected_fixed_curves.png",
        sft,
        legend_cols=4,
    )
    plot_metric_grid(
        df,
        SPECS,
        MAX1_ABLATION_KEYS,
        "Instruments Dynamic-Hint Max1 Ablation",
        ASSET_DIR / "max1-ablation-epoch-curves.png",
        sft,
        legend_cols=3,
    )
    plot_metric_grid(
        df,
        SPECS,
        MAX1_FOCUS_KEYS,
        "Instruments Focused Comparison: Fixed vs Dynamic Max1",
        ASSET_DIR / "max1-vs-fixed-epoch-curves.png",
        sft,
        legend_cols=3,
    )
    plot_metric_grid(
        df,
        SPECS,
        CE_FIXED_KEYS,
        "Instruments Fixed Hint CE Scaling",
        ASSET_DIR / "ce_scaling_three_variants_curves.png",
        sft,
        legend_cols=3,
    )
    plot_metric_grid(
        df,
        SPECS,
        CE_DYNAMIC_FIRSTLOOK_KEYS,
        "Instruments Dynamic Hint+CE First Look",
        ASSET_DIR / "ce_scaling_dynamic_first_look_curves.png",
        sft,
        legend_cols=4,
        x_lim=(0.0, 0.45),
    )
    plot_metric_grid(
        df,
        SPECS,
        SINGLE_HINT_FIXED_KEYS,
        "Instruments Single-Hint Mixed vs Fixed Family",
        ASSET_DIR / "single_hint_vs_fixed_family_epoch_curves.png",
        sft,
        legend_cols=4,
    )
    plot_metric_grid(
        df,
        SPECS,
        SINGLE_HINT_BASELINE_KEYS,
        "Instruments Single-Hint Mixed vs Dynamic Family",
        ASSET_DIR / "single_hint_mixed_vs_baselines_compact_curves.png",
        sft,
        legend_cols=3,
    )
    plot_metric_grid(
        df,
        SPECS,
        ["dynamic_dual_task", "fixed_dual_task", "dynamic_gather_fix", "fixed_taskfix_sid_only"],
        "Instruments Dual-Task Family vs References",
        ASSET_DIR / "dual_task_vs_references_curves.png",
        sft,
        legend_cols=4,
    )
    plot_single_metric(
        df,
        SPECS,
        PROMISING_CANDIDATE_KEYS,
        "NDCG@10",
        "Instruments Promising Candidates: NDCG@10",
        ASSET_DIR / "promising_candidates_ndcg10.png",
        sft,
        legend_cols=4,
    )
    plot_single_metric(
        df,
        SPECS,
        PROMISING_CANDIDATE_KEYS,
        "HR@10",
        "Instruments Promising Candidates: HR@10",
        ASSET_DIR / "promising_candidates_hr10.png",
        sft,
        legend_cols=4,
    )
    plot_single_metric(
        df,
        SPECS,
        PROMISING_CANDIDATE_KEYS,
        "NDCG@50",
        "Instruments Promising Candidates: NDCG@50",
        ASSET_DIR / "promising_candidates_ndcg50.png",
        sft,
        legend_cols=4,
    )
    plot_single_metric(
        df,
        SPECS,
        PROMISING_CANDIDATE_KEYS,
        "HR@50",
        "Instruments Promising Candidates: HR@50",
        ASSET_DIR / "promising_candidates_hr50.png",
        sft,
        legend_cols=4,
    )
    plot_fixed_hint_bug_depth_distribution(ASSET_DIR / "fixed-hint-bug-task-depth-distribution.png")

    print(f"asset_dir={ASSET_DIR}")
    print(f"metadata_json={ASSET_DIR / 'instrument_variants_metadata.json'}")


if __name__ == "__main__":
    main()
