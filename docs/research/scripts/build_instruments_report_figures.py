#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from instruments_plot_lib import (
    VariantSpec,
    build_best_summary,
    build_dataframe,
    export_metadata,
    load_eval_wandb_urls,
    load_sft_reference,
    plot_metric_grid,
    save_csv,
)


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "instruments-report"

SPECS = [
    VariantSpec(
        key="rule_only",
        label="RL rule-only",
        model_dir="Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#1b7f79",
        marker="o",
    ),
    VariantSpec(
        key="dynamic_sid_only",
        label="RL dynamic sid-only",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#4e79a7",
        marker="s",
        epoch_max_step=2652,
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-only.sh",
    ),
    VariantSpec(
        key="dynamic_gather_fix",
        label="RL dynamic gather-fix",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#12355b",
        marker="o",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh",
    ),
    VariantSpec(
        key="ranking_dynamic",
        label="RL ranking dynamic",
        model_dir="Instruments-grec-grpo-ranking-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#7b8fa1",
        marker="^",
        linestyle="--",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-ranking-dynamic-hint.sh",
    ),
    VariantSpec(
        key="fixed_taskfix",
        label="RL fixed taskfix",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        color="#d88c3a",
        marker="o",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint.sh",
    ),
    VariantSpec(
        key="fixed_taskfix_sid_only",
        label="RL fixed taskfix sid-only",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495",
        color="#e4572e",
        marker="s",
        epoch_max_step=2652,
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-only.sh",
    ),
    VariantSpec(
        key="fixed_old",
        label="RL fixed old",
        model_dir="Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#7d5a99",
        marker="D",
        linestyle="-.",
    ),
    VariantSpec(
        key="max1",
        label="RL dynamic max1",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#0a9396",
        marker="P",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-max1.sh",
    ),
    VariantSpec(
        key="hintce_batch_mean",
        label="RL fixed CE batch-mean",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-sft495",
        color="#8c6a43",
        marker="o",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh",
    ),
    VariantSpec(
        key="hintce_token_mean",
        label="RL fixed CE token-mean",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-2-sft495",
        color="#c77d11",
        marker="s",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh",
    ),
    VariantSpec(
        key="hintce_coef_005",
        label="RL fixed CE coef=0.005",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
        color="#bc3908",
        marker="^",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh",
    ),
    VariantSpec(
        key="single_hint_mixed",
        label="RL single-hint mixed",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-hint-only-mixed-sft495",
        color="#c1128d",
        marker="P",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-hint-only-mixed.sh",
    ),
    VariantSpec(
        key="dynamic_dual_task",
        label="RL dynamic dual-task",
        model_dir="Instruments-grec-grpo-rule-only-dynamic-hint-sid-title-desc-qwen2.5-3b-qwen4B-4-256-from-sft495",
        color="#3d5a80",
        marker="o",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-title-desc.sh",
    ),
    VariantSpec(
        key="fixed_dual_task",
        label="RL fixed dual-task",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-title-desc-sft495",
        color="#8d99ae",
        marker="s",
        epoch_max_step_ref_key="dynamic_dual_task",
        launcher_path="hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh",
    ),
]


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    eval_urls = load_eval_wandb_urls()
    export_metadata(ASSET_DIR / "instrument_variants_metadata.json", SPECS, eval_urls)

    df = build_dataframe(SPECS)
    best_df = build_best_summary(df)
    sft = load_sft_reference()

    save_csv(ASSET_DIR / "all_variant_checkpoint_metrics.csv", df)
    save_csv(ASSET_DIR / "all_variant_best_summary.csv", best_df)

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
        ["fixed_taskfix", "hintce_batch_mean", "hintce_token_mean", "hintce_coef_005"],
        "Instruments Fixed Hint CE Scaling",
        ASSET_DIR / "ce_scaling_three_variants_curves.png",
        sft,
        legend_cols=4,
    )
    plot_metric_grid(
        df,
        SPECS,
        ["rule_only", "dynamic_gather_fix", "fixed_old", "fixed_taskfix_sid_only", "single_hint_mixed"],
        "Instruments Single-Hint Mixed vs Baselines",
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

    print(f"asset_dir={ASSET_DIR}")
    print(f"metadata_json={ASSET_DIR / 'instrument_variants_metadata.json'}")


if __name__ == "__main__":
    main()
