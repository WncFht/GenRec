#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
ASSET_DIR = TABLE_DIR / "assets"
OUTPUT_TEX = TABLE_DIR / "genrec-only-tables.tex"

METRICS = [
    "HR@1",
    "HR@5",
    "HR@10",
    "HR@20",
    "HR@50",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
]

SFT_CURVE_METRICS = [
    "HR@1",
    "HR@3",
    "HR@5",
    "HR@10",
    "HR@20",
    "HR@50",
    "NDCG@1",
    "NDCG@3",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
]

HEADLINE_CURVE_METRICS = (
    "NDCG@10",
    "HR@10",
    "NDCG@50",
    "HR@50",
)


@dataclass(frozen=True)
class VariantSpec:
    column_name: str
    model_dir: str
    is_rl: bool = False
    first_epoch_max_step: int | None = None
    curve_total_max_step: int | None = None


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    variants: tuple[VariantSpec, ...]
    caption: str
    label: str
    rl_total_max_step: int
    overall_best_variants: tuple[VariantSpec, ...] | None = None
    num_train_epochs: float = 2.0
    curve_mode: str = "rl_vs_sft"


@dataclass(frozen=True)
class CurveGroupSpec:
    dataset: str
    title: str
    asset_name: str
    figure_label: str
    caption: str
    table_caption: str
    table_label: str
    sft_model_dir: str
    total_max_step: int
    variants: tuple[VariantSpec, ...]
    num_train_epochs: float = 2.0
    curve_metrics: tuple[str, ...] = HEADLINE_CURVE_METRICS


@dataclass(frozen=True)
class LossDesignSpec:
    group_spec: CurveGroupSpec
    full_sequence_launcher: str
    summary: str


DATASET_SPECS = (
    DatasetSpec(
        dataset="Instruments",
        variants=(
            VariantSpec("GenRec(sft)", "Instruments-grec-sft-qwen4B-4-256-dsz0"),
            VariantSpec(
                "GenRec(rule)",
                "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(ranking)",
                "Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(fixed)",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.005)",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
        ),
        overall_best_variants=(
            VariantSpec("GenRec(sft)", "Instruments-grec-sft-qwen4B-4-256-dsz0"),
            VariantSpec(
                "GenRec(rule)",
                "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(ranking)",
                "Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(fixed)",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.001)",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-2-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.005)",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.01)",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-4-sft495",
                is_rl=True,
                first_epoch_max_step=1663,
            ),
        ),
        caption="Instruments 上仅保留 GenRec 系列方法的结果。",
        label="tab:genrec-only-instruments",
        rl_total_max_step=3326,
    ),
    DatasetSpec(
        dataset="Games",
        variants=(
            VariantSpec("GenRec(sft)", "Games-grec-sft-qwen4B-4-256-dsz0"),
            VariantSpec(
                "GenRec(rule)",
                "Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896",
                is_rl=True,
                first_epoch_max_step=4376,
            ),
            VariantSpec(
                "GenRec(ranking)",
                "Games-grec-genrec-ndcg-from-sft",
                is_rl=True,
                first_epoch_max_step=4376,
            ),
            VariantSpec(
                "GenRec(fixed)",
                "Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896",
                is_rl=True,
                first_epoch_max_step=4376,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.005)",
                "Games-grec-genrec-fixed-ce-from-sft",
                is_rl=True,
                first_epoch_max_step=4376,
            ),
        ),
        caption="Games 上当前已测到的 GenRec 系列结果。",
        label="tab:genrec-only-games",
        rl_total_max_step=8752,
    ),
    DatasetSpec(
        dataset="Arts",
        variants=(
            VariantSpec("GenRec(sft)", "Arts-grec-sft-qwen4B-4-256-dsz0"),
            VariantSpec(
                "GenRec(rule)",
                "Arts-grec-genrec-rule-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(ranking)",
                "Arts-grec-genrec-ndcg-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(fixed)",
                "Arts-grec-genrec-fixed-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.005)",
                "Arts-grec-genrec-fixed-ce-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
        ),
        overall_best_variants=(
            VariantSpec("GenRec(sft)", "Arts-grec-sft-qwen4B-4-256-dsz0"),
            VariantSpec(
                "GenRec(rule)",
                "Arts-grec-genrec-rule-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(ranking)",
                "Arts-grec-genrec-ndcg-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(fixed)",
                "Arts-grec-genrec-fixed-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.001)",
                "Arts-grec-genrec-fixed-ce-0001-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.005)",
                "Arts-grec-genrec-fixed-ce-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.01)",
                "Arts-grec-genrec-fixed-ce-001-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
            VariantSpec(
                "GenRec(fixed + ce0.1)",
                "Arts-grec-genrec-fixed-ce-01-from-sft",
                is_rl=True,
                first_epoch_max_step=2103,
            ),
        ),
        caption="Arts 上当前已测到的 GenRec 系列结果。",
        label="tab:genrec-only-arts",
        rl_total_max_step=4206,
    ),
)

OVERALL_BASELINE_VARIANTS = {
    "Instruments": (
        VariantSpec("Caser", "Instruments-caser-pytorch/checkpoint-best"),
        VariantSpec("GRU4Rec", "Instruments-gru4rec-pytorch-official/checkpoint-best"),
        VariantSpec("BERT4Rec", "Instruments-bert4rec-vanilla/checkpoint-best"),
        VariantSpec("SASRec", "Instruments-sasrec-recsys23/checkpoint-best"),
        VariantSpec("TIGER", "Instruments-tiger-grec-h50-reverse/checkpoint-best"),
    ),
    "Games": (
        VariantSpec("Caser", "Games-caser-pytorch/checkpoint-best"),
        VariantSpec("GRU4Rec", "Games-gru4rec-pytorch-official/checkpoint-best"),
        VariantSpec("BERT4Rec", "Games-bert4rec-vanilla/checkpoint-best"),
        VariantSpec("SASRec", "Games-sasrec-recsys23/checkpoint-best"),
        VariantSpec("TIGER", "Games-tiger-grec-h50-reverse/checkpoint-best"),
    ),
    "Arts": (
        VariantSpec("Caser", "Arts-caser-pytorch/checkpoint-best"),
        VariantSpec("GRU4Rec", "Arts-gru4rec-pytorch-official/checkpoint-best"),
        VariantSpec("BERT4Rec", "Arts-bert4rec-vanilla/checkpoint-best"),
        VariantSpec("SASRec", "Arts-sasrec-recsys23/checkpoint-best"),
        VariantSpec("TIGER", "Arts-tiger-grec-h50-reverse/checkpoint-best"),
    ),
}

OVERALL_GENREC_VARIANTS = {
    "Instruments": (
        VariantSpec("LC-Rec", "Instruments-grec-sft-qwen4B-4-256-dsz0"),
        VariantSpec(
            "MiniOnerec",
            "Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495",
            is_rl=True,
            first_epoch_max_step=1663,
        ),
        VariantSpec(
            "GenRec(rule)",
            "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
            is_rl=True,
            first_epoch_max_step=1663,
        ),
        VariantSpec(
            "GenRec(fixed + ce0.005)",
            "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
            is_rl=True,
            first_epoch_max_step=1663,
        ),
    ),
    "Games": (
        VariantSpec("LC-Rec", "Games-grec-sft-qwen4B-4-256-dsz0"),
        VariantSpec(
            "MiniOnerec",
            "Games-grec-genrec-ndcg-from-sft",
            is_rl=True,
            first_epoch_max_step=4376,
        ),
        VariantSpec(
            "GenRec(rule)",
            "Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896",
            is_rl=True,
            first_epoch_max_step=4376,
        ),
        VariantSpec(
            "GenRec(fixed + ce0.005)",
            "Games-grec-genrec-fixed-ce-from-sft",
            is_rl=True,
            first_epoch_max_step=4376,
        ),
    ),
    "Arts": (
        VariantSpec("LC-Rec", "Arts-grec-sft-qwen4B-4-256-dsz0"),
        VariantSpec(
            "MiniOnerec",
            "Arts-grec-genrec-ndcg-from-sft",
            is_rl=True,
            first_epoch_max_step=2103,
        ),
        VariantSpec(
            "GenRec(rule)",
            "Arts-grec-genrec-rule-from-sft",
            is_rl=True,
            first_epoch_max_step=2103,
        ),
        VariantSpec(
            "GenRec(fixed + ce0.005)",
            "Arts-grec-genrec-fixed-ce-from-sft",
            is_rl=True,
            first_epoch_max_step=2103,
        ),
    ),
}

VARIANT_STYLES = {
    "GenRec(sft)": {"color": "#4B5563", "marker": "o"},
    "GenRec(rule)": {"color": "#9C755F", "marker": "s"},
    "GenRec(ranking)": {"color": "#4E79A7", "marker": "o"},
    "GenRec(fixed)": {"color": "#59A14F", "marker": "^"},
    "GenRec(fixed + ce0.005)": {"color": "#E15759", "marker": "D"},
    "GenRec(fixed + ce0.1)": {"color": "#F28E2B", "marker": "P"},
    "Suffix-only GRPO": {"color": "#9C755F", "marker": "s"},
    "Prefix SFT + suffix-only GRPO": {"color": "#E15759", "marker": "D"},
    "Full-sequence SFT + GRPO": {"color": "#4E79A7", "marker": "o"},
    "Rule-only baseline": {"color": "#9C755F", "marker": "s"},
    "Ours (3-task)": {"color": "#59A14F", "marker": "^"},
    "Ours (fixed hint)": {"color": "#59A14F", "marker": "^"},
    "Adaptive hinting": {"color": "#4E79A7", "marker": "o"},
    "Dynamic hint": {"color": "#4E79A7", "marker": "o"},
    "Fixed first-token hint": {"color": "#E15759", "marker": "D"},
    "Fixed hint max1": {"color": "#E15759", "marker": "D"},
    "Adaptive first-token hint": {"color": "#76B7B2", "marker": "s"},
    "Adaptive hinting max1": {"color": "#F28E2B", "marker": "P"},
    "Dynamic hint max1": {"color": "#F28E2B", "marker": "P"},
    "Fixed(no CE)": {"color": "#9C755F", "marker": "s"},
    "CE=0.001": {"color": "#4E79A7", "marker": "o"},
    "CE=0.005": {"color": "#E15759", "marker": "D"},
    "CE=0.01": {"color": "#59A14F", "marker": "^"},
    "CE=0.1": {"color": "#F28E2B", "marker": "P"},
    "Fixed(taskfix)": {"color": "#F4A261", "marker": "^"},
    "Fixed(sid-only)": {"color": "#E76F51", "marker": "D"},
    "Fixed(sid+title+desc)": {"color": "#8D99AE", "marker": "o"},
}

FIXED_HINT_TASK_VARIANTS = (
    VariantSpec(
        "Fixed(taskfix)",
        "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        is_rl=True,
        first_epoch_max_step=1663,
        curve_total_max_step=3326,
    ),
    VariantSpec(
        "Fixed(sid-only)",
        "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495",
        is_rl=True,
        first_epoch_max_step=1326,
        curve_total_max_step=2652,
    ),
    VariantSpec(
        "Fixed(sid+title+desc)",
        "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-title-desc-sft495",
        is_rl=True,
        first_epoch_max_step=1506,
        curve_total_max_step=3012,
    ),
)

FIXED_HINT_TASK_SFT_MODEL_DIR = "Instruments-grec-sft-qwen4B-4-256-dsz0"
FIXED_HINT_TASK_SECTION_TITLE = "Instruments Fixed-Hint Task Variants"
FIXED_HINT_TASK_SECTION_INTRO = (
    r"这一节单独抽出三个 fixed-hint taskfix 变体：默认 \texttt{taskfix}、"
    r"\texttt{sid-only}、以及 \texttt{sid-title-desc}。"
    r"三列都在各自完整训练轨迹上按 \texttt{NDCG@10} 选出唯一 best checkpoint；"
    r"下表整行指标与下图曲线中的颜色一一对应。"
)
FIXED_HINT_TASK_TABLE_CAPTION = r"Instruments 上三个 fixed-hint taskfix 变体的整体 best 对比。"
FIXED_HINT_TASK_TABLE_LABEL = "tab:genrec-only-instruments-fixed-task-variants"
FIXED_HINT_TASK_CURVE_ASSET_NAME = "genrec-only-instruments-fixed-task-variants-curves.png"
FIXED_HINT_TASK_FIGURE_LABEL = "fig:genrec-only-instruments-fixed-task-variants-curves"

PREFIX_HINT_SIGNAL_VARIANTS = (
    VariantSpec("GenRec(sft)", "Instruments-grec-sft-qwen4B-4-256-dsz0"),
    VariantSpec(
        "Rule-only baseline",
        "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
        is_rl=True,
    ),
    VariantSpec(
        "Adaptive hinting",
        "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495",
        is_rl=True,
        curve_total_max_step=3326,
    ),
    VariantSpec(
        "Ours (3-task)",
        "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        is_rl=True,
        curve_total_max_step=3326,
    ),
)

PREFIX_HINT_2X2_SPEC = CurveGroupSpec(
    dataset="Instruments",
    title="Instruments hint comparison",
    asset_name="genrec-only-instruments-hint-comparison-curves.png",
    figure_label="fig:genrec-only-instruments-hint-comparison-curves",
    caption=(
        r"Instruments 上 \texttt{Ours(fixed hint)}、\texttt{dynamic hint}、"
        r"\texttt{dynamic hint max1} 的完整曲线对比。"
        r"图中统一展示 9 个指标；横轴统一使用 epoch，其中"
        r" full hint / dynamic 主线按 \texttt{3326 step = 2 epoch} 归一化，"
        r"\texttt{dynamic hint max1} 也按 \texttt{3326 step = 2 epoch} 归一化。"
    ),
    table_caption=r"Instruments 上 \texttt{Ours(fixed hint)}、\texttt{dynamic hint} 与 \texttt{dynamic hint max1} 的对比。",
    table_label="tab:genrec-only-instruments-hint-comparison",
    sft_model_dir="Instruments-grec-sft-qwen4B-4-256-dsz0",
    total_max_step=3326,
    variants=(
        VariantSpec(
            "Ours (fixed hint)",
            "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
            is_rl=True,
            first_epoch_max_step=1663,
            curve_total_max_step=3326,
        ),
        VariantSpec(
            "Dynamic hint",
            "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495",
            is_rl=True,
            first_epoch_max_step=1663,
            curve_total_max_step=3326,
        ),
        VariantSpec(
            "Dynamic hint max1",
            "Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495",
            is_rl=True,
            first_epoch_max_step=1663,
            curve_total_max_step=3326,
        ),
    ),
    curve_metrics=tuple(METRICS),
)

LOSS_DESIGN_SPECS = (
    LossDesignSpec(
        group_spec=CurveGroupSpec(
            dataset="Instruments",
            title="Instruments loss design",
            asset_name="genrec-only-instruments-loss-design-curves.png",
            figure_label="fig:genrec-only-instruments-loss-design-curves",
            caption=(
                r"Instruments 上 loss design 对比曲线。图中同时比较 \texttt{suffix-only GRPO}、"
                r"\texttt{prefix SFT + suffix-only GRPO} 与 \texttt{full-sequence SFT + GRPO}；"
                r"横轴按 \texttt{3326 step = 2 epoch} 归一化；"
                r"虚线表示 \texttt{GenRec(sft)} 的整体 best，竖向点线表示第一个 epoch 的 cutoff。"
                r"如果某条线当前只同步到少量 checkpoint，则图中会表现为短线或单点。"
            ),
            table_caption=r"RQ3 中 Instruments loss design 的当前状态表。",
            table_label="tab:genrec-only-rq3-loss-design-instruments",
            sft_model_dir="Instruments-grec-sft-qwen4B-4-256-dsz0",
            total_max_step=3326,
            variants=(
                VariantSpec(
                    "Suffix-only GRPO",
                    "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
                    is_rl=True,
                    first_epoch_max_step=1663,
                    curve_total_max_step=3326,
                ),
                VariantSpec(
                    "Prefix SFT + suffix-only GRPO",
                    "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
                    is_rl=True,
                    first_epoch_max_step=1663,
                    curve_total_max_step=3326,
                ),
                VariantSpec(
                    "Full-sequence SFT + GRPO",
                    "Instruments-grec-genrec-fixed-full-sequence-sft-from-sft",
                    is_rl=True,
                    first_epoch_max_step=1663,
                    curve_total_max_step=3326,
                ),
            ),
            curve_metrics=tuple(METRICS),
        ),
        full_sequence_launcher="hope/Instruments-genrec/rl_fixed_full_sequence_sft.sh",
        summary=(
            r"回到完整 2 epoch 的 best-checkpoint 口径后，\texttt{prefix SFT + suffix-only GRPO} 仍然是最稳的一条线："
            r"它拿到最高的 \texttt{HR@1 / HR@50} 与全部 \texttt{NDCG} 指标。"
            r"\texttt{suffix-only GRPO} 只在 \texttt{HR@5 / HR@10 / HR@20} 上保有优势，"
            r"\texttt{full-sequence SFT + GRPO} 则整体落后于前两者。"
            r"因此在当前结果里，最有性价比的 loss design 仍然是 prefix-SFT 这一条。"
        ),
    ),
    LossDesignSpec(
        group_spec=CurveGroupSpec(
            dataset="Arts",
            title="Arts loss design",
            asset_name="genrec-only-arts-loss-design-curves.png",
            figure_label="fig:genrec-only-arts-loss-design-curves",
            caption=(
                r"Arts 上 loss design 对比曲线。图中同时比较 \texttt{suffix-only GRPO}、"
                r"\texttt{prefix SFT + suffix-only GRPO} 与 \texttt{full-sequence SFT + GRPO}；"
                r"横轴按 \texttt{4206 step = 2 epoch} 归一化；"
                r"虚线表示 \texttt{GenRec(sft)} 的整体 best，竖向点线表示第一个 epoch 的 cutoff。"
                r"如果某条线当前只同步到少量 checkpoint，则图中会表现为短线或单点。"
            ),
            table_caption=r"RQ3 中 Arts loss design 的当前状态表。",
            table_label="tab:genrec-only-rq3-loss-design-arts",
            sft_model_dir="Arts-grec-sft-qwen4B-4-256-dsz0",
            total_max_step=4206,
            variants=(
                VariantSpec(
                    "Suffix-only GRPO",
                    "Arts-grec-genrec-fixed-from-sft",
                    is_rl=True,
                    first_epoch_max_step=2103,
                    curve_total_max_step=4206,
                ),
                VariantSpec(
                    "Prefix SFT + suffix-only GRPO",
                    "Arts-grec-genrec-fixed-ce-from-sft",
                    is_rl=True,
                    first_epoch_max_step=2103,
                    curve_total_max_step=4206,
                ),
                VariantSpec(
                    "Full-sequence SFT + GRPO",
                    "Arts-grec-genrec-fixed-full-sequence-sft-from-sft",
                    is_rl=True,
                    first_epoch_max_step=2103,
                    curve_total_max_step=4206,
                ),
            ),
            curve_metrics=tuple(METRICS),
        ),
        full_sequence_launcher="hope/Arts-genrec/rl_fixed_full_sequence_sft.sh",
        summary=(
            r"Arts 上的 loss design 没有出现 Instruments 那种单边压制："
            r"\texttt{prefix SFT + suffix-only GRPO} 拿到最高的 \texttt{HR@5 / HR@10 / HR@20}，"
            r"\texttt{suffix-only GRPO} 则保住 \texttt{HR@1 / HR@50 / NDCG@50}，"
            r"两者在 \texttt{NDCG@10 / NDCG@20} 上打平。"
            r"\texttt{full-sequence SFT + GRPO} 目前仍明显落后，说明 Arts 这边最值得比较的仍然是 no-CE 与 prefix-SFT 两条线。"
        ),
    ),
)

CE_SCALING_SECTION_TITLE = "CE Coefficient Ablations"
CE_SCALING_SECTION_INTRO = (
    r"这一节单独比较 fixed family 里的 CE 系数。"
    r"Arts 侧四条 CE 线对应 \texttt{0.001 / 0.005 / 0.01 / 0.1}；"
    r" Instruments 侧的 \texttt{hintce-2 / hintce-3 / hintce-4} 也对应同样三档系数。"
)

CE_SCALING_GROUP_SPECS = (
    CurveGroupSpec(
        dataset="Arts",
        title="Arts fixed CE coefficients",
        asset_name="genrec-only-arts-fixed-ce-coefficients-curves.png",
        figure_label="fig:genrec-only-arts-fixed-ce-coefficients-curves",
        caption=(
            r"Arts 上 fixed CE 系数对比曲线。图中同时保留 \texttt{Fixed(no CE)} 基线，"
            r"另外四条 CE 线分别对应 \texttt{CE=0.001 / 0.005 / 0.01 / 0.1}；"
            r"横轴按 \texttt{4206 step = 2 epoch} 归一化；"
            r"虚线表示 \texttt{GenRec(sft)} 的整体 best，竖向点线表示第一个 epoch 的 cutoff。"
        ),
        table_caption=r"Arts 上 fixed CE 系数对比（含无 CE baseline）。",
        table_label="tab:genrec-only-arts-fixed-ce-coefficients",
        sft_model_dir="Arts-grec-sft-qwen4B-4-256-dsz0",
        total_max_step=4206,
        variants=(
            VariantSpec("Fixed(no CE)", "Arts-grec-genrec-fixed-from-sft", is_rl=True),
            VariantSpec("CE=0.001", "Arts-grec-genrec-fixed-ce-0001-from-sft", is_rl=True),
            VariantSpec("CE=0.005", "Arts-grec-genrec-fixed-ce-from-sft", is_rl=True),
            VariantSpec("CE=0.01", "Arts-grec-genrec-fixed-ce-001-from-sft", is_rl=True),
            VariantSpec("CE=0.1", "Arts-grec-genrec-fixed-ce-01-from-sft", is_rl=True),
        ),
        curve_metrics=tuple(METRICS),
    ),
    CurveGroupSpec(
        dataset="Instruments",
        title="Instruments fixed CE coefficients",
        asset_name="genrec-only-instruments-fixed-ce-coefficients-curves.png",
        figure_label="fig:genrec-only-instruments-fixed-ce-coefficients-curves",
        caption=(
            r"Instruments 上 fixed-hint CE 系数对比曲线。图中同时保留 \texttt{Fixed(no CE)} 基线，"
            r"另外三条线分别对应 \texttt{CE=0.001 / 0.005 / 0.01}；"
            r"横轴按 \texttt{3326 step = 2 epoch} 归一化；"
            r"虚线表示 \texttt{GenRec(sft)} 的整体 best，竖向点线表示第一个 epoch 的 cutoff。"
        ),
        table_caption=r"Instruments 上 fixed-hint CE 系数对比（含无 CE baseline）。",
        table_label="tab:genrec-only-instruments-fixed-ce-coefficients",
        sft_model_dir="Instruments-grec-sft-qwen4B-4-256-dsz0",
        total_max_step=3326,
        variants=(
            VariantSpec(
                "Fixed(no CE)",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
                is_rl=True,
            ),
            VariantSpec(
                "CE=0.001",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-2-sft495",
                is_rl=True,
            ),
            VariantSpec(
                "CE=0.005",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
                is_rl=True,
            ),
            VariantSpec(
                "CE=0.01",
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-4-sft495",
                is_rl=True,
            ),
        ),
        curve_metrics=tuple(METRICS),
    ),
)


def checkpoint_step(name: str) -> int:
    match = re.search(r"checkpoint-(\d+)", name)
    if not match:
        return -1
    return int(match.group(1))


def read_metrics_json(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    return {key: float(value) for key, value in data.items()}


def collect_metrics_dir(model_dir: str) -> list[tuple[str, dict[str, float]]]:
    root = RESULTS_DIR / model_dir
    if not root.exists():
        return []
    direct_metrics_path = root / "metrics.json"
    if direct_metrics_path.exists():
        return [(root.name, read_metrics_json(direct_metrics_path))]

    entries: list[tuple[str, dict[str, float]]] = []
    for ckpt_dir in sorted(root.glob("checkpoint-*"), key=lambda path: checkpoint_step(path.name)):
        metrics_path = ckpt_dir / "metrics.json"
        if metrics_path.exists():
            entries.append((ckpt_dir.name, read_metrics_json(metrics_path)))
    return entries


def resolve_model_dir_best(model_dir: str) -> tuple[str, dict[str, float]] | None:
    return select_best(collect_metrics_dir(model_dir))


def select_best(entries: list[tuple[str, dict[str, float]]]) -> tuple[str, dict[str, float]] | None:
    if not entries:
        return None
    return max(entries, key=lambda item: (item[1].get("NDCG@10", float("-inf")), checkpoint_step(item[0])))


def select_last(entries: list[tuple[str, dict[str, float]]]) -> tuple[str, dict[str, float]] | None:
    if not entries:
        return None
    return max(entries, key=lambda item: checkpoint_step(item[0]))


def checkpoint_epoch(step: int, max_step: int, num_train_epochs: float) -> float:
    if step <= 0 or max_step <= 0:
        return float("nan")
    return step / max_step * num_train_epochs


def maybe_value(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or math.isnan(value):
        return None
    return value


def fmt_metric(value: float | None) -> str:
    if value is None:
        return r"\textemdash"
    return f"{value:.4f}"


def fmt_checkpoint_name(checkpoint_name: str) -> str:
    return r"\texttt{" + checkpoint_name + "}"


def select_rl_first_epoch_best(
    variant: VariantSpec, entries: list[tuple[str, dict[str, float]]]
) -> tuple[str, dict[str, float]] | None:
    if not entries:
        return None
    if not variant.is_rl or variant.first_epoch_max_step is None:
        return select_best(entries)

    first_epoch_entries = [entry for entry in entries if checkpoint_step(entry[0]) <= variant.first_epoch_max_step]
    return select_best(first_epoch_entries)


def select_nearest_step(
    entries: list[tuple[str, dict[str, float]]], target_step: int
) -> tuple[str, dict[str, float]] | None:
    if not entries:
        return None

    return min(
        entries,
        key=lambda item: (abs(checkpoint_step(item[0]) - target_step), checkpoint_step(item[0])),
    )


def select_target_step_or_last_before(
    entries: list[tuple[str, dict[str, float]]], target_step: int
) -> tuple[str, dict[str, float]] | None:
    if not entries:
        return None

    exact_entries = [entry for entry in entries if checkpoint_step(entry[0]) == target_step]
    if exact_entries:
        return select_last(exact_entries)

    before_or_equal_entries = [entry for entry in entries if checkpoint_step(entry[0]) <= target_step]
    if before_or_equal_entries:
        return select_last(before_or_equal_entries)

    return select_last(entries)


def resolve_variants(
    variants: tuple[VariantSpec, ...],
    *,
    rl_first_epoch_only: bool = False,
) -> list[tuple[VariantSpec, str, dict[str, float]]]:
    resolved: list[tuple[VariantSpec, str, dict[str, float]]] = []
    for variant in variants:
        entries = collect_metrics_dir(variant.model_dir)
        best = select_rl_first_epoch_best(variant, entries) if rl_first_epoch_only else select_best(entries)
        if best is None:
            continue
        checkpoint_name, metrics = best
        resolved.append((variant, checkpoint_name, metrics))
    return resolved


def resolve_variants_keep_all(
    variants: tuple[VariantSpec, ...],
    *,
    rl_first_epoch_only: bool = False,
) -> list[tuple[VariantSpec, str | None, dict[str, float]]]:
    resolved: list[tuple[VariantSpec, str | None, dict[str, float]]] = []
    for variant in variants:
        entries = collect_metrics_dir(variant.model_dir)
        best = select_rl_first_epoch_best(variant, entries) if rl_first_epoch_only else select_best(entries)
        if best is None:
            resolved.append((variant, None, {}))
            continue
        checkpoint_name, metrics = best
        resolved.append((variant, checkpoint_name, metrics))
    return resolved


def resolve_variants_near_first_epoch(
    variants: tuple[VariantSpec, ...],
) -> list[tuple[VariantSpec, str, dict[str, float]]]:
    resolved: list[tuple[VariantSpec, str, dict[str, float]]] = []
    for variant in variants:
        entries = collect_metrics_dir(variant.model_dir)
        if variant.first_epoch_max_step is None:
            selected = select_best(entries)
        else:
            selected = select_nearest_step(entries, variant.first_epoch_max_step)
        if selected is None:
            continue
        checkpoint_name, metrics = selected
        resolved.append((variant, checkpoint_name, metrics))
    return resolved


def resolve_variants_at_target_step(
    variants: tuple[VariantSpec, ...],
    *,
    target_step: int | None = None,
    prefer_variant_curve_total: bool = True,
) -> list[tuple[VariantSpec, str, dict[str, float]]]:
    resolved: list[tuple[VariantSpec, str, dict[str, float]]] = []
    for variant in variants:
        entries = collect_metrics_dir(variant.model_dir)
        effective_target_step = (
            variant.curve_total_max_step if prefer_variant_curve_total and variant.curve_total_max_step else target_step
        )
        if effective_target_step is None:
            selected = select_last(entries)
        else:
            selected = select_target_step_or_last_before(entries, effective_target_step)
        if selected is None:
            continue
        checkpoint_name, metrics = selected
        resolved.append((variant, checkpoint_name, metrics))
    return resolved


def resolve_dataset(
    spec: DatasetSpec,
    *,
    variants: tuple[VariantSpec, ...] | None = None,
    rl_first_epoch_only: bool = False,
) -> list[tuple[VariantSpec, str, dict[str, float]]]:
    return resolve_variants(variants or spec.variants, rl_first_epoch_only=rl_first_epoch_only)


def resolve_dataset_keep_all(
    spec: DatasetSpec,
    *,
    variants: tuple[VariantSpec, ...] | None = None,
    rl_first_epoch_only: bool = False,
) -> list[tuple[VariantSpec, str | None, dict[str, float]]]:
    return resolve_variants_keep_all(variants or spec.variants, rl_first_epoch_only=rl_first_epoch_only)


def resolve_variant_best(
    spec: DatasetSpec,
    column_name: str,
    *,
    rl_first_epoch_only: bool = False,
) -> tuple[VariantSpec, str, dict[str, float]] | None:
    resolved = resolve_dataset(spec, rl_first_epoch_only=rl_first_epoch_only)
    for variant, checkpoint_name, metrics in resolved:
        if variant.column_name == column_name:
            return variant, checkpoint_name, metrics
    return None


def overall_variants_for(spec: DatasetSpec) -> tuple[VariantSpec, ...]:
    return OVERALL_BASELINE_VARIANTS.get(spec.dataset, ()) + OVERALL_GENREC_VARIANTS.get(spec.dataset, ())


def has_any_measured_variant(
    resolved_variants: list[tuple[VariantSpec, str | None, dict[str, float]]],
) -> bool:
    return any(bool(metrics) for _, _, metrics in resolved_variants)


def rl_first_epoch_limit(spec: DatasetSpec) -> int | None:
    limits = [
        variant.first_epoch_max_step for variant in spec.variants if variant.is_rl and variant.first_epoch_max_step
    ]
    if not limits:
        return None
    return max(limits)


def has_rl_variants(spec: DatasetSpec) -> bool:
    return any(variant.is_rl for variant in spec.variants)


def collect_variant_curve_points(
    variant: VariantSpec,
    *,
    total_max_step: int | None = None,
    num_train_epochs: float = 2.0,
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    curve_max_step = variant.curve_total_max_step or total_max_step
    for checkpoint_name, metrics in collect_metrics_dir(variant.model_dir):
        step = checkpoint_step(checkpoint_name)
        if step < 0:
            continue
        effective_max_step = curve_max_step or step
        points.append(
            {
                "checkpoint": checkpoint_name,
                "step": step,
                "epoch": checkpoint_epoch(step, effective_max_step, num_train_epochs),
                "metrics": metrics,
            }
        )
    return points


def collect_rl_curve_series(spec: DatasetSpec) -> list[tuple[VariantSpec, list[dict[str, object]]]]:
    series: list[tuple[VariantSpec, list[dict[str, object]]]] = []
    for variant in spec.variants:
        if not variant.is_rl:
            continue
        points = collect_variant_curve_points(
            variant,
            total_max_step=spec.rl_total_max_step,
            num_train_epochs=spec.num_train_epochs,
        )
        if points:
            series.append((variant, points))
    return series


def get_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required to build checkpoint curves. "
            "Run `uv run --no-project --with matplotlib python docs/baseline/table/build_genrec_only_tables.py` "
            "or install matplotlib in the active Python environment."
        ) from exc

    return plt


def save_figure(fig, asset_path: Path, dpi: int = 180) -> None:
    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    fig.savefig(asset_path, dpi=dpi, **save_kwargs)
    fig.savefig(asset_path.with_suffix(".pdf"), **save_kwargs)


def build_rl_curve_asset(spec: DatasetSpec) -> Path | None:
    sft_best = resolve_variant_best(spec, "GenRec(sft)")
    rl_series = collect_rl_curve_series(spec)
    if sft_best is None or not rl_series:
        return None

    _, sft_checkpoint, sft_metrics = sft_best
    plt = get_matplotlib_pyplot()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    asset_path = ASSET_DIR / f"genrec-only-{spec.dataset.lower()}-curves.png"

    fig, axes = plt.subplots(5, 2, figsize=(11.6, 14.0), sharex=True)
    axes_flat = list(axes.flat)

    for ax, metric in zip(axes_flat, METRICS):
        for variant, points in rl_series:
            xs: list[float] = []
            ys: list[float] = []
            for point in points:
                value = maybe_value(point["metrics"], metric)  # type: ignore[arg-type]
                if value is None:
                    continue
                xs.append(float(point["epoch"]))  # type: ignore[arg-type]
                ys.append(value)

            style = VARIANT_STYLES.get(variant.column_name, {"color": "#4B5563", "marker": "o"})
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4.5,
                label=variant.column_name,
            )

        sft_value = maybe_value(sft_metrics, metric)
        if sft_value is not None:
            ax.axhline(
                sft_value,
                linestyle="--",
                linewidth=1.4,
                color="#6B7280",
                label=f"GenRec(sft) best ({sft_checkpoint})",
            )

        ax.axvline(
            1.0,
            linestyle=":",
            linewidth=1.2,
            color="#111827",
            alpha=0.7,
            label="Epoch 1 cutoff",
        )
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_xlim(0.0, spec.num_train_epochs)
        ax.grid(alpha=0.22)

    for ax in axes_flat[len(METRICS) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label not in unique:
            unique[label] = handle

    legend_cols = 3
    legend_rows = math.ceil(len(unique) / legend_cols) if unique else 1
    legend_y = 1.012 if legend_rows > 1 else 0.992
    suptitle_y = 0.95 if legend_rows > 1 else 0.965
    tight_layout_top = 0.905 if legend_rows > 1 else 0.94

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_cols,
        frameon=False,
    )
    fig.suptitle(f"{spec.dataset} GenRec checkpoint curves by metric", y=suptitle_y)
    fig.tight_layout(rect=(0, 0, 1, tight_layout_top))
    save_figure(fig, asset_path, dpi=180)
    plt.close(fig)
    return asset_path


def build_sft_curve_asset(spec: DatasetSpec) -> Path | None:
    sft_best = resolve_variant_best(spec, "GenRec(sft)")
    if sft_best is None:
        return None

    variant, sft_checkpoint, sft_metrics = sft_best
    entries = collect_metrics_dir(variant.model_dir)
    if not entries:
        return None

    plt = get_matplotlib_pyplot()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    asset_path = ASSET_DIR / f"genrec-only-{spec.dataset.lower()}-sft-curves.png"
    style = VARIANT_STYLES["GenRec(sft)"]

    num_cols = 2
    num_rows = math.ceil(len(SFT_CURVE_METRICS) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(11.2, 15.0), sharex=True)
    axes_flat = list(axes.flat)

    for ax, metric in zip(axes_flat, SFT_CURVE_METRICS):
        xs: list[int] = []
        ys: list[float] = []
        for checkpoint_name, metrics in entries:
            value = maybe_value(metrics, metric)
            step = checkpoint_step(checkpoint_name)
            if value is None or step < 0:
                continue
            xs.append(step)
            ys.append(value)

        if not xs:
            ax.axis("off")
            continue

        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4.5,
            label="GenRec(sft)",
        )
        best_value = maybe_value(sft_metrics, metric)
        if best_value is not None:
            ax.axhline(
                best_value,
                linestyle="--",
                linewidth=1.4,
                color="#6B7280",
                label=f"SFT best ({sft_checkpoint})",
            )
        best_step = checkpoint_step(sft_checkpoint)
        if best_step >= 0:
            ax.axvline(
                best_step,
                linestyle=":",
                linewidth=1.2,
                color="#111827",
                alpha=0.7,
                label="Best checkpoint",
            )

        ax.set_title(metric)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel(metric)
        ax.set_xlim(0.0, max(xs))
        ax.grid(alpha=0.22)

    for ax in axes_flat[len(SFT_CURVE_METRICS) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label not in unique:
            unique[label] = handle

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(f"{spec.dataset} GenRec SFT checkpoint curves by metric", y=0.968)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, asset_path, dpi=180)
    plt.close(fig)
    return asset_path


def build_curve_asset(spec: DatasetSpec) -> Path | None:
    if spec.curve_mode == "sft_only":
        return build_sft_curve_asset(spec)
    return build_rl_curve_asset(spec)


def build_curve_assets() -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for spec in DATASET_SPECS:
        asset = build_curve_asset(spec)
        if asset is not None:
            assets[spec.dataset] = asset
    return assets


def build_fixed_hint_task_curve_asset() -> Path | None:
    sft_best = resolve_model_dir_best(FIXED_HINT_TASK_SFT_MODEL_DIR)
    if sft_best is None:
        return None

    sft_checkpoint, sft_metrics = sft_best
    series: list[tuple[VariantSpec, list[dict[str, object]]]] = []
    for variant in FIXED_HINT_TASK_VARIANTS:
        points = collect_variant_curve_points(variant, num_train_epochs=2.0)
        if points:
            series.append((variant, points))
    if not series:
        return None

    plt = get_matplotlib_pyplot()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    asset_path = ASSET_DIR / FIXED_HINT_TASK_CURVE_ASSET_NAME
    fig, axes = plt.subplots(5, 2, figsize=(11.6, 14.0), sharex=True)
    axes_flat = list(axes.flat)

    for ax, metric in zip(axes_flat, METRICS):
        for variant, points in series:
            xs: list[float] = []
            ys: list[float] = []
            for point in points:
                value = maybe_value(point["metrics"], metric)  # type: ignore[arg-type]
                if value is None:
                    continue
                xs.append(float(point["epoch"]))  # type: ignore[arg-type]
                ys.append(value)

            style = VARIANT_STYLES.get(variant.column_name, {"color": "#4B5563", "marker": "o"})
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4.5,
                label=variant.column_name,
            )

        sft_value = maybe_value(sft_metrics, metric)
        if sft_value is not None:
            ax.axhline(
                sft_value,
                linestyle="--",
                linewidth=1.4,
                color="#6B7280",
                label=f"GenRec(sft) best ({sft_checkpoint})",
            )

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_xlim(0.0, 2.0)
        ax.grid(alpha=0.22)

    for ax in axes_flat[len(METRICS) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label not in unique:
            unique[label] = handle

    legend_cols = 3
    legend_rows = math.ceil(len(unique) / legend_cols) if unique else 1
    legend_y = 1.012 if legend_rows > 1 else 0.992
    suptitle_y = 0.95 if legend_rows > 1 else 0.965
    tight_layout_top = 0.905 if legend_rows > 1 else 0.94

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_cols,
        frameon=False,
    )
    fig.suptitle("Instruments fixed-hint task variants", y=suptitle_y)
    fig.tight_layout(rect=(0, 0, 1, tight_layout_top))
    save_figure(fig, asset_path, dpi=180)
    plt.close(fig)
    return asset_path


def build_curve_group_asset(spec: CurveGroupSpec) -> Path | None:
    sft_best = resolve_model_dir_best(spec.sft_model_dir)
    if sft_best is None:
        return None

    sft_checkpoint, sft_metrics = sft_best
    series: list[tuple[VariantSpec, list[dict[str, object]]]] = []
    for variant in spec.variants:
        points = collect_variant_curve_points(
            variant,
            total_max_step=spec.total_max_step,
            num_train_epochs=spec.num_train_epochs,
        )
        if points:
            series.append((variant, points))
    if not series:
        return None

    plt = get_matplotlib_pyplot()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    asset_path = ASSET_DIR / spec.asset_name
    num_cols = 2
    num_rows = math.ceil(len(spec.curve_metrics) / num_cols)
    fig_height = 8.2 if len(spec.curve_metrics) <= 4 else 14.0
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(11.2, fig_height), sharex=True)
    axes_flat = list(axes.flat)

    for ax, metric in zip(axes_flat, spec.curve_metrics):
        for variant, points in series:
            xs: list[float] = []
            ys: list[float] = []
            for point in points:
                value = maybe_value(point["metrics"], metric)  # type: ignore[arg-type]
                if value is None:
                    continue
                xs.append(float(point["epoch"]))  # type: ignore[arg-type]
                ys.append(value)

            style = VARIANT_STYLES.get(variant.column_name, {"color": "#4B5563", "marker": "o"})
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4.5,
                label=variant.column_name,
            )

        sft_value = maybe_value(sft_metrics, metric)
        if sft_value is not None:
            ax.axhline(
                sft_value,
                linestyle="--",
                linewidth=1.4,
                color="#6B7280",
                label=f"GenRec(sft) best ({sft_checkpoint})",
            )
        ax.axvline(
            1.0,
            linestyle=":",
            linewidth=1.2,
            color="#111827",
            alpha=0.7,
            label="Epoch 1 cutoff",
        )

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_xlim(0.0, spec.num_train_epochs)
        ax.grid(alpha=0.22)

    for ax in axes_flat[len(spec.curve_metrics) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label not in unique:
            unique[label] = handle

    legend_cols = 3
    legend_rows = math.ceil(len(unique) / legend_cols) if unique else 1
    legend_y = 1.012 if legend_rows > 1 else 0.992
    suptitle_y = 0.95 if legend_rows > 1 else 0.965
    tight_layout_top = 0.905 if legend_rows > 1 else 0.94

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_cols,
        frameon=False,
    )
    fig.suptitle(spec.title, y=suptitle_y)
    fig.tight_layout(rect=(0, 0, 1, tight_layout_top))
    save_figure(fig, asset_path, dpi=180)
    plt.close(fig)
    return asset_path


def build_ce_scaling_assets() -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for spec in CE_SCALING_GROUP_SPECS:
        asset = build_curve_group_asset(spec)
        if asset is not None:
            assets[spec.figure_label] = asset
    return assets


def build_optional_curve_group_assets(*specs: CurveGroupSpec) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for spec in specs:
        asset = build_curve_group_asset(spec)
        if asset is not None:
            assets[spec.figure_label] = asset
    return assets


def best_and_second_columns_for_metric(
    resolved_variants: list[tuple[VariantSpec, str | None, dict[str, float]]], metric: str
) -> tuple[set[str], set[str]]:
    measured: list[tuple[str, float]] = []
    for variant, _, metrics in resolved_variants:
        value = maybe_value(metrics, metric)
        if value is not None:
            measured.append((variant.column_name, value))
    if not measured:
        return set(), set()

    unique_values = sorted({value for _, value in measured}, reverse=True)
    best_value = unique_values[0]
    second_value = unique_values[1] if len(unique_values) > 1 else None
    best_columns = {column_name for column_name, value in measured if abs(value - best_value) < 1e-12}
    second_columns = (
        {column_name for column_name, value in measured if abs(value - second_value) < 1e-12}
        if second_value is not None
        else set()
    )
    return best_columns, second_columns


def render_results_table(
    resolved_variants: list[tuple[VariantSpec, str | None, dict[str, float]]],
    *,
    caption: str,
    label: str,
    metrics_list: tuple[str, ...] | list[str] = METRICS,
) -> str:
    columns = [variant.column_name for variant, _, _ in resolved_variants]
    col_spec = "l" + "c" * len(columns)
    should_resize = len(columns) >= 5

    parts: list[str] = []
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{4pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.08}")
    if should_resize:
        parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{" + col_spec + r"}")
    parts.append(r"\toprule")
    parts.append("Metric & " + " & ".join(columns) + r" \\")
    parts.append(r"\midrule")

    for metric in metrics_list:
        best_columns, second_columns = best_and_second_columns_for_metric(resolved_variants, metric)
        rendered_cells = []
        for variant, _, metrics in resolved_variants:
            text = fmt_metric(maybe_value(metrics, metric))
            if variant.column_name in best_columns and text != r"\textemdash":
                text = r"\textbf{" + text + "}"
            elif variant.column_name in second_columns and text != r"\textemdash":
                text = r"\underline{" + text + "}"
            rendered_cells.append(text)
        parts.append(metric + " & " + " & ".join(rendered_cells) + r" \\")

    parts.append(r"\midrule")
    parts.append(
        "Selected ckpt & "
        + " & ".join(
            fmt_checkpoint_name(checkpoint_name) if checkpoint_name is not None else r"\textemdash"
            for _, checkpoint_name, _ in resolved_variants
        )
        + r" \\"
    )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}" + ("%" if should_resize else ""))
    if should_resize:
        parts.append(r"}")
    parts.append(r"\caption{" + caption + r"}")
    parts.append(r"\label{" + label + r"}")
    parts.append(r"\end{table}")
    parts.append("")
    return "\n".join(parts)


def render_heading(level: str, title: str) -> str:
    return rf"\{level}{{{title}}}"


def render_dataset_table(
    spec: DatasetSpec,
    resolved_variants: list[tuple[VariantSpec, str | None, dict[str, float]]],
    *,
    caption_suffix: str = "",
    label_suffix: str = "",
    caption_override: str | None = None,
) -> str:
    return render_results_table(
        resolved_variants,
        caption=caption_override or (spec.caption + caption_suffix),
        label=spec.label + label_suffix,
    )


def build_section(
    title: str,
    intro: str,
    *,
    rl_first_epoch_only: bool,
    caption_suffix: str = "",
    label_suffix: str = "",
    include_overall_baselines: bool = False,
) -> list[str]:
    parts: list[str] = []
    parts.append(render_heading("section", title))
    parts.append("")
    parts.append(intro)
    parts.append("")

    for spec in DATASET_SPECS:
        if rl_first_epoch_only and not has_rl_variants(spec):
            continue
        if include_overall_baselines:
            section_variants = overall_variants_for(spec)
            resolved_variants = resolve_dataset_keep_all(
                spec,
                variants=section_variants,
                rl_first_epoch_only=rl_first_epoch_only,
            )
            caption_override = (
                spec.dataset
                + r" 上 overall 结果：补入 \texttt{Caser / GRU4Rec / BERT4Rec / SASRec / TIGER} 后的当前对照。"
            )
        else:
            section_variants = spec.overall_best_variants if not rl_first_epoch_only else None
            resolved_variants = resolve_dataset(
                spec,
                variants=section_variants,
                rl_first_epoch_only=rl_first_epoch_only,
            )
            caption_override = None
        if has_any_measured_variant(resolved_variants):
            parts.append(render_heading("subsection", spec.dataset))
            parts.append("")
            parts.append(
                render_dataset_table(
                    spec,
                    resolved_variants,
                    caption_suffix=caption_suffix,
                    label_suffix=label_suffix,
                    caption_override=caption_override,
                )
            )

    return parts


def build_rl_first_epoch_intro() -> str:
    clauses = [r"这一节只对 RL 变体收紧选点范围："]
    for spec in DATASET_SPECS:
        limit = rl_first_epoch_limit(spec)
        if limit is None:
            continue
        clauses.append(rf" {spec.dataset} 的 RL 列只在 \texttt{{checkpoint step <= {limit}}} 内选 best；")
    clauses.append(
        r" 非 RL 的 \texttt{GenRec(sft)} 仍保持其整体 best。"
        r" 每个变体一旦按 \texttt{NDCG@10} 选出 checkpoint，整行指标与最后一行列出的 \texttt{ckpt} 都固定来自这个同一个 checkpoint。"
    )
    return "".join(clauses)


def render_curve_figure(spec: DatasetSpec, asset_path: Path) -> str:
    relative_asset_path = asset_path.relative_to(TABLE_DIR).as_posix()

    parts: list[str] = []
    parts.append(r"\begin{figure}[p]")
    parts.append(r"\centering")
    parts.append(r"\includegraphics[width=\textwidth]{" + relative_asset_path + r"}")
    if spec.curve_mode == "sft_only":
        sft_best = resolve_variant_best(spec, "GenRec(sft)")
        best_checkpoint = sft_best[1] if sft_best is not None else "best"
        parts.append(
            r"\caption{"
            + spec.dataset
            + r" 上 \texttt{GenRec(sft)} 的 checkpoint 曲线。图中统一展示 12 个指标："
            + r"\texttt{HR@1/3/5/10/20/50} 与 \texttt{NDCG@1/3/5/10/20/50}；"
            + r"横轴直接使用 checkpoint step；"
            + rf"虚线和竖向点线共同标出按 \texttt{{NDCG@10}} 选出的整体 best checkpoint（\texttt{{{best_checkpoint}}}）。"
            + r"}"
        )
        parts.append(r"\label{fig:genrec-only-" + spec.dataset.lower() + r"-sft-curves}")
    else:
        first_epoch_step = rl_first_epoch_limit(spec) or (spec.rl_total_max_step // 2)
        parts.append(
            r"\caption{"
            + spec.dataset
            + r" 上 GenRec RL 变体的完整 checkpoint 曲线。图中统一展示 9 个指标："
            + r"\texttt{HR@1/5/10/20/50} 与 \texttt{NDCG@5/10/20/50}；"
            + rf"横轴按 {spec.dataset} RL 主线的 \texttt{{{spec.rl_total_max_step} step = {spec.num_train_epochs:.0f} epoch}} 归一化；"
            + r"虚线表示 \texttt{GenRec(sft)} 的整体 best，"
            + rf"竖向点线表示第一个 epoch 的 cutoff（\texttt{{step <= {first_epoch_step}}}）。"
            + r"}"
        )
        parts.append(r"\label{fig:genrec-only-" + spec.dataset.lower() + r"-curves}")
    parts.append(r"\end{figure}")
    parts.append("")
    return "\n".join(parts)


def build_curve_section(curve_assets: dict[str, Path]) -> list[str]:
    parts: list[str] = []
    parts.append(render_heading("section", "Supplementary Full Curves"))
    parts.append("")
    parts.append(
        r"这一节保留完整 checkpoint 曲线，作为 RQ1--RQ4 正文之外的补充证据。"
        r" Instruments、Games 与 Arts 都保留 RL 全轨迹并标出 first-epoch 边界；"
        r" 各图中的虚线统一表示 \texttt{GenRec(sft)} 的整体 best。"
    )
    parts.append("")

    for spec in DATASET_SPECS:
        asset_path = curve_assets.get(spec.dataset)
        if asset_path is not None:
            parts.append(render_heading("subsection", spec.dataset))
            parts.append("")
            parts.append(render_curve_figure(spec, asset_path))

    return parts


def render_curve_group_figure(spec: CurveGroupSpec, asset_path: Path) -> str:
    relative_asset_path = asset_path.relative_to(TABLE_DIR).as_posix()

    parts: list[str] = []
    parts.append(r"\begin{figure}[p]")
    parts.append(r"\centering")
    parts.append(r"\includegraphics[width=\textwidth]{" + relative_asset_path + r"}")
    parts.append(r"\caption{" + spec.caption + r"}")
    parts.append(r"\label{" + spec.figure_label + r"}")
    parts.append(r"\end{figure}")
    parts.append("")
    return "\n".join(parts)


def build_ce_scaling_section(ce_scaling_assets: dict[str, Path]) -> list[str]:
    parts: list[str] = []
    parts.append(render_heading("section", CE_SCALING_SECTION_TITLE))
    parts.append("")
    parts.append(CE_SCALING_SECTION_INTRO)
    parts.append("")

    for spec in CE_SCALING_GROUP_SPECS:
        resolved_variants = resolve_variants(spec.variants)
        if resolved_variants:
            parts.append(render_heading("subsection", spec.dataset))
            parts.append("")
            parts.append(
                render_results_table(
                    resolved_variants,
                    caption=spec.table_caption,
                    label=spec.table_label,
                )
            )
        asset_path = ce_scaling_assets.get(spec.figure_label)
        if asset_path is not None:
            parts.append(render_curve_group_figure(spec, asset_path))

    return parts


def build_rq2_section(rq2_assets: dict[str, Path]) -> list[str]:
    parts: list[str] = []
    parts.append(render_heading("section", "RQ2: Deep Analysis for Prefix Hint"))
    parts.append("")
    parts.append(
        r"这一节聚焦两个问题：其一，hint-conditioned training signal 是否真的有效；"
        r"其二，prefix hint strategy 在 fixed / adaptive 与 task-aware / first-token 两个轴上如何比较。"
    )
    parts.append("")

    parts.append(render_heading("subsection", "Hint-conditioned Training Signal"))
    parts.append("")
    parts.append(
        r"先把问题压缩到最核心的四条线："
        r"\texttt{GenRec(sft)}、\texttt{Rule-only baseline}、"
        r"\texttt{Adaptive hinting} 和 \texttt{Ours(3-task)}。"
    )
    parts.append("")
    signal_variants = resolve_variants(PREFIX_HINT_SIGNAL_VARIANTS)
    if signal_variants:
        parts.append(
            render_results_table(
                signal_variants,
                caption=r"Instruments 上 hint-conditioned training signal 的第一层证据。",
                label="tab:genrec-only-instruments-prefix-hint-signal",
            )
        )
    parts.append(
        r"\texttt{Rule-only baseline} 继续提供最高的无 hint top-10，"
        r"但 \texttt{Adaptive hinting} 和 \texttt{Ours(3-task)} 都把 coverage 拉回到了 SFT 以上；"
        r"这说明 prefix hint 的训练信号不是表面装饰，而是会真实改变 RL frontier 的形状。"
    )
    parts.append("")

    parts.append(render_heading("subsection", "Hint Strategy Comparison"))
    parts.append("")
    parts.append(
        r"这里把对照收缩为三条主线：\texttt{Ours(fixed hint)}、"
        r"\texttt{dynamic hint} 与 \texttt{dynamic hint max1}。"
        r"对应的目标是回答两个问题：fixed 是否优于 dynamic，以及 dynamic 在线索收缩到 max1 后会不会更强。"
    )
    parts.append("")
    prefix_variants = resolve_variants(PREFIX_HINT_2X2_SPEC.variants)
    if prefix_variants:
        parts.append(
            render_results_table(
                prefix_variants,
                caption=PREFIX_HINT_2X2_SPEC.table_caption,
                label=PREFIX_HINT_2X2_SPEC.table_label,
            )
        )
    parts.append(
        r"在完整 2 epoch 口径下，\texttt{Ours(fixed hint)} 仍然保住最高的"
        r" \texttt{HR@5 / HR@10 / HR@20 / HR@50}，"
        r"\texttt{dynamic hint max1} 则拿到最高的"
        r" \texttt{HR@1 / NDCG@5 / NDCG@10 / NDCG@20 / NDCG@50}。"
        r"\texttt{dynamic hint} 主线整体落在这两条线之后，说明真正有竞争力的 dynamic 版本来自 max1 收缩，而不是 full-hint dynamic 本身。"
    )
    parts.append("")
    parts.append(
        r"如果不让每条线各自挑 peak，而是统一读取 \texttt{2 epoch} 末尾的 checkpoint，"
        r"那么对比会更偏向 long-run stability。"
        r"下表保留原表不动，额外补一张固定读取终点 ckpt 的 readout。"
    )
    parts.append("")
    prefix_last_variants = resolve_variants_at_target_step(
        PREFIX_HINT_2X2_SPEC.variants,
        target_step=PREFIX_HINT_2X2_SPEC.total_max_step,
    )
    if prefix_last_variants:
        parts.append(
            render_results_table(
                prefix_last_variants,
                caption=(
                    r"Instruments 上 \texttt{Ours(fixed hint)}、\texttt{dynamic hint} 与 "
                    r"\texttt{dynamic hint max1} 在 2 epoch 末尾 checkpoint 的对比。"
                ),
                label="tab:genrec-only-instruments-hint-comparison-final-epoch",
            )
        )
    parts.append(
        r"固定到终点 checkpoint 后，\texttt{dynamic hint max1} 的领先基本消失；"
        r"\texttt{dynamic hint} 主线只保住 \texttt{HR@1 / NDCG@5 / NDCG@10}，"
        r"而 \texttt{Ours(fixed hint)} 仍然稳住其余 coverage 指标。"
        r"这说明 max1 的优势更像 early peak，而不是 2 epoch 末尾仍然稳定存在的优势。"
    )
    parts.append("")
    prefix_asset = rq2_assets.get(PREFIX_HINT_2X2_SPEC.figure_label)
    if prefix_asset is not None:
        parts.append(render_curve_group_figure(PREFIX_HINT_2X2_SPEC, prefix_asset))
    return parts


def render_loss_ablation_block(
    loss_design_spec: LossDesignSpec,
    asset_path: Path | None,
) -> str:
    spec = loss_design_spec.group_spec
    anchor_variant = next(
        (variant for variant in spec.variants if variant.column_name == "Prefix SFT + suffix-only GRPO"),
        None,
    )
    anchor_best: tuple[str, dict[str, float]] | None = None
    anchor_step: int | None = None
    if spec.dataset == "Instruments" and anchor_variant is not None:
        anchor_best = select_best(collect_metrics_dir(anchor_variant.model_dir))
        if anchor_best is not None:
            anchor_step = checkpoint_step(anchor_best[0])
    if spec.dataset == "Instruments" and anchor_step is not None and anchor_step > 0:
        resolved_variants = resolve_variants_at_target_step(
            spec.variants,
            target_step=anchor_step,
            prefer_variant_curve_total=False,
        )
    else:
        resolved_variants = resolve_variants(spec.variants)
    parts: list[str] = []
    parts.append(render_heading("paragraph", spec.dataset))
    parts.append("")
    if resolved_variants:
        parts.append(
            render_results_table(
                resolved_variants,
                caption=spec.table_caption,
                label=spec.table_label,
            )
        )

    full_sequence_variant = next(
        variant for variant in spec.variants if variant.column_name == "Full-sequence SFT + GRPO"
    )
    full_sequence_entries = collect_metrics_dir(full_sequence_variant.model_dir)
    full_sequence_best = select_best(full_sequence_entries)
    if full_sequence_best is None:
        full_sequence_status = r"当前 full-sequence 列还没有同步到可读的 \texttt{metrics.json}。"
    else:
        full_sequence_ckpt, _ = full_sequence_best
        if spec.dataset == "Instruments" and anchor_step is not None and anchor_step > 0:
            full_sequence_status = (
                r"当前 full-sequence 列已经接入 \texttt{results/"
                + full_sequence_variant.model_dir.replace("_", r"\_")
                + r"}；"
                + rf"当前共同步到 \texttt{{{len(full_sequence_entries)}}} 个 checkpoint，"
                + rf"这一版表格不再让三列各自挑 best，而是统一读取"
                + rf" \texttt{{Prefix SFT + suffix-only GRPO}} 当前选中的 \texttt{{checkpoint-{anchor_step}}}。"
            )
        else:
            full_sequence_status = (
                r"当前 full-sequence 列已经接入 \texttt{results/"
                + full_sequence_variant.model_dir.replace("_", r"\_")
                + r"}；"
                + rf"当前共同步到 \texttt{{{len(full_sequence_entries)}}} 个 checkpoint，"
                + rf"按 \texttt{{NDCG@10}} 选出的 readout 是 \texttt{{{full_sequence_ckpt}}}。"
            )
        if len(full_sequence_entries) == 1:
            full_sequence_status += r"因此曲线图里它暂时会表现为一个早期单点。"

    parts.append(
        full_sequence_status
        + r" 对应 launcher：\texttt{"
        + loss_design_spec.full_sequence_launcher.replace("_", r"\_")
        + r"}；"
        r"已完成的 prefix-SFT 列这里使用 \texttt{CE=0.005} 作为代表性 fixed+CE readout。"
    )
    parts.append("")
    parts.append(loss_design_spec.summary)
    parts.append("")
    if asset_path is not None:
        parts.append(render_curve_group_figure(spec, asset_path))
    return "\n".join(parts)


def build_rq3_section(fixed_hint_task_asset: Path | None, loss_design_assets: dict[str, Path]) -> list[str]:
    parts: list[str] = []
    parts.append(render_heading("section", "RQ3: Ablation Study"))
    parts.append("")
    parts.append(
        r"这一节把消融分成两层：训练任务范围，以及 loss 设计。"
        r"前者只比较当前已经跑完的 three-way fixed prefix variants；"
        r"后者则把 \texttt{fixed}、\texttt{fixed+CE} 和 full-sequence 线分别在"
        r" \texttt{Instruments / Arts} 两个数据集上并列到表格与曲线图里。"
    )
    parts.append("")

    parts.append(render_heading("subsection", "Training Task Scope"))
    parts.append("")
    parts.append(
        r"这一小节回到完整 2 epoch 里按 \texttt{NDCG@10} 选 best 的默认口径。"
        r"当前已经落地的三档训练任务范围是：默认 \texttt{taskfix}、"
        r"\texttt{sid-only}，以及两任务变体 \texttt{sid-title-desc}。"
    )
    parts.append("")
    resolved_variants = resolve_variants(FIXED_HINT_TASK_VARIANTS)
    if resolved_variants:
        parts.append(
            render_results_table(
                resolved_variants,
                caption=r"Instruments 上训练任务范围的 three-way ablation。",
                label="tab:genrec-only-rq3-task-scope",
            )
        )
    parts.append(
        r"回到完整 2 epoch 的 best-checkpoint 口径后，\texttt{sid-only} 重新成为最强主线："
        r"它拿到最高的 \texttt{HR@5 / HR@10 / HR@20} 与全部 \texttt{NDCG} 指标。"
        r"\texttt{taskfix} 只在 \texttt{HR@50} 上略占优，"
        r"\texttt{sid-title-desc} 则只保留 \texttt{HR@1} 的领先。"
        r"也就是说，长程训练结束后，最稳的 task scope 仍然是更窄、更聚焦的 \texttt{sid-only}。"
    )
    parts.append("")
    if fixed_hint_task_asset is not None:
        parts.append(render_fixed_hint_task_figure(fixed_hint_task_asset))

    parts.append(render_heading("subsection", "Loss Design"))
    parts.append("")
    parts.append(
        r"这一小节也回到完整 2 epoch 里按 \texttt{NDCG@10} 选 best 的默认口径。"
        r"loss 层面的比较只保留三种口径："
        r"\texttt{suffix-only GRPO}、\texttt{prefix SFT + suffix-only GRPO}、"
        r"以及 \texttt{Full-sequence SFT + GRPO}。"
        r"下面保留 Instruments 原表，同时把 Arts 的对应结果和曲线一并补进来。"
    )
    parts.append("")
    for loss_design_spec in LOSS_DESIGN_SPECS:
        parts.append(
            render_loss_ablation_block(
                loss_design_spec,
                loss_design_assets.get(loss_design_spec.group_spec.figure_label),
            )
        )
    return parts


def build_rq4_section(ce_scaling_assets: dict[str, Path]) -> list[str]:
    parts: list[str] = []
    parts.append(render_heading("section", "RQ4: CE Loss Influence"))
    parts.append("")
    parts.append(
        r"这一节专门分析 fixed CE coefficient 对后训练结果的影响。"
        r"正文只保留 \texttt{Instruments} 和 \texttt{Arts} 两个数据集上的"
        r" CE sweep；其中 \texttt{Arts} 额外补入 \texttt{CE=0.1}。"
    )
    parts.append("")

    for spec in CE_SCALING_GROUP_SPECS:
        resolved_variants = resolve_variants(spec.variants)
        if resolved_variants:
            parts.append(render_heading("subsection", spec.dataset))
            parts.append("")
            parts.append(
                render_results_table(
                    resolved_variants,
                    caption=spec.table_caption,
                    label=spec.table_label,
                )
            )
        asset_path = ce_scaling_assets.get(spec.figure_label)
        if asset_path is not None:
            parts.append(render_curve_group_figure(spec, asset_path))

    parts.append(render_heading("subsection", "Cross-Dataset Summary"))
    parts.append("")
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{4pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.08}")
    parts.append(r"\begin{tabular}{p{2.0cm} p{3.0cm} p{3.0cm} p{5.0cm}}")
    parts.append(r"\toprule")
    parts.append(r"Dataset & Best top-10 coefficient & Best coverage coefficient & Reading \\")
    parts.append(r"\midrule")
    parts.append(
        r"Instruments & \texttt{CE=0.01} & \texttt{CE=0.005} & 大系数继续抬高 long-run top-10，但 coverage 峰值仍出现在中档系数 \\"
    )
    parts.append(
        r"Arts & \texttt{CE=0.005} / no-CE tie & \texttt{CE=0.001} & 小系数更像 mild coverage regularizer，\texttt{CE=0.1} 则在 top-10 与 coverage 上都明显退化 \\"
    )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\caption{RQ4 的跨数据集 summary。}")
    parts.append(r"\label{tab:genrec-only-rq4-cross-dataset-summary}")
    parts.append(r"\end{table}")
    parts.append("")
    return parts


def render_fixed_hint_task_figure(asset_path: Path) -> str:
    relative_asset_path = asset_path.relative_to(TABLE_DIR).as_posix()
    sft_best = resolve_model_dir_best(FIXED_HINT_TASK_SFT_MODEL_DIR)
    sft_checkpoint = sft_best[0] if sft_best is not None else "best"

    parts: list[str] = []
    parts.append(r"\begin{figure}[p]")
    parts.append(r"\centering")
    parts.append(r"\includegraphics[width=\textwidth]{" + relative_asset_path + r"}")
    parts.append(
        r"\caption{"
        r"Instruments 上三个 fixed-hint taskfix 变体的完整 checkpoint 曲线。"
        r"图中统一展示 9 个指标：\texttt{HR@1/5/10/20/50} 与 \texttt{NDCG@5/10/20/50}；"
        r"横轴统一使用 epoch，并分别按 \texttt{3326 / 2652 / 3012 step = 2 epoch} 归一化；"
        + rf"虚线表示 \texttt{{GenRec(sft)}} 的整体 best（\texttt{{{sft_checkpoint}}}）。"
        + r"}"
    )
    parts.append(r"\label{" + FIXED_HINT_TASK_FIGURE_LABEL + r"}")
    parts.append(r"\end{figure}")
    parts.append("")
    return "\n".join(parts)


def build_fixed_hint_task_section(asset_path: Path | None) -> list[str]:
    parts: list[str] = []
    parts.append(render_heading("section", FIXED_HINT_TASK_SECTION_TITLE))
    parts.append("")
    parts.append(FIXED_HINT_TASK_SECTION_INTRO)
    parts.append("")

    resolved_variants = resolve_variants(FIXED_HINT_TASK_VARIANTS)
    if resolved_variants:
        parts.append(render_heading("subsection", "Overall Best Table"))
        parts.append("")
        parts.append(
            render_results_table(
                resolved_variants,
                caption=FIXED_HINT_TASK_TABLE_CAPTION,
                label=FIXED_HINT_TASK_TABLE_LABEL,
            )
        )

    if asset_path is not None:
        parts.append(render_heading("subsection", "Full Curves"))
        parts.append("")
        parts.append(render_fixed_hint_task_figure(asset_path))

    return parts


def build_document(
    curve_assets: dict[str, Path],
    fixed_hint_task_asset: Path | None,
    ce_scaling_assets: dict[str, Path],
    rq2_assets: dict[str, Path],
    rq3_assets: dict[str, Path],
) -> str:
    parts: list[str] = []
    parts.append(r"\documentclass[11pt,a4paper]{ctexart}")
    parts.append("")
    parts.append(r"\usepackage[margin=2.2cm]{geometry}")
    parts.append(r"\usepackage{booktabs}")
    parts.append(r"\usepackage{float}")
    parts.append(r"\usepackage{graphicx}")
    parts.append(r"\usepackage[unicode,hidelinks,bookmarksopen,bookmarksdepth=2]{hyperref}")
    parts.append("")
    parts.append(r"\begin{document}")
    parts.append("")
    parts.append(r"\setcounter{secnumdepth}{2}")
    parts.append("")
    parts.append(r"\section{GenRec-Only Tables}")
    parts.append("")
    parts.append(
        r"本文件由 \texttt{build\_genrec\_only\_tables.py} 自动生成。"
        r"本轮重排按 \texttt{RQ1--RQ4} 组织，只保留当前 index 下已测到的 GenRec 系列结果，"
        r"并把 full curves / first-epoch tables 统一放到 appendix。"
    )
    parts.append("")
    parts.append(r"\tableofcontents")
    parts.append(r"\newpage")
    parts.append("")

    parts.extend(
        build_section(
            "RQ1: Overall Performance",
            r"这一节沿用当前默认口径：每个变体都在其全部已同步 checkpoint 中按 \texttt{NDCG@10} 选出唯一 best checkpoint。"
            r" 在保留 GenRec 主线的同时，overall 里额外补入"
            r" \texttt{Caser / GRU4Rec / BERT4Rec / SASRec / TIGER}；"
            r" 其中缺失结果仍保留为空列，已同步结果直接读取当前 \texttt{results/} 里的 best readout。"
            r" GenRec 侧只保留 \texttt{MiniOnerec}、\texttt{LC-Rec}、\texttt{GenRec(rule)}"
            r" 和 \texttt{GenRec(fixed + ce0.005)}，从而把 overall frontier 压到一张更紧凑的对照表里。",
            rl_first_epoch_only=False,
            include_overall_baselines=True,
        )
    )
    parts.extend(build_rq2_section(rq2_assets))
    parts.extend(build_rq3_section(fixed_hint_task_asset, rq3_assets))
    parts.extend(build_rq4_section(ce_scaling_assets))
    parts.append(r"\clearpage")
    parts.append(r"\appendix")
    parts.append("")
    parts.extend(build_curve_section(curve_assets))
    parts.extend(
        build_section(
            "RL First-Epoch Best",
            build_rl_first_epoch_intro(),
            rl_first_epoch_only=True,
            caption_suffix=r"（RL 列仅在第一个 epoch 内按 \texttt{NDCG@10} 选 best）。",
            label_suffix="-rl-first-epoch",
        )
    )

    parts.append(r"\end{document}")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    curve_assets = build_curve_assets()
    fixed_hint_task_asset = build_fixed_hint_task_curve_asset()
    ce_scaling_assets = build_ce_scaling_assets()
    rq2_assets = build_optional_curve_group_assets(PREFIX_HINT_2X2_SPEC)
    rq3_assets = build_optional_curve_group_assets(*(spec.group_spec for spec in LOSS_DESIGN_SPECS))
    OUTPUT_TEX.write_text(
        build_document(curve_assets, fixed_hint_task_asset, ce_scaling_assets, rq2_assets, rq3_assets)
    )


if __name__ == "__main__":
    main()
