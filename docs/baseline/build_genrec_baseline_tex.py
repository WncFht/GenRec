#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "docs" / "baseline"
RESULTS_DIR = ROOT / "results"
INSTRUMENTS_REPORT_ASSET_DIR = ROOT / "docs" / "research" / "assets" / "instruments-report"
INSTRUMENTS_BEST_CSV = ROOT / "docs" / "research" / "assets" / "instruments-report" / "all_variant_best_summary.csv"
GAMES_SFT_BEST_CSV = ROOT / "docs" / "research" / "assets" / "games-report" / "games_sft_best_summary.csv"
GAMES_RL_BEST_CSV = ROOT / "docs" / "research" / "assets" / "games-report" / "games_rl_best_summary.csv"
ARTS_CKPT_CSV = ROOT / "docs" / "research" / "assets" / "arts-report" / "arts_checkpoint_metrics.csv"

MAIN_TEX = BASELINE_DIR / "genrec-baseline-main.tex"
APPENDIX_TEX = BASELINE_DIR / "genrec-baseline-appendix.tex"
MASTER_TEX = BASELINE_DIR / "genrec-baseline.tex"

ALIGNED_RULE_MODEL_DIR = "Instruments-grec-genrec-aligned-rule"
ALIGNED_SFT_MODEL_DIR = "Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu"
ALIGNED_RULE_NUM_EPOCHS = 2.0
ALIGNED_RULE_EPOCH_NORMALIZER_STEP = 3326
ALIGNED_RULE_CURVES_PNG = INSTRUMENTS_REPORT_ASSET_DIR / "genrec_aligned_rule_epoch_curves.png"
ALIGNED_RULE_METRICS_CSV = INSTRUMENTS_REPORT_ASSET_DIR / "genrec_aligned_rule_checkpoint_metrics.csv"


METRICS = ["HR@1", "HR@5", "HR@10", "HR@20", "HR@50", "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50"]

MAIN_TABLE_COLUMNS = [
    "Caser",
    "GRU4Rec",
    "BERT4Rec",
    "SASRec",
    "TIGER",
    "LC-Rec",
    "GenRec(sft)",
    "GenRec(rule)",
    "GenRec(ranking)",
    "GenRec(fixed)",
    "GenRec(fixed + ce0.005)",
]

PAPER_REFERENCE_METHODS = [
    "GRU4Rec",
    "BERT4Rec",
    "SASRec",
    "FDSA",
    r"S$^3$-Rec",
    "VQ-Rec",
    "MISSRec",
    "P5-CID",
    "VIP5",
    "TIGER",
    "MQL4GRec",
]

PAPER_REFERENCE_ROWS: list[tuple[str, list[tuple[str, dict[str, float | str]]]]] = [
    (
        "Instruments",
        [
            (
                "HR@1",
                {
                    "GRU4Rec": 0.0566,
                    "BERT4Rec": 0.0450,
                    "SASRec": 0.0318,
                    "FDSA": 0.0530,
                    r"S$^3$-Rec": 0.0339,
                    "VQ-Rec": 0.0502,
                    "MISSRec": 0.0723,
                    "P5-CID": 0.0512,
                    "VIP5": 0.0737,
                    "TIGER": 0.0754,
                    "MQL4GRec": 0.0833,
                    "Improv.": "+10.48\\%",
                },
            ),
            (
                "HR@5",
                {
                    "GRU4Rec": 0.0975,
                    "BERT4Rec": 0.0856,
                    "SASRec": 0.0946,
                    "FDSA": 0.0987,
                    r"S$^3$-Rec": 0.0937,
                    "VQ-Rec": 0.1062,
                    "MISSRec": 0.1089,
                    "P5-CID": 0.0839,
                    "VIP5": 0.0892,
                    "TIGER": 0.1007,
                    "MQL4GRec": 0.1115,
                    "Improv.": "+2.39\\%",
                },
            ),
            (
                "HR@10",
                {
                    "GRU4Rec": 0.1207,
                    "BERT4Rec": 0.1081,
                    "SASRec": 0.1233,
                    "FDSA": 0.1249,
                    r"S$^3$-Rec": 0.1123,
                    "VQ-Rec": 0.1357,
                    "MISSRec": 0.1361,
                    "P5-CID": 0.1119,
                    "VIP5": 0.1071,
                    "TIGER": 0.1221,
                    "MQL4GRec": 0.1375,
                    "Improv.": "+1.03\\%",
                },
            ),
            (
                "NDCG@5",
                {
                    "GRU4Rec": 0.0783,
                    "BERT4Rec": 0.0667,
                    "SASRec": 0.0654,
                    "FDSA": 0.0775,
                    r"S$^3$-Rec": 0.0693,
                    "VQ-Rec": 0.0796,
                    "MISSRec": 0.0797,
                    "P5-CID": 0.0678,
                    "VIP5": 0.0815,
                    "TIGER": 0.0882,
                    "MQL4GRec": 0.0977,
                    "Improv.": "+10.77\\%",
                },
            ),
            (
                "NDCG@10",
                {
                    "GRU4Rec": 0.0857,
                    "BERT4Rec": 0.0739,
                    "SASRec": 0.0746,
                    "FDSA": 0.0859,
                    r"S$^3$-Rec": 0.0743,
                    "VQ-Rec": 0.0891,
                    "MISSRec": 0.0880,
                    "P5-CID": 0.0704,
                    "VIP5": 0.0872,
                    "TIGER": 0.0950,
                    "MQL4GRec": 0.1060,
                    "Improv.": "+11.58\\%",
                },
            ),
        ],
    ),
    (
        "Arts",
        [
            (
                "HR@1",
                {
                    "GRU4Rec": 0.0365,
                    "BERT4Rec": 0.0289,
                    "SASRec": 0.0212,
                    "FDSA": 0.0380,
                    r"S$^3$-Rec": 0.0172,
                    "VQ-Rec": 0.0408,
                    "MISSRec": 0.0479,
                    "P5-CID": 0.0421,
                    "VIP5": 0.0474,
                    "TIGER": 0.0532,
                    "MQL4GRec": 0.0672,
                    "Improv.": "+26.32\\%",
                },
            ),
            (
                "HR@5",
                {
                    "GRU4Rec": 0.0817,
                    "BERT4Rec": 0.0697,
                    "SASRec": 0.0951,
                    "FDSA": 0.0832,
                    r"S$^3$-Rec": 0.0739,
                    "VQ-Rec": 0.1038,
                    "MISSRec": 0.1021,
                    "P5-CID": 0.0713,
                    "VIP5": 0.0704,
                    "TIGER": 0.0894,
                    "MQL4GRec": 0.1037,
                    "Improv.": "-",
                },
            ),
            (
                "HR@10",
                {
                    "GRU4Rec": 0.1088,
                    "BERT4Rec": 0.0922,
                    "SASRec": 0.1250,
                    "FDSA": 0.1190,
                    r"S$^3$-Rec": 0.1030,
                    "VQ-Rec": 0.1386,
                    "MISSRec": 0.1321,
                    "P5-CID": 0.0994,
                    "VIP5": 0.0859,
                    "TIGER": 0.1167,
                    "MQL4GRec": 0.1327,
                    "Improv.": "-",
                },
            ),
            (
                "NDCG@5",
                {
                    "GRU4Rec": 0.0602,
                    "BERT4Rec": 0.0502,
                    "SASRec": 0.0610,
                    "FDSA": 0.0583,
                    r"S$^3$-Rec": 0.0511,
                    "VQ-Rec": 0.0732,
                    "MISSRec": 0.0699,
                    "P5-CID": 0.0607,
                    "VIP5": 0.0586,
                    "TIGER": 0.0718,
                    "MQL4GRec": 0.0857,
                    "Improv.": "+17.08\\%",
                },
            ),
            (
                "NDCG@10",
                {
                    "GRU4Rec": 0.0690,
                    "BERT4Rec": 0.0575,
                    "SASRec": 0.0706,
                    "FDSA": 0.0695,
                    r"S$^3$-Rec": 0.0630,
                    "VQ-Rec": 0.0844,
                    "MISSRec": 0.0815,
                    "P5-CID": 0.0662,
                    "VIP5": 0.0635,
                    "TIGER": 0.0806,
                    "MQL4GRec": 0.0950,
                    "Improv.": "+12.56\\%",
                },
            ),
        ],
    ),
    (
        "Games",
        [
            (
                "HR@1",
                {
                    "GRU4Rec": 0.0140,
                    "BERT4Rec": 0.0115,
                    "SASRec": 0.0069,
                    "FDSA": 0.0163,
                    r"S$^3$-Rec": 0.0136,
                    "VQ-Rec": 0.0075,
                    "MISSRec": 0.0201,
                    "P5-CID": 0.0169,
                    "VIP5": 0.0173,
                    "TIGER": 0.0166,
                    "MQL4GRec": 0.0203,
                    "Improv.": "+1.00\\%",
                },
            ),
            (
                "HR@5",
                {
                    "GRU4Rec": 0.0544,
                    "BERT4Rec": 0.0426,
                    "SASRec": 0.0587,
                    "FDSA": 0.0614,
                    r"S$^3$-Rec": 0.0527,
                    "VQ-Rec": 0.0408,
                    "MISSRec": 0.0674,
                    "P5-CID": 0.0532,
                    "VIP5": 0.0480,
                    "TIGER": 0.0523,
                    "MQL4GRec": 0.0637,
                    "Improv.": "-",
                },
            ),
            (
                "HR@10",
                {
                    "GRU4Rec": 0.0895,
                    "BERT4Rec": 0.0725,
                    "SASRec": 0.0985,
                    "FDSA": 0.0988,
                    r"S$^3$-Rec": 0.0903,
                    "VQ-Rec": 0.0679,
                    "MISSRec": 0.1048,
                    "P5-CID": 0.0824,
                    "VIP5": 0.0758,
                    "TIGER": 0.0857,
                    "MQL4GRec": 0.1033,
                    "Improv.": "-",
                },
            ),
            (
                "NDCG@5",
                {
                    "GRU4Rec": 0.0341,
                    "BERT4Rec": 0.0270,
                    "SASRec": 0.0333,
                    "FDSA": 0.0389,
                    r"S$^3$-Rec": 0.0351,
                    "VQ-Rec": 0.0242,
                    "MISSRec": 0.0385,
                    "P5-CID": 0.0331,
                    "VIP5": 0.0328,
                    "TIGER": 0.0345,
                    "MQL4GRec": 0.0421,
                    "Improv.": "+8.23\\%",
                },
            ),
            (
                "NDCG@10",
                {
                    "GRU4Rec": 0.0453,
                    "BERT4Rec": 0.0366,
                    "SASRec": 0.0461,
                    "FDSA": 0.0509,
                    r"S$^3$-Rec": 0.0468,
                    "VQ-Rec": 0.0329,
                    "MISSRec": 0.0499,
                    "P5-CID": 0.0454,
                    "VIP5": 0.0418,
                    "TIGER": 0.0453,
                    "MQL4GRec": 0.0548,
                    "Improv.": "+7.66\\%",
                },
            ),
        ],
    ),
]


@dataclass
class RunSpec:
    dataset: str
    label: str
    column_name: str
    model_dir: str | None = None
    best_checkpoint: str | None = None
    pending: bool = False
    note: str | None = None


RUN_SPECS: list[RunSpec] = [
    RunSpec("Instruments", "Caser", "Caser", model_dir="Instruments-caser-pytorch/checkpoint-best"),
    RunSpec("Instruments", "GRU4Rec", "GRU4Rec", model_dir="Instruments-gru4rec-pytorch-official/checkpoint-best"),
    RunSpec(
        "Instruments",
        "BERT4Rec (RecSys23)",
        "BERT4Rec (RecSys23)",
        model_dir="Instruments-bert4rec-recsys23/checkpoint-best",
    ),
    RunSpec("Instruments", "BERT4Rec", "BERT4Rec", model_dir="Instruments-bert4rec-vanilla/checkpoint-best"),
    RunSpec("Instruments", "SASRec", "SASRec", model_dir="Instruments-sasrec-recsys23/checkpoint-best"),
    RunSpec("Instruments", "TIGER", "TIGER", model_dir="Instruments-tiger-grec-h50-reverse/checkpoint-best"),
    RunSpec("Instruments", "LC-Rec", "LC-Rec", model_dir="Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu"),
    RunSpec("Instruments", "GenRec (SFT)", "GenRec(sft)", model_dir="Instruments-grec-sft-qwen4B-4-256-dsz0"),
    RunSpec(
        "Instruments",
        "GenRec (rule)",
        "GenRec(rule)",
        model_dir="Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
    ),
    RunSpec(
        "Instruments",
        "GenRec (ranking)",
        "GenRec(ranking)",
        model_dir="Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495",
    ),
    RunSpec(
        "Instruments",
        "GenRec (fixed)",
        "GenRec(fixed)",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
    ),
    RunSpec(
        "Instruments",
        "GenRec (fixed + ce0.005)",
        "GenRec(fixed + ce0.005)",
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
    ),
    RunSpec("Games", "Caser", "Caser", model_dir="Games-caser-pytorch/checkpoint-best"),
    RunSpec("Games", "GRU4Rec", "GRU4Rec", model_dir="Games-gru4rec-pytorch-official/checkpoint-best"),
    RunSpec(
        "Games", "BERT4Rec (RecSys23)", "BERT4Rec (RecSys23)", model_dir="Games-bert4rec-recsys23/checkpoint-best"
    ),
    RunSpec("Games", "BERT4Rec", "BERT4Rec", model_dir="Games-bert4rec-vanilla/checkpoint-best"),
    RunSpec("Games", "SASRec", "SASRec", model_dir="Games-sasrec-recsys23/checkpoint-best"),
    RunSpec("Games", "TIGER", "TIGER", model_dir="Games-tiger-grec-h50-reverse/checkpoint-best"),
    RunSpec("Games", "LC-Rec", "LC-Rec", pending=True, note="pending"),
    RunSpec("Games", "GenRec (SFT)", "GenRec(sft)", model_dir="Games-grec-sft-qwen4B-4-256-dsz0"),
    RunSpec(
        "Games",
        "GenRec (rule)",
        "GenRec(rule)",
        model_dir="Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896",
    ),
    RunSpec("Games", "GenRec (ranking)", "GenRec(ranking)", pending=True, note="not measured"),
    RunSpec(
        "Games", "GenRec (fixed)", "GenRec(fixed)", model_dir="Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896"
    ),
    RunSpec("Games", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", pending=True, note="not measured"),
    RunSpec("Arts", "Caser", "Caser", model_dir="Arts-caser-pytorch/checkpoint-best"),
    RunSpec("Arts", "GRU4Rec", "GRU4Rec", model_dir="Arts-gru4rec-pytorch-official/checkpoint-best"),
    RunSpec("Arts", "BERT4Rec (RecSys23)", "BERT4Rec (RecSys23)", model_dir="Arts-bert4rec-recsys23/checkpoint-best"),
    RunSpec("Arts", "BERT4Rec", "BERT4Rec", model_dir="Arts-bert4rec-vanilla/checkpoint-best"),
    RunSpec("Arts", "SASRec", "SASRec", model_dir="Arts-sasrec-recsys23/checkpoint-best"),
    RunSpec("Arts", "TIGER", "TIGER", model_dir="Arts-tiger-grec-h50-reverse/checkpoint-best"),
    RunSpec("Arts", "LC-Rec", "LC-Rec", pending=True, note="pending"),
    RunSpec("Arts", "GenRec (SFT)", "GenRec(sft)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (rule)", "GenRec(rule)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (ranking)", "GenRec(ranking)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (fixed)", "GenRec(fixed)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", pending=True, note="not measured"),
]

LC_REC_RUN_SPECS: list[RunSpec] = [
    RunSpec("Instruments", "Caser", "Caser", model_dir="Instruments-caser-pytorch/checkpoint-best"),
    RunSpec("Instruments", "GRU4Rec", "GRU4Rec", model_dir="Instruments-gru4rec-pytorch-official/checkpoint-best"),
    RunSpec(
        "Instruments",
        "BERT4Rec (RecSys23)",
        "BERT4Rec (RecSys23)",
        model_dir="Instruments-bert4rec-recsys23/checkpoint-best",
    ),
    RunSpec("Instruments", "BERT4Rec", "BERT4Rec", model_dir="Instruments-bert4rec-vanilla/checkpoint-best"),
    RunSpec("Instruments", "SASRec", "SASRec", model_dir="Instruments-sasrec-recsys23/checkpoint-best"),
    RunSpec("Instruments", "TIGER", "TIGER", model_dir="Instruments-tiger-grec-h50-reverse/checkpoint-best"),
    RunSpec("Instruments", "LC-Rec", "LC-Rec", model_dir="Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu"),
    RunSpec(
        "Instruments",
        "GenRec (SFT)",
        "GenRec(sft)",
        model_dir="Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu",
    ),
    RunSpec("Instruments", "GenRec (rule)", "GenRec(rule)", model_dir="Instruments-grec-genrec-aligned-rule"),
    RunSpec("Instruments", "GenRec (ranking)", "GenRec(ranking)", pending=True, note="not measured"),
    RunSpec("Instruments", "GenRec (fixed)", "GenRec(fixed)", pending=True, note="not measured"),
    RunSpec("Instruments", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", pending=True, note="not measured"),
    RunSpec("Games", "Caser", "Caser", model_dir="Games-caser-pytorch/checkpoint-best"),
    RunSpec("Games", "GRU4Rec", "GRU4Rec", model_dir="Games-gru4rec-pytorch-official/checkpoint-best"),
    RunSpec(
        "Games", "BERT4Rec (RecSys23)", "BERT4Rec (RecSys23)", model_dir="Games-bert4rec-recsys23/checkpoint-best"
    ),
    RunSpec("Games", "BERT4Rec", "BERT4Rec", model_dir="Games-bert4rec-vanilla/checkpoint-best"),
    RunSpec("Games", "SASRec", "SASRec", model_dir="Games-sasrec-recsys23/checkpoint-best"),
    RunSpec("Games", "TIGER", "TIGER", model_dir="Games-tiger-grec-h50-reverse/checkpoint-best"),
    RunSpec("Games", "LC-Rec", "LC-Rec", pending=True, note="pending"),
    RunSpec("Games", "GenRec (SFT)", "GenRec(sft)", pending=True, note="not measured"),
    RunSpec("Games", "GenRec (rule)", "GenRec(rule)", pending=True, note="not measured"),
    RunSpec("Games", "GenRec (ranking)", "GenRec(ranking)", pending=True, note="not measured"),
    RunSpec("Games", "GenRec (fixed)", "GenRec(fixed)", pending=True, note="not measured"),
    RunSpec("Games", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", pending=True, note="not measured"),
    RunSpec("Arts", "Caser", "Caser", model_dir="Arts-caser-pytorch/checkpoint-best"),
    RunSpec("Arts", "GRU4Rec", "GRU4Rec", model_dir="Arts-gru4rec-pytorch-official/checkpoint-best"),
    RunSpec("Arts", "BERT4Rec (RecSys23)", "BERT4Rec (RecSys23)", model_dir="Arts-bert4rec-recsys23/checkpoint-best"),
    RunSpec("Arts", "BERT4Rec", "BERT4Rec", model_dir="Arts-bert4rec-vanilla/checkpoint-best"),
    RunSpec("Arts", "SASRec", "SASRec", model_dir="Arts-sasrec-recsys23/checkpoint-best"),
    RunSpec("Arts", "TIGER", "TIGER", model_dir="Arts-tiger-grec-h50-reverse/checkpoint-best"),
    RunSpec("Arts", "LC-Rec", "LC-Rec", pending=True, note="pending"),
    RunSpec("Arts", "GenRec (SFT)", "GenRec(sft)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (rule)", "GenRec(rule)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (ranking)", "GenRec(ranking)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (fixed)", "GenRec(fixed)", pending=True, note="not measured"),
    RunSpec("Arts", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", pending=True, note="not measured"),
]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def checkpoint_step(name: str) -> int:
    match = re.search(r"checkpoint-(\d+)", name)
    if not match:
        return -1
    return int(match.group(1))


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def fmt_texttt(value: str) -> str:
    return r"\texttt{" + value.replace("_", r"\_") + "}"


def read_metrics_json(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    return {k: float(v) for k, v in data.items()}


def collect_metrics_dir(model_dir: str) -> list[tuple[str, dict[str, float]]]:
    root = RESULTS_DIR / model_dir
    if not root.exists():
        return []
    direct_metrics_path = root / "metrics.json"
    if direct_metrics_path.exists():
        return [(root.name, read_metrics_json(direct_metrics_path))]
    entries: list[tuple[str, dict[str, float]]] = []
    for ckpt_dir in sorted(root.glob("checkpoint-*"), key=lambda p: checkpoint_step(p.name)):
        metrics_path = ckpt_dir / "metrics.json"
        if metrics_path.exists():
            entries.append((ckpt_dir.name, read_metrics_json(metrics_path)))
    return entries


def checkpoint_epoch(checkpoint_name: str, max_step: int, num_train_epochs: float) -> float:
    step = checkpoint_step(checkpoint_name)
    if step <= 0 or max_step <= 0:
        return float("nan")
    return step / max_step * num_train_epochs


def select_best(entries: list[tuple[str, dict[str, float]]]) -> tuple[str, dict[str, float]] | None:
    if not entries:
        return None
    return max(entries, key=lambda item: (item[1].get("NDCG@10", float("-inf")), checkpoint_step(item[0])))


def build_arts_sft_best() -> tuple[str, dict[str, float]]:
    rows = load_csv_rows(ARTS_CKPT_CSV)
    best = max(rows, key=lambda r: float(r["NDCG@10"]))
    metrics = {
        "HR@1": float(best["HR@1"]),
        "HR@5": float(best["HR@5"]),
        "HR@10": float(best["HR@10"]),
        "HR@20": float("nan"),
        "HR@50": float(best["HR@50"]),
        "NDCG@5": float(best["NDCG@5"]),
        "NDCG@10": float(best["NDCG@10"]),
        "NDCG@20": float("nan"),
        "NDCG@50": float(best["NDCG@50"]),
    }
    return f"checkpoint-{best['step']}", metrics


def lookup_best_from_csv(model_dir: str, rows: list[dict[str, str]]) -> tuple[str, dict[str, float]] | None:
    for row in rows:
        if row.get("model_dir") == model_dir:
            metrics = {
                "HR@5": float(row["HR@5"]),
                "HR@10": float(row["HR@10"]),
                "HR@50": float(row["HR@50"]),
                "NDCG@5": float(row["NDCG@5"]),
                "NDCG@10": float(row["NDCG@10"]),
                "NDCG@50": float(row["NDCG@50"]),
            }
            return row["checkpoint"], metrics
    return None


def maybe_value(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or math.isnan(value):
        return None
    return value


def build_aligned_rule_assets() -> list[dict[str, float | str]]:
    rule_entries = collect_metrics_dir(ALIGNED_RULE_MODEL_DIR)
    sft_best = select_best(collect_metrics_dir(ALIGNED_SFT_MODEL_DIR))
    if not rule_entries or sft_best is None:
        return []

    sft_checkpoint, sft_metrics = sft_best

    rows: list[dict[str, float | str]] = []
    for checkpoint_name, metrics in rule_entries:
        row: dict[str, float | str] = {
            "checkpoint": checkpoint_name,
            "step": checkpoint_step(checkpoint_name),
            "epoch": checkpoint_epoch(
                checkpoint_name,
                ALIGNED_RULE_EPOCH_NORMALIZER_STEP,
                ALIGNED_RULE_NUM_EPOCHS,
            ),
        }
        for metric in METRICS:
            row[metric] = float(metrics[metric])
        rows.append(row)

    INSTRUMENTS_REPORT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with ALIGNED_RULE_METRICS_CSV.open("w", newline="") as f:
        fieldnames = ["checkpoint", "step", "epoch", *METRICS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(11.6, 14.0), sharex=True)
    axes_flat = list(axes.flat)
    epochs = [float(row["epoch"]) for row in rows]

    for ax, metric in zip(axes_flat, METRICS):
        values = [float(row[metric]) for row in rows]
        ax.plot(
            epochs,
            values,
            color="#9C755F",
            marker="s",
            linewidth=2.2,
            markersize=5,
            label="GenRec(rule)",
        )
        ax.axhline(
            float(sft_metrics[metric]),
            linestyle="--",
            linewidth=1.4,
            color="#6B7280",
            label=f"GenRec(sft) best ({sft_checkpoint})",
        )
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_xlim(0.0, ALIGNED_RULE_NUM_EPOCHS)
        ax.grid(alpha=0.22)

    for ax in axes_flat[len(METRICS):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=2, frameon=False)
    fig.suptitle("Instruments GenRec aligned rule: all main-table metrics vs SFT best", y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ALIGNED_RULE_CURVES_PNG, dpi=180)
    plt.close(fig)
    return rows


def resolve_run(
    spec: RunSpec,
    instruments_best: list[dict[str, str]],
    games_sft_best: list[dict[str, str]],
    games_rl_best: list[dict[str, str]],
) -> dict[str, object]:
    if spec.pending:
        return {"spec": spec, "status": "pending", "best_checkpoint": None, "metrics": {}}

    if spec.dataset == "Instruments":
        if spec.model_dir is not None:
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        else:
            best = None
    elif spec.dataset == "Games":
        if spec.model_dir == "Games-tiger-grec-h50-reverse/checkpoint-best":
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        elif spec.model_dir == "Games-grec-sft-qwen4B-4-256-dsz0":
            found = lookup_best_from_csv(spec.model_dir, games_sft_best)
            if found is None:
                entries = collect_metrics_dir(spec.model_dir)
                best = select_best(entries)
            else:
                ckpt, metrics = found
                best = (ckpt, {**metrics, **read_metrics_json(RESULTS_DIR / spec.model_dir / ckpt / "metrics.json")})
        elif spec.model_dir is not None:
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        else:
            found = lookup_best_from_csv(spec.model_dir or "", games_rl_best)
            if found is None:
                entries = collect_metrics_dir(spec.model_dir or "")
                best = select_best(entries)
            else:
                ckpt, metrics = found
                best = (
                    ckpt,
                    {**metrics, **read_metrics_json(RESULTS_DIR / (spec.model_dir or "") / ckpt / "metrics.json")},
                )
    elif spec.dataset == "Arts":
        if spec.model_dir == "Arts-tiger-grec-h50-reverse/checkpoint-best":
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        elif spec.model_dir is None and spec.column_name == "GenRec(sft)":
            best = build_arts_sft_best()
        elif spec.model_dir is not None:
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        else:
            best = None
    else:
        best = None

    if best is None:
        return {"spec": spec, "status": "pending", "best_checkpoint": None, "metrics": {}}
    ckpt, metrics = best
    return {"spec": spec, "status": "ok", "best_checkpoint": ckpt, "metrics": metrics}


def best_highlights(dataset_rows: list[dict[str, object]], metric: str) -> tuple[set[str], set[str]]:
    measured = []
    for row in dataset_rows:
        value = maybe_value(row["metrics"], metric)  # type: ignore[arg-type]
        if value is not None:
            measured.append((row["spec"].column_name, value))  # type: ignore[attr-defined]
    if not measured:
        return set(), set()
    best_value = max(v for _, v in measured)
    best_cols = {c for c, v in measured if abs(v - best_value) < 1e-12}
    baseline_candidates = [
        item
        for item in measured
        if item[0]
        not in {
            "GenRec(sft)",
            "GenRec(rule)",
            "GenRec(ranking)",
            "GenRec(fixed)",
            "GenRec(fixed + ce0.005)",
            "BERT4Rec (RecSys23)",
        }
    ]
    if not baseline_candidates:
        return best_cols, set()
    baseline_best_value = max(v for _, v in baseline_candidates)
    baseline_best_cols = {c for c, v in baseline_candidates if abs(v - baseline_best_value) < 1e-12}
    return best_cols, baseline_best_cols


def render_cell(col_name: str, metric: str, row: dict[str, object], best_cols: set[str]) -> str:
    metrics = row["metrics"]  # type: ignore[assignment]
    value = maybe_value(metrics, metric)
    if value is None:
        return r"\textemdash"
    text = fmt_metric(value)
    if col_name in best_cols:
        return r"\textbf{" + text + "}"
    return text


def dataset_has_measured_requested_method(dataset_rows: list[dict[str, object]]) -> bool:
    for row in dataset_rows:
        if row["status"] == "ok":
            return True
    return False


def visible_datasets_for(resolved: list[dict[str, object]], columns: list[str]) -> list[str]:
    visible_datasets = []
    for dataset in ["Instruments", "Arts", "Games"]:
        dataset_rows = [row for row in resolved if row["spec"].dataset == dataset]  # type: ignore[attr-defined]
        scoped_rows = [row for row in dataset_rows if row["spec"].column_name in columns]  # type: ignore[attr-defined]
        if dataset_has_measured_requested_method(scoped_rows):
            visible_datasets.append(dataset)
    return visible_datasets


def render_result_table(
    resolved: list[dict[str, object]],
    columns: list[str],
    caption: str,
    label: str,
) -> list[str]:
    visible_datasets = visible_datasets_for(resolved, columns)
    parts: list[str] = []
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{3pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.08}")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{ll" + "c" * len(columns) + r"}")
    parts.append(r"\toprule")
    parts.append("Dataset & Metric & " + " & ".join(columns) + r" \\")
    parts.append(r"\midrule")

    for dataset_idx, dataset in enumerate(visible_datasets):
        dataset_rows = [row for row in resolved if row["spec"].dataset == dataset]  # type: ignore[attr-defined]
        visible_dataset_rows = [row for row in dataset_rows if row["spec"].column_name in columns]  # type: ignore[attr-defined]
        for idx, metric in enumerate(METRICS):
            best_cols, _ = best_highlights(visible_dataset_rows, metric)
            cells = []
            for col in columns:
                row = next(row for row in dataset_rows if row["spec"].column_name == col)  # type: ignore[attr-defined]
                cells.append(render_cell(col, metric, row, best_cols))
            prefix = dataset if idx == 0 else ""
            parts.append(" & ".join([prefix, metric, *cells]) + r" \\")
        if dataset_idx != len(visible_datasets) - 1:
            parts.append(r"\midrule")

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    parts.append(rf"\caption{{{caption}}}")
    parts.append(rf"\label{{{label}}}")
    parts.append(r"\end{table}")
    parts.append("")
    return parts


def build_main_table(resolved: list[dict[str, object]]) -> str:
    aligned_rule_row = next(
        row
        for row in resolved
        if row["spec"].dataset == "Instruments" and row["spec"].column_name == "GenRec(rule)" and row["status"] == "ok"
    )
    aligned_rule_ckpt = str(aligned_rule_row["best_checkpoint"])
    aligned_rule_metrics: dict[str, float] = aligned_rule_row["metrics"]  # type: ignore[assignment]
    parts: list[str] = []
    parts.append(r"\section{GenRec Index Baseline Tables}")
    parts.append("")
    parts.append(r"\subsection{Main Table}")
    parts.append("")
    parts.extend(
        render_result_table(
            resolved,
            MAIN_TABLE_COLUMNS,
            r"GenRec index 口径下当前已测结果的主表。当前第一页仅纳入已经按该口径完成实测的结果：Instruments 上 \texttt{LC-Rec} 与 \texttt{GenRec(sft)} 共享同一条 \texttt{genrec-aligned-sft} 结果，\texttt{GenRec(rule)} 使用 \texttt{Instruments-grec-genrec-aligned-rule}；其余尚未跑出的列暂保留为空。原始旧版 GenRec(sft) 基座总表移到附录，RecSys23 版 BERT4Rec 只保留在附录中作参考。所有 GenRec 变体若存在多个 checkpoint，统一按 NDCG@10 选 best。",
            "tab:genrec-index-main",
        )
    )

    parts.append(r"\subsection{Best Checkpoint Summary}")
    parts.append("")
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{4pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.06}")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{llc" + "c" * len(METRICS) + r"}")
    parts.append(r"\toprule")
    parts.append("Dataset & Variant & Best ckpt & " + " & ".join(METRICS) + r" \\")
    parts.append(r"\midrule")
    visible_datasets = visible_datasets_for(resolved, MAIN_TABLE_COLUMNS)
    for dataset_idx, dataset in enumerate(visible_datasets):
        dataset_rows = [
            row
            for row in resolved
            if row["spec"].dataset == dataset and row["status"] == "ok" and row["spec"].column_name != "BERT4Rec (RecSys23)"
        ]  # type: ignore[attr-defined]
        for idx, row in enumerate(dataset_rows):
            spec: RunSpec = row["spec"]  # type: ignore[assignment]
            metrics: dict[str, float] = row["metrics"]  # type: ignore[assignment]
            cells = [fmt_metric(maybe_value(metrics, metric)) for metric in METRICS]
            parts.append(
                " & ".join(
                    [
                        dataset if idx == 0 else "",
                        spec.label.replace("GenRec ", "GenRec "),
                        fmt_texttt(row["best_checkpoint"]),  # type: ignore[arg-type]
                        *cells,
                    ]
                )
                + r" \\"
            )
        if dataset_idx != len(visible_datasets) - 1:
            parts.append(r"\midrule")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    parts.append(r"\caption{当前主表中所有已测方法的 best checkpoint 汇总。}")
    parts.append(r"\label{tab:genrec-index-best-summary}")
    parts.append(r"\end{table}")
    parts.append("")

    parts.append(
        r"\paragraph{Readout.} 当前首页主表只保留已经完成的 GenRec index 口径实测。"
    )
    parts.append(
        rf"Instruments 上，\texttt{{LC-Rec}} 与 \texttt{{GenRec(sft)}} 目前都对齐到 \texttt{{Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu}}，最佳 checkpoint 都是 \texttt{{checkpoint-2751 / NDCG@10=0.1033 / HR@50=0.2195}}；当前唯一补进来的 RL 结果是 \texttt{{GenRec(rule)}}，best checkpoint 为 \texttt{{{aligned_rule_ckpt} / NDCG@10={fmt_metric(maybe_value(aligned_rule_metrics, 'NDCG@10'))} / HR@50={fmt_metric(maybe_value(aligned_rule_metrics, 'HR@50'))}}}。"
    )
    parts.append(
        r"Arts 与 Games 在这张表里暂时只保留 baseline，因为当前还没有按同一口径补齐新的 \texttt{GenRec(sft) / rule / fixed} 结果。"
    )
    parts.append(
        r"旧版基座下的完整对照仍保留在附录，方便继续和历史结果比对。"
    )
    parts.append("")
    parts.append(build_aligned_rule_section())
    return "\n".join(parts) + "\n"


def build_aligned_rule_section() -> str:
    rule_entries = collect_metrics_dir(ALIGNED_RULE_MODEL_DIR)
    sft_best = select_best(collect_metrics_dir(ALIGNED_SFT_MODEL_DIR))
    if not rule_entries or sft_best is None:
        return ""

    last_synced_step = max(checkpoint_step(ckpt) for ckpt, _ in rule_entries)
    sft_checkpoint, sft_metrics = sft_best
    best_rule = select_best(rule_entries)
    assert best_rule is not None
    best_rule_checkpoint, best_rule_metrics = best_rule

    parts: list[str] = []
    parts.append(r"\subsection{Instruments: GenRec(rule) Checkpoint Curves}")
    parts.append("")
    parts.append(r"\begin{figure}[H]")
    parts.append(r"\centering")
    parts.append(r"\includegraphics[width=\textwidth]{../research/assets/instruments-report/genrec_aligned_rule_epoch_curves.png}")
    parts.append(
        rf"\caption{{Instruments 上 \texttt{{GenRec(rule)}} 的完整 checkpoint 曲线。图中统一展示主表里的九个指标：\texttt{{HR@1/5/10/20/50}} 与 \texttt{{NDCG@5/10/20/50}}；横轴统一按 Instruments RL 主线的 \texttt{{3326 step = 2 epoch}} 归一化，因此当前最后一个已同步点 \texttt{{checkpoint-{last_synced_step}}} 对应约 \texttt{{{checkpoint_epoch(f'checkpoint-{last_synced_step}', ALIGNED_RULE_EPOCH_NORMALIZER_STEP, ALIGNED_RULE_NUM_EPOCHS):.3f} epoch}}，虚线表示 \texttt{{GenRec(sft)}}/\texttt{{LC-Rec}} 共享的最佳 SFT 基线 \texttt{{{sft_checkpoint}}}。}}"
    )
    parts.append(r"\label{fig:genrec-index-aligned-rule-curves}")
    parts.append(r"\end{figure}")
    parts.append("")
    parts.append(
        rf"\paragraph{{Readout.}} \texttt{{GenRec(rule)}} 最优点出现在 \texttt{{{best_rule_checkpoint}}}，对应 \texttt{{NDCG@10={fmt_metric(maybe_value(best_rule_metrics, 'NDCG@10'))}}}。从九个指标整体看，这条 rule 线在前排 hit/NDCG 上超过了当前 best SFT，但 \texttt{{HR@50}} 仍始终低于 \texttt{{GenRec(sft)}} 的 \texttt{{{fmt_metric(maybe_value(sft_metrics, 'HR@50'))}}}。"
    )
    parts.append("")
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{3.5pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.05}")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{c c " + " c" * len(METRICS) + r"}")
    parts.append(r"\toprule")
    parts.append("Epoch & Step & " + " & ".join(METRICS) + r" \\")
    parts.append(r"\midrule")
    for checkpoint_name, metrics in rule_entries:
        step = checkpoint_step(checkpoint_name)
        epoch = checkpoint_epoch(checkpoint_name, ALIGNED_RULE_EPOCH_NORMALIZER_STEP, ALIGNED_RULE_NUM_EPOCHS)
        cells = [fmt_metric(maybe_value(metrics, metric)) for metric in METRICS]
        parts.append(" & ".join([f"{epoch:.3f}", str(step), *cells]) + r" \\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    parts.append(
        rf"\caption{{\texttt{{{ALIGNED_RULE_MODEL_DIR.replace('_', r'\_')}}} 的逐 checkpoint 指标明细。epoch 统一按 \texttt{{step / {ALIGNED_RULE_EPOCH_NORMALIZER_STEP} * 2}} 计算，对齐 Instruments RL 主线的 \texttt{{3326 step = 2 epoch}} 口径；因此当前最后一个已同步点 \texttt{{checkpoint-{last_synced_step}}} 对应约 \texttt{{{checkpoint_epoch(f'checkpoint-{last_synced_step}', ALIGNED_RULE_EPOCH_NORMALIZER_STEP, ALIGNED_RULE_NUM_EPOCHS):.3f} epoch}}。}}"
    )
    parts.append(r"\label{tab:genrec-index-aligned-rule-all-points}")
    parts.append(r"\end{table}")
    parts.append("")
    return "\n".join(parts)


def appendix_run_specs() -> list[RunSpec]:
    return [
        RunSpec("Instruments", "Caser", "Caser", model_dir="Instruments-caser-pytorch/checkpoint-best"),
        RunSpec("Instruments", "GRU4Rec", "GRU4Rec", model_dir="Instruments-gru4rec-pytorch-official/checkpoint-best"),
        RunSpec(
            "Instruments",
            "BERT4Rec (RecSys23)",
            "BERT4Rec (RecSys23)",
            model_dir="Instruments-bert4rec-recsys23/checkpoint-best",
        ),
        RunSpec(
            "Instruments",
            "BERT4Rec",
            "BERT4Rec",
            model_dir="Instruments-bert4rec-vanilla/checkpoint-best",
        ),
        RunSpec("Instruments", "SASRec", "SASRec", model_dir="Instruments-sasrec-recsys23/checkpoint-best"),
        RunSpec("Instruments", "TIGER", "TIGER", model_dir="Instruments-tiger-grec-h50-reverse/checkpoint-best"),
        RunSpec(
            "Instruments", "LC-Rec", "LC-Rec", model_dir="Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu"
        ),
        RunSpec(
            "Instruments",
            "GenRec aligned rule",
            "GenRec(rule)",
            model_dir="Instruments-grec-genrec-aligned-rule",
        ),
        RunSpec("Instruments", "GenRec (SFT)", "GenRec(sft)", model_dir="Instruments-grec-sft-qwen4B-4-256-dsz0"),
        RunSpec(
            "Instruments",
            "GenRec (rule)",
            "GenRec(rule)",
            model_dir="Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
        ),
        RunSpec(
            "Instruments",
            "GenRec (ranking)",
            "GenRec(ranking)",
            model_dir="Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495",
        ),
        RunSpec(
            "Instruments",
            "GenRec (fixed)",
            "GenRec(fixed)",
            model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        ),
        RunSpec(
            "Instruments",
            "GenRec (fixed + ce0.005)",
            "GenRec(fixed + ce0.005)",
            model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
        ),
        RunSpec("Games", "Caser", "Caser", model_dir="Games-caser-pytorch/checkpoint-best"),
        RunSpec("Games", "GRU4Rec", "GRU4Rec", model_dir="Games-gru4rec-pytorch-official/checkpoint-best"),
        RunSpec(
            "Games", "BERT4Rec (RecSys23)", "BERT4Rec (RecSys23)", model_dir="Games-bert4rec-recsys23/checkpoint-best"
        ),
        RunSpec("Games", "BERT4Rec", "BERT4Rec", model_dir="Games-bert4rec-vanilla/checkpoint-best"),
        RunSpec("Games", "SASRec", "SASRec", model_dir="Games-sasrec-recsys23/checkpoint-best"),
        RunSpec("Games", "TIGER", "TIGER", model_dir="Games-tiger-grec-h50-reverse/checkpoint-best"),
        RunSpec("Games", "GenRec (SFT)", "GenRec(sft)", model_dir="Games-grec-sft-qwen4B-4-256-dsz0"),
        RunSpec(
            "Games",
            "GenRec (rule)",
            "GenRec(rule)",
            model_dir="Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896",
        ),
        RunSpec(
            "Games",
            "GenRec (fixed)",
            "GenRec(fixed)",
            model_dir="Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896",
        ),
        RunSpec("Arts", "Caser", "Caser", model_dir="Arts-caser-pytorch/checkpoint-best"),
        RunSpec("Arts", "GRU4Rec", "GRU4Rec", model_dir="Arts-gru4rec-pytorch-official/checkpoint-best"),
        RunSpec(
            "Arts", "BERT4Rec (RecSys23)", "BERT4Rec (RecSys23)", model_dir="Arts-bert4rec-recsys23/checkpoint-best"
        ),
        RunSpec("Arts", "BERT4Rec", "BERT4Rec", model_dir="Arts-bert4rec-vanilla/checkpoint-best"),
        RunSpec("Arts", "SASRec", "SASRec", model_dir="Arts-sasrec-recsys23/checkpoint-best"),
        RunSpec("Arts", "TIGER", "TIGER", model_dir="Arts-tiger-grec-h50-reverse/checkpoint-best"),
    ]


def render_appendix_table(spec: RunSpec) -> str:
    entries = collect_metrics_dir(spec.model_dir or "")
    if not entries:
        return ""

    parts: list[str] = []
    safe_label = re.sub(r"[^a-z0-9]+", "-", f"{spec.dataset}-{spec.label}".lower()).strip("-")
    parts.append(rf"\subsection{{{spec.dataset}: {spec.label}}}")
    parts.append("")
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{3.5pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.05}")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{c" + "c" * len(METRICS) + r"}")
    parts.append(r"\toprule")
    parts.append("Checkpoint & " + " & ".join(METRICS) + r" \\")
    parts.append(r"\midrule")
    for ckpt, metrics in entries:
        cells = [fmt_metric(maybe_value(metrics, metric)) for metric in METRICS]
        parts.append(" & ".join([fmt_texttt(ckpt), *cells]) + r" \\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    parts.append(
        rf"\caption{{{spec.dataset} 上 \texttt{{{(spec.model_dir or '').replace('_', r'\_')}}} 的完整 checkpoint 指标。}}"
    )
    parts.append(rf"\label{{tab:appendix-{safe_label}}}")
    parts.append(r"\end{table}")
    parts.append("")
    return "\n".join(parts)


def format_reference_value(value: float, is_best: bool, is_second: bool) -> str:
    text = fmt_metric(value)
    if is_best:
        return r"\textbf{" + text + "}"
    if is_second:
        return r"\underline{" + text + "}"
    return text


def build_paper_reference_table() -> str:
    parts: list[str] = []
    parts.append(r"\subsection{Paper Reference Table}")
    parts.append("")
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{3pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.06}")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{ll" + "c" * len(PAPER_REFERENCE_METHODS) + r"c}")
    parts.append(r"\toprule")
    parts.append("Dataset & Metrics & " + " & ".join(PAPER_REFERENCE_METHODS) + r" & Improv. \\")
    parts.append(r"\midrule")

    for dataset_idx, (dataset, rows) in enumerate(PAPER_REFERENCE_ROWS):
        for row_idx, (metric, values) in enumerate(rows):
            metric_values = {method: float(values[method]) for method in PAPER_REFERENCE_METHODS}
            sorted_unique = sorted(set(metric_values.values()), reverse=True)
            best_value = sorted_unique[0]
            second_value = sorted_unique[1] if len(sorted_unique) > 1 else None
            rendered = []
            for method in PAPER_REFERENCE_METHODS:
                value = metric_values[method]
                rendered.append(
                    format_reference_value(
                        value,
                        abs(value - best_value) < 1e-12,
                        second_value is not None and abs(value - second_value) < 1e-12,
                    )
                )
            prefix = dataset if row_idx == 0 else ""
            parts.append(" & ".join([prefix, metric, *rendered, str(values["Improv."])]) + r" \\")
        if dataset_idx != len(PAPER_REFERENCE_ROWS) - 1:
            parts.append(r"\midrule")

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    parts.append(
        r"\caption{Performance comparison of different methods on the three datasets. The best and second-best performances are indicated in bold and underlined font, respectively.}"
    )
    parts.append(r"\label{tab:appendix-paper-reference-mql4grec}")
    parts.append(r"\end{table}")
    parts.append("")
    return "\n".join(parts)


def build_appendix() -> str:
    parts: list[str] = []
    parts.append(r"\appendix")
    parts.append(r"\section{Supplementary Tables}")
    parts.append("")
    parts.append(
        r"本附录先放一张论文参考主表，随后按结果目录逐条展开当前已经同步回仓库的完整 checkpoint 指标，方便后续继续补充主表、做 early-stop 对照，或替换 best 选点。"
    )
    parts.append("")
    parts.append(build_paper_reference_table())
    parts.append(r"\subsection{Original GenRec(sft)-Based Table}")
    parts.append("")
    parts.extend(
        render_result_table(
            [resolve_run(spec, [], [], []) for spec in RUN_SPECS],
            MAIN_TABLE_COLUMNS,
            r"原始 GenRec(sft) 基座下的结果总表。该表从正文移至附录，用于和第一页展示的 LC-Rec 对齐 RL 结果区分。",
            "tab:genrec-index-original-appendix",
        )
    )
    parts.append(r"\subsection{Full Checkpoint Tables}")
    parts.append("")
    for spec in appendix_run_specs():
        table = render_appendix_table(spec)
        if table:
            parts.append(table)
    return "\n".join(parts) + "\n"


def build_master() -> str:
    return r"""\documentclass[11pt,a4paper]{ctexart}

\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{float}
\usepackage{graphicx}
\usepackage{textcomp}
\graphicspath{{../research/assets/instruments-report/}}

\begin{document}

\input{genrec-baseline-main}
\input{genrec-baseline-appendix}

\end{document}
"""


def main() -> None:
    resolved = [resolve_run(spec, [], [], []) for spec in LC_REC_RUN_SPECS]
    build_aligned_rule_assets()

    MAIN_TEX.write_text(build_main_table(resolved))
    APPENDIX_TEX.write_text(build_appendix())
    MASTER_TEX.write_text(build_master())


if __name__ == "__main__":
    main()
