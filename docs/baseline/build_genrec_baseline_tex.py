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
INSTRUMENTS_BEST_CSV = ROOT / "docs" / "research" / "assets" / "instruments-report" / "all_variant_best_summary.csv"
GAMES_SFT_BEST_CSV = ROOT / "docs" / "research" / "assets" / "games-report" / "games_sft_best_summary.csv"
GAMES_RL_BEST_CSV = ROOT / "docs" / "research" / "assets" / "games-report" / "games_rl_best_summary.csv"
ARTS_CKPT_CSV = ROOT / "docs" / "research" / "assets" / "arts-report" / "arts_checkpoint_metrics.csv"

MAIN_TEX = BASELINE_DIR / "genrec-baseline-main.tex"
APPENDIX_TEX = BASELINE_DIR / "genrec-baseline-appendix.tex"
MASTER_TEX = BASELINE_DIR / "genrec-baseline.tex"


METRICS = ["HR@1", "HR@5", "HR@10", "HR@20", "HR@50", "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50"]


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
    RunSpec("Instruments", "Caser", "Caser", pending=True, note="pending"),
    RunSpec("Instruments", "GRU4Rec", "GRU4Rec", pending=True, note="pending"),
    RunSpec("Instruments", "BERT4Rec", "BERT4Rec", pending=True, note="pending"),
    RunSpec("Instruments", "SASRec", "SASRec", pending=True, note="pending"),
    RunSpec("Instruments", "TIGER", "TIGER", pending=True, note="pending"),
    RunSpec("Instruments", "GenRec (SFT)", "GenRec(sft)", model_dir="Instruments-grec-sft-qwen4B-4-256-dsz0"),
    RunSpec("Instruments", "GenRec (rule)", "GenRec(rule)", model_dir="Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495"),
    RunSpec("Instruments", "GenRec (ranking)", "GenRec(ranking)", model_dir="Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495"),
    RunSpec("Instruments", "GenRec (fixed)", "GenRec(fixed)", model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495"),
    RunSpec("Instruments", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495"),
    RunSpec("Games", "Caser", "Caser", pending=True, note="pending"),
    RunSpec("Games", "GRU4Rec", "GRU4Rec", pending=True, note="pending"),
    RunSpec("Games", "BERT4Rec", "BERT4Rec", pending=True, note="pending"),
    RunSpec("Games", "SASRec", "SASRec", pending=True, note="pending"),
    RunSpec("Games", "TIGER", "TIGER", pending=True, note="pending"),
    RunSpec("Games", "GenRec (SFT)", "GenRec(sft)", model_dir="Games-grec-sft-qwen4B-4-256-dsz0"),
    RunSpec("Games", "GenRec (rule)", "GenRec(rule)", model_dir="Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896"),
    RunSpec("Games", "GenRec (ranking)", "GenRec(ranking)", pending=True, note="not measured"),
    RunSpec("Games", "GenRec (fixed)", "GenRec(fixed)", model_dir="Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896"),
    RunSpec("Games", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", pending=True, note="not measured"),
    RunSpec("Arts", "Caser", "Caser", pending=True, note="pending"),
    RunSpec("Arts", "GRU4Rec", "GRU4Rec", pending=True, note="pending"),
    RunSpec("Arts", "BERT4Rec", "BERT4Rec", pending=True, note="pending"),
    RunSpec("Arts", "SASRec", "SASRec", pending=True, note="pending"),
    RunSpec("Arts", "TIGER", "TIGER", pending=True, note="pending"),
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
        raise ValueError(f"Bad checkpoint name: {name}")
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
    entries: list[tuple[str, dict[str, float]]] = []
    for ckpt_dir in sorted(root.glob("checkpoint-*"), key=lambda p: checkpoint_step(p.name)):
        metrics_path = ckpt_dir / "metrics.json"
        if metrics_path.exists():
            entries.append((ckpt_dir.name, read_metrics_json(metrics_path)))
    return entries


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


def resolve_run(spec: RunSpec, instruments_best: list[dict[str, str]], games_sft_best: list[dict[str, str]], games_rl_best: list[dict[str, str]]) -> dict[str, object]:
    if spec.pending:
        return {"spec": spec, "status": "pending", "best_checkpoint": None, "metrics": {}}

    if spec.dataset == "Instruments":
        if spec.model_dir == "Instruments-grec-sft-qwen4B-4-256-dsz0":
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        elif spec.model_dir in {
            "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495",
            "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
            "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495",
        }:
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        elif spec.model_dir == "Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495":
            entries = collect_metrics_dir(spec.model_dir)
            best = select_best(entries)
        else:
            entries = []
            best = None
    elif spec.dataset == "Games":
        if spec.model_dir == "Games-grec-sft-qwen4B-4-256-dsz0":
            found = lookup_best_from_csv(spec.model_dir, games_sft_best)
            if found is None:
                entries = collect_metrics_dir(spec.model_dir)
                best = select_best(entries)
            else:
                ckpt, metrics = found
                best = (ckpt, {**metrics, **read_metrics_json(RESULTS_DIR / spec.model_dir / ckpt / "metrics.json")})
        else:
            found = lookup_best_from_csv(spec.model_dir or "", games_rl_best)
            if found is None:
                entries = collect_metrics_dir(spec.model_dir or "")
                best = select_best(entries)
            else:
                ckpt, metrics = found
                best = (ckpt, {**metrics, **read_metrics_json(RESULTS_DIR / (spec.model_dir or "") / ckpt / "metrics.json")})
    elif spec.dataset == "Arts":
        if spec.model_dir is None and spec.column_name == "GenRec(sft)":
            best = build_arts_sft_best()
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
    baseline_candidates = [item for item in measured if item[0] not in {"GenRec(sft)", "GenRec(rule)", "GenRec(ranking)", "GenRec(fixed)", "GenRec(fixed + ce0.005)"}]
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
    requested_cols = {
        "GenRec(sft)",
        "GenRec(rule)",
        "GenRec(ranking)",
        "GenRec(fixed)",
        "GenRec(fixed + ce0.005)",
    }
    for row in dataset_rows:
        spec: RunSpec = row["spec"]  # type: ignore[assignment]
        if spec.column_name in requested_cols and row["status"] == "ok":
            return True
    return False


def build_main_table(resolved: list[dict[str, object]]) -> str:
    visible_datasets = []
    for dataset in ["Instruments", "Games", "Arts"]:
        dataset_rows = [row for row in resolved if row["spec"].dataset == dataset]  # type: ignore[attr-defined]
        if dataset_has_measured_requested_method(dataset_rows):
            visible_datasets.append(dataset)

    parts: list[str] = []
    parts.append(r"\section{GenRec Index Baseline Tables}")
    parts.append("")
    parts.append(r"\subsection{Main Table}")
    parts.append("")
    parts.append(r"\begin{table}[H]")
    parts.append(r"\centering")
    parts.append(r"\scriptsize")
    parts.append(r"\setlength{\tabcolsep}{3pt}")
    parts.append(r"\renewcommand{\arraystretch}{1.08}")
    parts.append(r"\resizebox{\textwidth}{!}{%")
    parts.append(r"\begin{tabular}{llcccccccccc}")
    parts.append(r"\toprule")
    parts.append(r"Dataset & Metric & Caser & GRU4Rec & BERT4Rec & SASRec & TIGER & GenRec(sft) & GenRec(rule) & GenRec(ranking) & GenRec(fixed) & GenRec(fixed + ce0.005) \\")
    parts.append(r"\midrule")

    for dataset_idx, dataset in enumerate(visible_datasets):
        dataset_rows = [row for row in resolved if row["spec"].dataset == dataset]  # type: ignore[attr-defined]
        for idx, metric in enumerate(METRICS):
            best_cols, _ = best_highlights(dataset_rows, metric)
            cells = []
            for col in ["Caser", "GRU4Rec", "BERT4Rec", "SASRec", "TIGER", "GenRec(sft)", "GenRec(rule)", "GenRec(ranking)", "GenRec(fixed)", "GenRec(fixed + ce0.005)"]:
                row = next(row for row in dataset_rows if row["spec"].column_name == col)  # type: ignore[attr-defined]
                cells.append(render_cell(col, metric, row, best_cols))
            prefix = dataset if idx == 0 else ""
            parts.append(" & ".join([prefix, metric, *cells]) + r" \\")
        if dataset_idx != len(visible_datasets) - 1:
            parts.append(r"\midrule")

    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    parts.append(r"\caption{GenRec index 口径下当前已测结果的主表。主表只展示当前至少有一条已测目标方法的数据集；因此本轮先保留 Instruments 与 Games。当前仓库中尚未找到与本表同口径的 Caser / GRU4Rec / BERT4Rec / SASRec / TIGER 实测结果，这些列先保留为占位，避免混入 LC-Rec 论文原表的异口径数字。所有 GenRec 变体若存在多个 checkpoint，统一按 NDCG@10 选 best。}")
    parts.append(r"\label{tab:genrec-index-main}")
    parts.append(r"\end{table}")
    parts.append("")

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
    for dataset_idx, dataset in enumerate(visible_datasets):
        dataset_rows = [row for row in resolved if row["spec"].dataset == dataset and row["status"] == "ok"]  # type: ignore[attr-defined]
        for idx, row in enumerate(dataset_rows):
            spec: RunSpec = row["spec"]  # type: ignore[assignment]
            metrics: dict[str, float] = row["metrics"]  # type: ignore[assignment]
            cells = [fmt_metric(maybe_value(metrics, metric)) for metric in METRICS]
            parts.append(" & ".join([
                dataset if idx == 0 else "",
                spec.label.replace("GenRec ", "GenRec "),
                fmt_texttt(row["best_checkpoint"]),  # type: ignore[arg-type]
                *cells,
            ]) + r" \\")
        if dataset_idx != len(visible_datasets) - 1:
            parts.append(r"\midrule")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    parts.append(r"\caption{当前主表中所有已测方法的 best checkpoint 汇总。}")
    parts.append(r"\label{tab:genrec-index-best-summary}")
    parts.append(r"\end{table}")
    parts.append("")

    parts.append(r"\paragraph{Readout.} 本轮主表只纳入当前仓库里已经能追溯到 \texttt{GenRec index} 统一评测口径的结果。")
    parts.append(r"Instruments 上当前 best \texttt{NDCG@10} 仍是 \texttt{GenRec(rule)} 的 \texttt{0.0960}；")
    parts.append(r"如果看 coverage，\texttt{GenRec(fixed + ce0.005)} 在当前 best 点达到 \texttt{HR@50=0.1985}。")
    parts.append(r"Games 上当前已经测出的最强线是 \texttt{GenRec(fixed)}，best 点为 \texttt{checkpoint-8752 / NDCG@10=0.0480 / HR@50=0.1972}。")
    parts.append(r"Arts 当前仓库里只有 LC-Rec aligned SFT 的 GenRec 评测结果，还没有你这次想放进主表的 GenRec RL 变体，所以 Arts 先只放在附录。")
    return "\n".join(parts) + "\n"


def appendix_run_specs() -> list[RunSpec]:
    return [
        RunSpec("Instruments", "GenRec (SFT)", "GenRec(sft)", model_dir="Instruments-grec-sft-qwen4B-4-256-dsz0"),
        RunSpec("Instruments", "GenRec (rule)", "GenRec(rule)", model_dir="Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495"),
        RunSpec("Instruments", "GenRec (ranking)", "GenRec(ranking)", model_dir="Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495"),
        RunSpec("Instruments", "GenRec (fixed)", "GenRec(fixed)", model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495"),
        RunSpec("Instruments", "GenRec (fixed + ce0.005)", "GenRec(fixed + ce0.005)", model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495"),
        RunSpec("Games", "GenRec (SFT)", "GenRec(sft)", model_dir="Games-grec-sft-qwen4B-4-256-dsz0"),
        RunSpec("Games", "GenRec (rule)", "GenRec(rule)", model_dir="Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896"),
        RunSpec("Games", "GenRec (fixed)", "GenRec(fixed)", model_dir="Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896"),
        RunSpec("Instruments", "LC-Rec aligned SFT", "LC-Rec aligned SFT", model_dir="Instruments-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu"),
        RunSpec("Games", "LC-Rec aligned SFT", "LC-Rec aligned SFT", model_dir="Games-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu"),
        RunSpec("Arts", "LC-Rec aligned SFT", "LC-Rec aligned SFT", model_dir="Arts-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu"),
    ]


def render_appendix_table(spec: RunSpec) -> str:
    entries = collect_metrics_dir(spec.model_dir or "")
    if not entries:
        return ""

    parts: list[str] = []
    safe_label = re.sub(r"[^a-z0-9]+", "-", f"{spec.dataset}-{spec.label}".lower()).strip("-")
    parts.append(rf"\subsection{{{spec.dataset}: {spec.label}}}")
    parts.append("")
    parts.append(rf"\begin{{table}}[H]")
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
    parts.append(rf"\caption{{{spec.dataset} 上 \texttt{{{(spec.model_dir or '').replace('_', r'\_')}}} 的完整 checkpoint 指标。}}")
    parts.append(rf"\label{{tab:appendix-{safe_label}}}")
    parts.append(r"\end{table}")
    parts.append("")
    return "\n".join(parts)


def build_appendix() -> str:
    parts: list[str] = []
    parts.append(r"\appendix")
    parts.append(r"\section{Full Checkpoint Tables}")
    parts.append("")
    parts.append(r"本附录按结果目录逐条展开当前已经同步回仓库的完整 checkpoint 指标，方便后续继续补充主表、做 early-stop 对照，或替换 best 选点。")
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

\begin{document}

\input{genrec-baseline-main}
\input{genrec-baseline-appendix}

\end{document}
"""


def main() -> None:
    instruments_best = load_csv_rows(INSTRUMENTS_BEST_CSV)
    games_sft_best = load_csv_rows(GAMES_SFT_BEST_CSV)
    games_rl_best = load_csv_rows(GAMES_RL_BEST_CSV)
    resolved = [resolve_run(spec, instruments_best, games_sft_best, games_rl_best) for spec in RUN_SPECS]

    MAIN_TEX.write_text(build_main_table(resolved))
    APPENDIX_TEX.write_text(build_appendix())
    MASTER_TEX.write_text(build_master())


if __name__ == "__main__":
    main()
