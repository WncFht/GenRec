#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
SFT_METRICS_PATH = RESULTS_ROOT / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"
METRIC_COLUMNS = ["NDCG@10", "HR@10", "NDCG@50", "HR@50"]


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    model_dir: str
    color: str
    launcher_path: str | None = None
    marker: str = "o"
    linestyle: str = "-"
    linewidth: float = 2.2
    epoch_max_step: int | None = None
    epoch_max_step_ref_key: str | None = None


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


def _variant_map(specs: list[VariantSpec]) -> dict[str, VariantSpec]:
    return {spec.key: spec for spec in specs}


def _resolve_max_step(spec: VariantSpec, specs: list[VariantSpec], observed_max_step: int) -> int:
    if spec.epoch_max_step is not None:
        return spec.epoch_max_step
    if spec.epoch_max_step_ref_key is not None:
        ref = _variant_map(specs)[spec.epoch_max_step_ref_key]
        ref_root = RESULTS_ROOT / ref.model_dir
        ref_steps = sorted(int(path.parent.name.split("-")[-1]) for path in ref_root.glob("checkpoint-*/metrics.json"))
        if ref_steps:
            return max(ref_steps)
    return observed_max_step


def load_variant_rows(spec: VariantSpec, specs: list[VariantSpec]) -> list[dict[str, object]]:
    root = RESULTS_ROOT / spec.model_dir
    metrics_files = sorted(root.glob("checkpoint-*/metrics.json"), key=lambda path: int(path.parent.name.split("-")[-1]))
    if not metrics_files:
        return []

    observed_steps = [int(path.parent.name.split("-")[-1]) for path in metrics_files]
    observed_max_step = max(observed_steps)
    max_step = _resolve_max_step(spec, specs, observed_max_step)

    rows: list[dict[str, object]] = []
    for path in metrics_files:
        step = int(path.parent.name.split("-")[-1])
        metrics = json.loads(path.read_text())
        row: dict[str, object] = {
            "variant_key": spec.key,
            "variant_label": spec.label,
            "model_dir": spec.model_dir,
            "checkpoint": f"checkpoint-{step}",
            "step": step,
            "max_step": max_step,
            "observed_max_step": observed_max_step,
            "epoch_progress": step / max_step * 2.0 if max_step else 0.0,
        }
        for key in METRIC_COLUMNS + ["NDCG@5", "HR@5"]:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def build_dataframe(specs: list[VariantSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        rows.extend(load_variant_rows(spec, specs))
    return pd.DataFrame(rows)


def build_best_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_key in df["variant_key"].unique():
        variant_df = df[df["variant_key"] == variant_key]
        best_row = variant_df.sort_values(["NDCG@10", "HR@10", "NDCG@50", "HR@50"], ascending=False).iloc[0]
        peak_row = variant_df.sort_values(["HR@50", "NDCG@10", "HR@10", "NDCG@50"], ascending=False).iloc[0]
        row = best_row.to_dict()
        row["peak_hr50_checkpoint"] = peak_row["checkpoint"]
        row["peak_hr50_epoch"] = peak_row["epoch_progress"]
        row["peak_hr50"] = peak_row["HR@50"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_metadata(path: Path, specs: list[VariantSpec]) -> None:
    rows = []
    for spec in specs:
        rows.append(asdict(spec))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def _plot_variant(ax: plt.Axes, df: pd.DataFrame, spec: VariantSpec, metric: str, x_column: str) -> None:
    variant_df = df[df["variant_key"] == spec.key]
    if variant_df.empty:
        return
    ax.plot(
        variant_df[x_column],
        variant_df[metric],
        label=spec.label,
        color=spec.color,
        marker=spec.marker,
        linestyle=spec.linestyle,
        linewidth=spec.linewidth,
        markersize=4,
    )


def plot_metric_grid(
    df: pd.DataFrame,
    specs: list[VariantSpec],
    variant_keys: list[str],
    title: str,
    out_path: Path,
    sft_reference: dict[str, float | str],
    *,
    x_column: str = "epoch_progress",
    x_label: str = "Epoch",
    legend_cols: int = 3,
    figsize: tuple[float, float] = (11, 8),
    dpi: int = 180,
) -> Path:
    spec_map = _variant_map(specs)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    metrics = [("NDCG@10", "NDCG@10"), ("HR@10", "HR@10"), ("NDCG@50", "NDCG@50"), ("HR@50", "HR@50")]

    for ax, (metric, metric_title) in zip(axes.flat, metrics, strict=True):
        for key in variant_keys:
            _plot_variant(ax, df, spec_map[key], metric, x_column)
        ax.axhline(float(sft_reference[metric]), linestyle="--", linewidth=1.4, color="#6b7280", label="SFT495")
        ax.set_title(metric_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=legend_cols,
        frameon=False,
    )
    fig.suptitle(title, y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
