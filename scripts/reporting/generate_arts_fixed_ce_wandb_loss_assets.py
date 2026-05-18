#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "docs" / "baseline" / "table" / "assets"
PROJECT_PATH = "wncfht/MIMIGenRec-GRPO"
DEFAULT_RUN_IDS = ("x1s1ksog", "zw2nici2", "1zl0r0up")
TRAIN_KEYS = [
    "train/global_step",
    "train/epoch",
    "train/loss",
    "train/loss/rl_base",
    "train/loss/hint_ce_weighted",
    "train/hint_ce/loss",
    "train/hint_ce/token_count",
    "train/reward",
    "train/grad_norm",
]
ROLLING_WINDOW = 75

METRIC_STYLES = {
    "train/loss/rl_base": {"color": "#1d3557", "label": "RL base"},
    "train/loss/hint_ce_weighted": {"color": "#d62828", "label": "Weighted hint CE"},
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    coef: float
    coef_label: str
    run_name: str
    output_dir: str
    output_dir_name: str


def _coef_label(value: float) -> str:
    if value in {0.001, 0.005, 0.01}:
        return {0.001: "0.001", 0.005: "0.005", 0.01: "0.01"}[value]
    return f"{value:g}"


def infer_coef_from_output_dir(output_dir: str | None) -> float:
    if not output_dir:
        raise ValueError("Missing output_dir; cannot infer CE coefficient.")
    output_dir_name = Path(output_dir).name
    if output_dir_name == "Arts-grec-genrec-fixed-ce-from-sft":
        return 0.005
    match = re.fullmatch(r"Arts-grec-genrec-fixed-ce-(\d+)-from-sft", output_dir_name)
    if not match:
        raise ValueError(f"Unsupported Arts fixed-CE output_dir format: {output_dir_name}")
    token = match.group(1)
    token_map = {
        "0001": 0.001,
        "001": 0.01,
    }
    if token not in token_map:
        raise ValueError(f"Unknown CE token {token!r} from output_dir {output_dir_name}")
    return token_map[token]


def fetch_run_dataframe(api: wandb.Api, run_id: str) -> tuple[RunSpec, pd.DataFrame]:
    run = api.run(f"{PROJECT_PATH}/{run_id}")
    output_dir = str(run.config.get("output_dir") or "")
    coef = infer_coef_from_output_dir(output_dir)
    spec = RunSpec(
        run_id=run_id,
        coef=coef,
        coef_label=_coef_label(coef),
        run_name=str(run.name),
        output_dir=output_dir,
        output_dir_name=Path(output_dir).name,
    )

    df = run.history(keys=TRAIN_KEYS, samples=10000, pandas=True)
    if df.empty:
        raise ValueError(f"No train history returned for run {run_id}")

    for column in df.columns:
        if column != "_step":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["train/epoch", "train/loss/rl_base", "train/loss/hint_ce_weighted"]).copy()
    if df.empty:
        raise ValueError(f"Run {run_id} has no usable rows for rl_base + hint_ce_weighted")

    df["run_id"] = spec.run_id
    df["coef"] = spec.coef
    df["coef_label"] = spec.coef_label
    df["run_name"] = spec.run_name
    df["output_dir_name"] = spec.output_dir_name
    df["abs_rl_base"] = df["train/loss/rl_base"].abs()
    df["ce_over_abs_rl"] = df["train/loss/hint_ce_weighted"] / df["abs_rl_base"].replace(0, pd.NA)
    return spec, df


def rolling(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def build_summary(spec: RunSpec, df: pd.DataFrame) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "run_name": spec.run_name,
        "coef": spec.coef,
        "coef_label": spec.coef_label,
        "output_dir_name": spec.output_dir_name,
        "row_count": int(len(df)),
        "epoch_min": float(df["train/epoch"].min()),
        "epoch_max": float(df["train/epoch"].max()),
        "rl_base_median": float(df["train/loss/rl_base"].median()),
        "rl_base_abs_median": float(df["abs_rl_base"].median()),
        "weighted_hint_ce_median": float(df["train/loss/hint_ce_weighted"].median()),
        "weighted_hint_ce_mean": float(df["train/loss/hint_ce_weighted"].mean()),
        "ce_over_abs_rl_median": float(df["ce_over_abs_rl"].dropna().median()),
        "ce_gt_abs_rl_fraction": float((df["train/loss/hint_ce_weighted"] > df["abs_rl_base"]).mean()),
    }


def plot_run(spec: RunSpec, df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    epoch = df["train/epoch"]
    rl_base = df["train/loss/rl_base"]
    hint_ce_weighted = df["train/loss/hint_ce_weighted"]

    ax.plot(
        epoch,
        rl_base,
        color=METRIC_STYLES["train/loss/rl_base"]["color"],
        alpha=0.18,
        linewidth=0.8,
    )
    ax.plot(
        epoch,
        rolling(rl_base),
        color=METRIC_STYLES["train/loss/rl_base"]["color"],
        linewidth=2.2,
        label=f"{METRIC_STYLES['train/loss/rl_base']['label']} ({ROLLING_WINDOW}-pt rolling)",
    )
    ax.plot(
        epoch,
        hint_ce_weighted,
        color=METRIC_STYLES["train/loss/hint_ce_weighted"]["color"],
        alpha=0.18,
        linewidth=0.8,
    )
    ax.plot(
        epoch,
        rolling(hint_ce_weighted),
        color=METRIC_STYLES["train/loss/hint_ce_weighted"]["color"],
        linewidth=2.2,
        label=f"{METRIC_STYLES['train/loss/hint_ce_weighted']['label']} ({ROLLING_WINDOW}-pt rolling)",
    )

    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlim(left=0.0, right=max(2.0, float(epoch.max())))
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss component")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    summary = build_summary(spec, df)
    stats_text = (
        f"coef={spec.coef_label}\n"
        f"weighted CE median={summary['weighted_hint_ce_median']:.4f}\n"
        f"|RL base| median={summary['rl_base_abs_median']:.4f}\n"
        f"CE > |RL| frac={summary['ce_gt_abs_rl_fraction']:.2%}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#d0d0d0"},
    )

    fig.suptitle(f"Arts fixed-CE train loss components | coef={spec.coef_label} | run {spec.run_id}", y=0.98)
    fig.text(
        0.5,
        0.935,
        f"{spec.output_dir_name} | raw traces + rolling means",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)

    run_results: list[tuple[RunSpec, pd.DataFrame]] = []
    for run_id in DEFAULT_RUN_IDS:
        run_results.append(fetch_run_dataframe(api, run_id))
    run_results.sort(key=lambda item: item[0].coef)

    history_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    figure_paths: list[Path] = []
    for spec, df in run_results:
        history_frames.append(df)
        summaries.append(build_summary(spec, df))
        coef_slug = spec.coef_label.replace(".", "p")
        figure_path = ASSET_DIR / f"arts-fixed-ce-wandb-loss-components-coef-{coef_slug}.png"
        figure_paths.append(plot_run(spec, df, figure_path))

    combined_history = pd.concat(history_frames, ignore_index=True)
    history_csv_path = ASSET_DIR / "arts-fixed-ce-wandb-loss-components-history.csv"
    summary_json_path = ASSET_DIR / "arts-fixed-ce-wandb-loss-components-summary.json"
    combined_history.to_csv(history_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"history_csv={history_csv_path}")
    print(f"summary_json={summary_json_path}")
    for path in figure_paths:
        print(f"figure_png={path}")


if __name__ == "__main__":
    main()
