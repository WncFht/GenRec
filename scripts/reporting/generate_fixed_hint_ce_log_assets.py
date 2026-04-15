#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_LOG_PATH = Path(
    "/Users/fanghaotian/Desktop/src/GenRec/log/"
    "instruments_grec_rl_rule_only_fixed_hint_taskfix_b16_hint_ce_ckpt495_20260414_041008.log"
)
DEFAULT_ASSET_DIR = Path(
    "/Users/fanghaotian/Desktop/src/GenRec/docs/assets/2026-04-15-instruments-fixed-hint-ce-log-analysis"
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
INFO_RE = re.compile(r"^\[INFO\]\s+([A-Z0-9_]+)=(.+)$")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def parse_log(log_path: Path) -> tuple[dict[str, str], list[dict], list[dict]]:
    info: dict[str, str] = {}
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_ansi(raw_line).strip()
        if not line:
            continue

        info_match = INFO_RE.match(line)
        if info_match:
            key, value = info_match.groups()
            info[key] = value.strip()
            continue

        if "{" not in line or "}" not in line:
            continue

        payload_text = line[line.find("{") : line.rfind("}") + 1]
        try:
            payload = ast.literal_eval(payload_text)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue

        if "eval_loss" in payload:
            eval_rows.append(payload)
        elif "loss" in payload and "reward" in payload and "epoch" in payload:
            train_rows.append(payload)

    return info, train_rows, eval_rows


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_dataframes(info: dict[str, str], train_rows: list[dict], eval_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hint_ce_loss_coef = _to_float(info.get("HINT_CE_LOSS_COEF"), 0.0)
    grad_acc = max(int(float(info.get("GRAD_ACC", "1"))), 1)

    train_df = pd.DataFrame(train_rows).copy()
    if not train_df.empty:
        train_df["global_step"] = range(1, len(train_df) + 1)
        train_df["hint_ce_loss_coef"] = hint_ce_loss_coef
        train_df["grad_acc"] = grad_acc
        train_df["weighted_hint_ce_loss"] = train_df["hint_ce/loss"].astype(float) * hint_ce_loss_coef / grad_acc
        train_df["estimated_rl_base_loss"] = train_df["loss"].astype(float) - train_df["weighted_hint_ce_loss"]
        train_df["hint_ce_fraction"] = train_df["weighted_hint_ce_loss"] / train_df["loss"].replace(0, pd.NA)
        train_df["rl_base_fraction"] = train_df["estimated_rl_base_loss"] / train_df["loss"].replace(0, pd.NA)
        train_df["hint_ce_fraction"] = train_df["hint_ce_fraction"].astype(float)
        train_df["rl_base_fraction"] = train_df["rl_base_fraction"].astype(float)

    eval_df = pd.DataFrame(eval_rows).copy()
    if not eval_df.empty:
        eval_df["eval_index"] = range(1, len(eval_df) + 1)
        eval_steps = max(int(float(info.get("EVAL_STEP", "1"))), 1)
        eval_df["approx_global_step"] = eval_df["eval_index"] * eval_steps

    return train_df, eval_df


def build_summary(info: dict[str, str], train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "log_path": str(DEFAULT_LOG_PATH),
        "hint_ce_loss_coef": _to_float(info.get("HINT_CE_LOSS_COEF"), 0.0),
        "grad_acc": max(int(float(info.get("GRAD_ACC", "1"))), 1),
        "effective_hint_ce_multiplier": _to_float(info.get("HINT_CE_LOSS_COEF"), 0.0)
        / max(int(float(info.get("GRAD_ACC", "1"))), 1),
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
    }
    if not train_df.empty:
        summary["train"] = {
            "epoch_min": float(train_df["epoch"].min()),
            "epoch_max": float(train_df["epoch"].max()),
            "loss_min": float(train_df["loss"].min()),
            "loss_median": float(train_df["loss"].median()),
            "loss_max": float(train_df["loss"].max()),
            "hint_ce_loss_min": float(train_df["hint_ce/loss"].min()),
            "hint_ce_loss_median": float(train_df["hint_ce/loss"].median()),
            "hint_ce_loss_max": float(train_df["hint_ce/loss"].max()),
            "weighted_hint_ce_min": float(train_df["weighted_hint_ce_loss"].min()),
            "weighted_hint_ce_median": float(train_df["weighted_hint_ce_loss"].median()),
            "weighted_hint_ce_max": float(train_df["weighted_hint_ce_loss"].max()),
            "hint_ce_fraction_median": float(train_df["hint_ce_fraction"].median()),
            "hint_ce_fraction_p90": float(train_df["hint_ce_fraction"].quantile(0.9)),
            "hint_ce_fraction_max": float(train_df["hint_ce_fraction"].max()),
            "entropy_min": float(train_df["entropy"].min()),
            "entropy_max": float(train_df["entropy"].max()),
            "reward_min": float(train_df["reward"].min()),
            "reward_max": float(train_df["reward"].max()),
        }
    if not eval_df.empty:
        summary["eval"] = {
            "epoch_min": float(eval_df["epoch"].min()),
            "epoch_max": float(eval_df["epoch"].max()),
            "reward_start": float(eval_df["eval_reward"].iloc[0]),
            "reward_end": float(eval_df["eval_reward"].iloc[-1]),
            "reward_max": float(eval_df["eval_reward"].max()),
            "reward_best_epoch": float(eval_df.loc[eval_df["eval_reward"].idxmax(), "epoch"]),
            "entropy_start": float(eval_df["eval_entropy"].iloc[0]),
            "entropy_end": float(eval_df["eval_entropy"].iloc[-1]),
        }
    return summary


def _rolling(series: pd.Series, window: int = 50) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_loss_components(train_df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(train_df["epoch"], train_df["loss"], color="#1d3557", alpha=0.25, linewidth=1)
    axes[0].plot(train_df["epoch"], _rolling(train_df["loss"]), color="#1d3557", linewidth=2, label="Train total loss")
    axes[0].plot(
        train_df["epoch"],
        _rolling(train_df["estimated_rl_base_loss"]),
        color="#2a9d8f",
        linewidth=2,
        label="Estimated RL base loss",
    )
    axes[0].plot(
        train_df["epoch"],
        _rolling(train_df["weighted_hint_ce_loss"]),
        color="#d62828",
        linewidth=2,
        label="Weighted hint CE add-on",
    )
    axes[0].set_title("Loss Components vs Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(train_df["epoch"], train_df["hint_ce/loss"], color="#f4a261", alpha=0.25, linewidth=1)
    axes[1].plot(train_df["epoch"], _rolling(train_df["hint_ce/loss"]), color="#f4a261", linewidth=2)
    axes[1].set_title("Raw Hint CE Loss vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Hint CE loss")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_loss_fraction(train_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.stackplot(
        train_df["epoch"],
        _rolling(train_df["rl_base_fraction"].clip(lower=0.0, upper=1.0), window=100),
        _rolling(train_df["hint_ce_fraction"].clip(lower=0.0, upper=1.0), window=100),
        labels=["RL base fraction", "Weighted hint CE fraction"],
        colors=["#2a9d8f", "#d62828"],
        alpha=0.85,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Loss Fraction vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of train loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_entropy(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(train_df["epoch"], train_df["entropy"], color="#4c78a8", alpha=0.20, linewidth=1)
    ax.plot(train_df["epoch"], _rolling(train_df["entropy"]), color="#4c78a8", linewidth=2, label="Train entropy")
    if not eval_df.empty:
        ax.plot(eval_df["epoch"], eval_df["eval_entropy"], color="#e76f51", linewidth=1.5, marker="o", label="Eval entropy")
    ax.set_title("Entropy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_train_reward(train_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(train_df["epoch"], train_df["reward"], color="#2a9d8f", alpha=0.20, linewidth=1)
    ax.plot(train_df["epoch"], _rolling(train_df["reward"]), color="#2a9d8f", linewidth=2, label="Train reward")
    ax.set_title("Train Reward vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_eval_reward(eval_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(eval_df["epoch"], eval_df["eval_reward"], color="#e76f51", linewidth=2, marker="o", label="Eval reward")
    ax.plot(
        eval_df["epoch"],
        eval_df["eval_reward_std"],
        color="#8d99ae",
        linewidth=1.5,
        marker="o",
        linestyle="--",
        label="Eval reward std",
    )
    ax.set_title("Eval Reward vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info, train_rows, eval_rows = parse_log(args.log_path)
    train_df, eval_df = build_dataframes(info, train_rows, eval_rows)
    summary = build_summary(info, train_df, eval_df)

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    train_csv = args.asset_dir / "train_metrics.csv"
    eval_csv = args.asset_dir / "eval_metrics.csv"
    summary_json = args.asset_dir / "summary.json"
    train_df.to_csv(train_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    loss_path = plot_loss_components(train_df, args.asset_dir / "loss-components-vs-epoch.png")
    ratio_path = plot_loss_fraction(train_df, args.asset_dir / "loss-fraction-vs-epoch.png")
    entropy_path = plot_entropy(train_df, eval_df, args.asset_dir / "entropy-vs-epoch.png")
    train_reward_path = plot_train_reward(train_df, args.asset_dir / "train-reward-vs-epoch.png")
    eval_reward_path = plot_eval_reward(eval_df, args.asset_dir / "eval-reward-vs-epoch.png")

    print(f"train_csv={train_csv}")
    print(f"eval_csv={eval_csv}")
    print(f"summary_json={summary_json}")
    print(f"loss_plot={loss_path}")
    print(f"loss_fraction_plot={ratio_path}")
    print(f"entropy_plot={entropy_path}")
    print(f"train_reward_plot={train_reward_path}")
    print(f"eval_reward_plot={eval_reward_path}")


if __name__ == "__main__":
    main()
