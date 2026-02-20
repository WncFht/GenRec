#!/usr/bin/env python3
"""Incremental SFT checkpoint evaluator with persistent W&B logging.

This sidecar watches an SFT output directory for new checkpoint-* folders, evaluates
new checkpoints via evaluate_sft_3b.sh, and appends metrics into one long-lived
W&B eval run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
CB_RE = re.compile(r"(cb\d+(?:-\d+)*)")
METRIC_KEYS = [
    "HR@1",
    "HR@3",
    "HR@5",
    "HR@10",
    "NDCG@1",
    "NDCG@3",
    "NDCG@5",
    "NDCG@10",
]
TABLE_COLUMNS = ["checkpoint_step", "checkpoint_name", *METRIC_KEYS, "metrics_path"]


@dataclass(frozen=True)
class CheckpointInfo:
    step: int
    name: str
    path: Path


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sanitize_token(value: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-_.")
    return cleaned if cleaned else fallback


def resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def parse_checkpoint_step(name: str) -> int | None:
    match = CHECKPOINT_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def discover_checkpoints(sft_root: Path) -> list[CheckpointInfo]:
    checkpoints: list[CheckpointInfo] = []

    root_step = parse_checkpoint_step(sft_root.name)
    if root_step is not None and sft_root.is_dir():
        checkpoints.append(CheckpointInfo(step=root_step, name=sft_root.name, path=sft_root))
        return checkpoints

    if not sft_root.is_dir():
        return checkpoints

    for entry in sft_root.iterdir():
        if not entry.is_dir():
            continue
        step = parse_checkpoint_step(entry.name)
        if step is None:
            continue
        checkpoints.append(CheckpointInfo(step=step, name=entry.name, path=entry))

    checkpoints.sort(key=lambda item: item.step)
    return checkpoints


def checkpoint_is_ready(ckpt: CheckpointInfo, min_age_seconds: int) -> bool:
    try:
        stat = ckpt.path.stat()
    except FileNotFoundError:
        return False

    age = time.time() - stat.st_mtime
    if age < min_age_seconds:
        return False

    try:
        next(ckpt.path.iterdir())
    except (StopIteration, FileNotFoundError):
        return False

    return True


def metrics_output_dir(repo_root: Path, ckpt: CheckpointInfo) -> Path:
    model_parent = ckpt.path.parent.name
    return repo_root / "results" / model_parent / ckpt.name


def metrics_json_path(repo_root: Path, ckpt: CheckpointInfo) -> Path:
    return metrics_output_dir(repo_root, ckpt) / "metrics.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def stable_hash(payload: dict[str, Any], length: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def infer_model_name(sft_root: Path) -> str:
    step = parse_checkpoint_step(sft_root.name)
    if step is not None:
        return sft_root.parent.name
    return sft_root.name


def infer_dataset(category: str, test_data_path: Path) -> str:
    # Typical path: data/<dataset_variant>/sft/test.json
    parent = test_data_path.parent
    if parent.name == "sft" and parent.parent.name:
        return parent.parent.name
    if parent.name:
        return parent.name
    return category


def infer_eval_split(test_data_path: Path) -> str:
    return test_data_path.stem or "test"


def infer_cb_setting(candidates: list[str]) -> str:
    for value in candidates:
        match = CB_RE.search(value)
        if match:
            return match.group(1)
    return "unknown"


@contextmanager
def exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as fp:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another sidecar instance holds lock: {lock_path}") from exc

        fp.seek(0)
        fp.truncate(0)
        fp.write(str(os.getpid()))
        fp.flush()

        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def normalize_state(
    state: dict[str, Any],
    *,
    model_config_key: str,
    run_id: str,
    run_name: str,
) -> dict[str, Any]:
    processed_steps = sorted({int(step) for step in state.get("processed_steps", [])})
    table_rows = state.get("table_rows", [])
    if not isinstance(table_rows, list):
        table_rows = []

    normalized = {
        "version": 1,
        "model_config_key": model_config_key,
        "run_id": run_id,
        "run_name": run_name,
        "processed_steps": processed_steps,
        "failed_steps": state.get("failed_steps", {}),
        "last_seen_step": state.get("last_seen_step"),
        "last_update_time": state.get("last_update_time", now_utc_iso()),
        "table_rows": table_rows,
    }
    if not isinstance(normalized["failed_steps"], dict):
        normalized["failed_steps"] = {}
    return normalized


def upsert_table_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    step = int(new_row["checkpoint_step"])
    by_step = {int(row.get("checkpoint_step", -1)): row for row in rows}
    by_step[step] = new_row
    merged = [by_step[key] for key in sorted(by_step.keys()) if key >= 0]
    return merged


def build_summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not rows:
        return summary

    sorted_rows = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
    last = sorted_rows[-1]

    summary["last_step"] = int(last["checkpoint_step"])
    if last.get("HR@10") is not None:
        summary["last_hr@10"] = float(last["HR@10"])
    if last.get("NDCG@10") is not None:
        summary["last_ndcg@10"] = float(last["NDCG@10"])

    best_hr = None
    best_hr_step = None
    best_ndcg = None
    best_ndcg_step = None
    for row in sorted_rows:
        step = int(row["checkpoint_step"])
        hr = row.get("HR@10")
        ndcg = row.get("NDCG@10")

        if hr is not None and (best_hr is None or float(hr) > best_hr):
            best_hr = float(hr)
            best_hr_step = step

        if ndcg is not None and (best_ndcg is None or float(ndcg) > best_ndcg):
            best_ndcg = float(ndcg)
            best_ndcg_step = step

    if best_hr is not None:
        summary["best_hr@10"] = best_hr
        summary["best_step_for_hr@10"] = best_hr_step
    if best_ndcg is not None:
        summary["best_ndcg@10"] = best_ndcg
        summary["best_step_for_ndcg@10"] = best_ndcg_step

    return summary


def load_metrics(metrics_path: Path) -> dict[str, float]:
    raw = load_json(metrics_path)
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            parsed[key] = float(value)
    if not parsed:
        raise ValueError(f"No numeric metrics found in {metrics_path}")
    return parsed


def run_single_checkpoint_eval(
    args: argparse.Namespace,
    ckpt: CheckpointInfo,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CATEGORY": args.category,
            "CUDA_LIST": args.cuda_list,
            "PYTHON_BIN": args.python_bin,
            "TEST_DATA_PATH": str(args.test_data_path),
            "INDEX_PATH": str(args.index_path),
            "BATCH_SIZE": str(args.batch_size),
            "MAX_NEW_TOKENS": str(args.max_new_tokens),
            "NUM_BEAMS": str(args.num_beams),
            "TEMPERATURE": str(args.temperature),
            "DO_SAMPLE": str(args.do_sample),
            "LENGTH_PENALTY": str(args.length_penalty),
            "SID_LEVELS": str(args.sid_levels),
            "CKPT_LIST": str(ckpt.path),
        }
    )

    command = ["bash", str(args.eval_script), str(args.sft_root)]
    logging.info("Evaluating %s via %s", ckpt.path, " ".join(command))
    subprocess.run(command, cwd=str(args.repo_root), env=env, check=True)


def init_wandb(
    args: argparse.Namespace,
    run_id: str,
    run_name: str,
    config: dict[str, Any],
):
    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed in current environment; install wandb or run with --disable-wandb"
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=run_id,
        name=run_name,
        resume=args.wandb_resume,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        config=config,
    )

    wandb.define_metric("checkpoint_step")
    wandb.define_metric("eval/*", step_metric="checkpoint_step")

    return wandb, run


def log_metrics_to_wandb(
    wandb_module,
    run,
    *,
    ckpt: CheckpointInfo,
    metrics: dict[str, float],
    metrics_path: Path,
    table_rows: list[dict[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "checkpoint_step": ckpt.step,
        "eval/checkpoint_name": ckpt.name,
        "eval/metrics_path": str(metrics_path),
    }
    for key, value in metrics.items():
        payload[f"eval/{key}"] = float(value)

    run.log(payload)

    table = wandb_module.Table(columns=TABLE_COLUMNS)
    for row in sorted(table_rows, key=lambda item: int(item["checkpoint_step"])):
        table.add_data(*[row.get(column) for column in TABLE_COLUMNS])
    run.log({"eval/checkpoint_table": table})

    summary = build_summary_from_rows(table_rows)
    for key, value in summary.items():
        run.summary[key] = value


def build_row(ckpt: CheckpointInfo, metrics: dict[str, float], metrics_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_step": ckpt.step,
        "checkpoint_name": ckpt.name,
        "metrics_path": str(metrics_path),
    }
    for key in METRIC_KEYS:
        row[key] = float(metrics[key]) if key in metrics else None
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch SFT checkpoints, evaluate incrementally, and log to one W&B run."
    )

    default_repo_root = Path(__file__).resolve().parent

    parser.add_argument("--repo-root", default=str(default_repo_root), help="GenRec repo root")
    parser.add_argument("--sft-root", required=True, help="SFT output root containing checkpoint-* folders")
    parser.add_argument("--eval-script", default="evaluate_sft_3b.sh", help="Path to evaluate shell script")

    parser.add_argument("--category", default=os.environ.get("CATEGORY", "Industrial_and_Scientific"))
    parser.add_argument(
        "--test-data-path", default=os.environ.get("TEST_DATA_PATH", "data/Industrial_and_Scientific/sft/test.json")
    )
    parser.add_argument(
        "--index-path", default=os.environ.get("INDEX_PATH", "data/Industrial_and_Scientific/id2sid.json")
    )

    parser.add_argument("--cuda-list", default=os.environ.get("CUDA_LIST", "0"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "256")))
    parser.add_argument("--num-beams", type=int, default=int(os.environ.get("NUM_BEAMS", "50")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "1.0")))
    parser.add_argument("--do-sample", default=os.environ.get("DO_SAMPLE", "False"))
    parser.add_argument("--length-penalty", type=float, default=float(os.environ.get("LENGTH_PENALTY", "0.0")))
    parser.add_argument("--sid-levels", type=int, default=int(os.environ.get("SID_LEVELS", "-1")))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", default="")
    parser.add_argument("--cb-setting", default="")
    parser.add_argument("--eval-split", default="")

    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--checkpoint-ready-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true", help="Process pending checkpoints once then exit")
    parser.add_argument(
        "--force-eval", action="store_true", help="Always run evaluate script even if metrics.json exists"
    )
    parser.add_argument("--max-pending-per-cycle", type=int, default=0, help="0 means unlimited")

    parser.add_argument("--state-dir", default="state/wandb_eval_sidecar")

    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "MIMIGenRec-Eval"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME", ""))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb-resume", default=os.environ.get("WANDB_RESUME", "allow"))
    parser.add_argument("--wandb-job-type", default=os.environ.get("WANDB_JOB_TYPE", "eval"))

    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.repo_root = resolve_path(args.repo_root, Path.cwd())
    args.sft_root = resolve_path(args.sft_root, args.repo_root)
    args.eval_script = resolve_path(args.eval_script, args.repo_root)
    args.test_data_path = resolve_path(args.test_data_path, args.repo_root)
    args.index_path = resolve_path(args.index_path, args.repo_root)
    args.state_dir = resolve_path(args.state_dir, args.repo_root)

    if not args.eval_script.is_file():
        raise FileNotFoundError(f"evaluate script not found: {args.eval_script}")
    if not args.sft_root.exists():
        raise FileNotFoundError(f"SFT root not found: {args.sft_root}")
    if not args.test_data_path.is_file():
        raise FileNotFoundError(f"test data not found: {args.test_data_path}")
    if not args.index_path.is_file():
        raise FileNotFoundError(f"index file not found: {args.index_path}")

    return args


def main() -> int:
    args = parse_args()

    model_name = infer_model_name(args.sft_root)
    dataset = args.dataset or infer_dataset(args.category, args.test_data_path)
    eval_split = args.eval_split or infer_eval_split(args.test_data_path)
    cb_setting = args.cb_setting or infer_cb_setting(
        [str(args.sft_root), str(args.test_data_path), str(args.index_path), model_name, dataset]
    )

    model_config_payload = {
        "model_name": model_name,
        "dataset": dataset,
        "cb_setting": cb_setting,
        "seed": args.seed,
        "num_beams": args.num_beams,
        "sid_levels": args.sid_levels,
        "eval_split": eval_split,
        "sft_root": str(args.sft_root),
        "test_data_path": str(args.test_data_path),
        "index_path": str(args.index_path),
    }
    model_config_key = stable_hash(model_config_payload, length=20)

    default_run_id = f"eval-{model_config_key}"
    default_run_name = f"{sanitize_token(model_name)}-eval"

    state_prefix = sanitize_token(model_name)
    state_file = args.state_dir / f"{state_prefix}_{model_config_key}.json"
    lock_file = args.state_dir / f"{state_prefix}_{model_config_key}.lock"

    logging.info("repo_root=%s", args.repo_root)
    logging.info("sft_root=%s", args.sft_root)
    logging.info("state_file=%s", state_file)
    logging.info("wandb_project=%s", args.wandb_project)

    with exclusive_lock(lock_file):
        raw_state: dict[str, Any] = {}
        if state_file.is_file():
            raw_state = load_json(state_file)

        run_id = args.wandb_run_id or str(raw_state.get("run_id") or default_run_id)
        run_name = args.wandb_run_name or str(raw_state.get("run_name") or default_run_name)

        state = normalize_state(raw_state, model_config_key=model_config_key, run_id=run_id, run_name=run_name)
        state["last_update_time"] = now_utc_iso()
        save_json_atomic(state_file, state)

        wandb_module = None
        wandb_run = None
        if not args.disable_wandb:
            wandb_config = {
                "model_name": model_name,
                "dataset": dataset,
                "cb_setting": cb_setting,
                "seed": args.seed,
                "num_beams": args.num_beams,
                "sid_levels": args.sid_levels,
                "eval_split": eval_split,
            }
            wandb_module, wandb_run = init_wandb(args, run_id=run_id, run_name=run_name, config=wandb_config)

        try:
            while True:
                checkpoints = discover_checkpoints(args.sft_root)
                if checkpoints:
                    state["last_seen_step"] = checkpoints[-1].step
                else:
                    state["last_seen_step"] = None

                processed = {int(step) for step in state.get("processed_steps", [])}
                pending: list[CheckpointInfo] = []
                for ckpt in checkpoints:
                    if ckpt.step in processed:
                        continue
                    if not checkpoint_is_ready(ckpt, args.checkpoint_ready_seconds):
                        continue
                    pending.append(ckpt)

                if args.max_pending_per_cycle > 0:
                    pending = pending[: args.max_pending_per_cycle]

                if not pending:
                    state["last_update_time"] = now_utc_iso()
                    save_json_atomic(state_file, state)
                    if args.once:
                        logging.info("No pending checkpoints. Exit due to --once.")
                        break
                    time.sleep(args.poll_interval_seconds)
                    continue

                logging.info(
                    "Found %d pending checkpoint(s): %s", len(pending), ", ".join(item.name for item in pending)
                )

                for ckpt in pending:
                    metrics_path = metrics_json_path(args.repo_root, ckpt)
                    try:
                        if args.force_eval or not metrics_path.is_file():
                            run_single_checkpoint_eval(args, ckpt)

                        if not metrics_path.is_file():
                            raise FileNotFoundError(f"metrics.json not found after eval: {metrics_path}")

                        metrics = load_metrics(metrics_path)

                        state["processed_steps"] = sorted({*processed, ckpt.step})
                        processed.add(ckpt.step)
                        state_failed = state.get("failed_steps", {})
                        state_failed.pop(str(ckpt.step), None)
                        state["failed_steps"] = state_failed

                        row = build_row(ckpt, metrics, metrics_path)
                        state["table_rows"] = upsert_table_row(state.get("table_rows", []), row)
                        state["last_update_time"] = now_utc_iso()

                        if wandb_run is not None and wandb_module is not None:
                            log_metrics_to_wandb(
                                wandb_module,
                                wandb_run,
                                ckpt=ckpt,
                                metrics=metrics,
                                metrics_path=metrics_path,
                                table_rows=state["table_rows"],
                            )

                        save_json_atomic(state_file, state)
                        logging.info("Processed %s successfully", ckpt.name)
                    except Exception as exc:  # pylint: disable=broad-except
                        failed_steps = state.get("failed_steps", {})
                        failed_steps[str(ckpt.step)] = {
                            "checkpoint": ckpt.name,
                            "error": str(exc),
                            "time": now_utc_iso(),
                        }
                        state["failed_steps"] = failed_steps
                        state["last_update_time"] = now_utc_iso()
                        save_json_atomic(state_file, state)
                        logging.exception("Failed to process %s", ckpt.name)

                if args.once:
                    logging.info("Completed one pass due to --once.")
                    break

                time.sleep(args.poll_interval_seconds)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
