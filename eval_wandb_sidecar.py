#!/usr/bin/env python3
"""Generate/upload eval metrics manifests for W&B.

This script supports two commands:

1) prepare-manifest (remote):
   Scan results/<model_dir>/checkpoint-*/metrics.json and generate a manifest.

2) upload (local):
   Read manifest + results and incrementally upload metrics to W&B.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
CB_RE = re.compile(r"(cb\d+(?:-\d+)*)", re.IGNORECASE)
WIDTH_CB_RE = re.compile(r"(?:^|-)4-(\d+)(?:-|$)")

MANIFEST_VERSION = 1

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


@dataclass(frozen=True)
class ModelSpec:
    model_dir: str
    dataset: str
    cb_setting: str
    eval_split: str
    seed: int
    num_beams: int
    sid_levels: int
    wandb_project: str
    wandb_entity: str | None
    wandb_run_id: str
    wandb_run_name: str


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


def stable_hash(payload: dict[str, Any], length: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def parse_checkpoint_step(name: str) -> int | None:
    match = CHECKPOINT_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def discover_checkpoint_dirs(model_dir_path: Path) -> list[CheckpointInfo]:
    checkpoints: list[CheckpointInfo] = []
    if not model_dir_path.is_dir():
        return checkpoints

    for entry in model_dir_path.iterdir():
        if not entry.is_dir():
            continue
        step = parse_checkpoint_step(entry.name)
        if step is None:
            continue
        checkpoints.append(CheckpointInfo(step=step, name=entry.name, path=entry))

    checkpoints.sort(key=lambda item: item.step)
    return checkpoints


def discover_model_dirs(results_root: Path) -> list[str]:
    models: list[str] = []
    if not results_root.is_dir():
        return models

    for entry in results_root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        ckpts = discover_checkpoint_dirs(entry)
        if ckpts:
            models.append(entry.name)

    models.sort()
    return models


def metrics_json_path(ckpt: CheckpointInfo) -> Path:
    return ckpt.path / "metrics.json"


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


@contextmanager
def exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as fp:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another uploader process holds lock: {lock_path}") from exc

        fp.seek(0)
        fp.truncate(0)
        fp.write(str(os.getpid()))
        fp.flush()

        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def infer_dataset_from_model_dir(model_dir: str) -> str:
    split_tokens = ["-sft-", "-qwen", "-llama", "-mistral", "-grpo", "-rl"]
    prefix = model_dir
    for token in split_tokens:
        if token in model_dir:
            prefix = model_dir.split(token, 1)[0]
            break

    if prefix:
        return prefix.replace("-", "_")
    return model_dir.replace("-", "_")


def infer_cb_setting_from_model_dir(model_dir: str) -> str:
    cb_match = CB_RE.search(model_dir)
    if cb_match:
        return cb_match.group(1)

    width_match = WIDTH_CB_RE.search(model_dir)
    if width_match:
        return f"cb{width_match.group(1)}"

    return "none"


def load_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}

    raw = load_json(path)

    if "models" in raw:
        models_obj = raw["models"]
        if isinstance(models_obj, dict):
            return {str(k): dict(v) for k, v in models_obj.items() if isinstance(v, dict)}
        if isinstance(models_obj, list):
            parsed: dict[str, dict[str, Any]] = {}
            for item in models_obj:
                if not isinstance(item, dict):
                    continue
                model_dir = item.get("model_dir")
                if isinstance(model_dir, str) and model_dir:
                    payload = {k: v for k, v in item.items() if k != "model_dir"}
                    parsed[model_dir] = payload
            return parsed
        raise ValueError(f"Unsupported 'models' format in overrides: {path}")

    parsed: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            parsed[str(k)] = dict(v)
    return parsed


def build_model_manifest_item(
    *,
    model_dir: str,
    defaults: dict[str, Any],
    override: dict[str, Any],
    run_id_prefix: str,
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(override)

    dataset = str(merged.get("dataset") or defaults["dataset"])
    cb_setting = str(merged.get("cb_setting") or defaults["cb_setting"])
    eval_split = str(merged.get("eval_split") or defaults["eval_split"])
    seed = int(merged.get("seed", defaults["seed"]))
    num_beams = int(merged.get("num_beams", defaults["num_beams"]))
    sid_levels = int(merged.get("sid_levels", defaults["sid_levels"]))
    wandb_project = str(merged.get("wandb_project") or defaults["wandb_project"])
    wandb_entity = merged.get("wandb_entity", defaults["wandb_entity"])
    wandb_entity = str(wandb_entity) if wandb_entity not in (None, "") else None

    default_run_id = f"{run_id_prefix}-{stable_hash({'model_dir': model_dir, 'dataset': dataset, 'cb_setting': cb_setting, 'eval_split': eval_split}, 20)}"
    wandb_run_id = str(merged.get("wandb_run_id") or default_run_id)
    wandb_run_name = str(merged.get("wandb_run_name") or f"{sanitize_token(model_dir)}-eval")

    return {
        "model_dir": model_dir,
        "dataset": dataset,
        "cb_setting": cb_setting,
        "eval_split": eval_split,
        "seed": seed,
        "num_beams": num_beams,
        "sid_levels": sid_levels,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
    }


def cmd_prepare_manifest(args: argparse.Namespace) -> int:
    overrides = load_overrides(args.overrides)

    model_dirs = discover_model_dirs(args.results_root)
    if not model_dirs:
        logging.warning("No model directories found under results root: %s", args.results_root)

    models: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        defaults = {
            "dataset": infer_dataset_from_model_dir(model_dir),
            "cb_setting": infer_cb_setting_from_model_dir(model_dir),
            "eval_split": args.default_eval_split,
            "seed": 42,
            "num_beams": 50,
            "sid_levels": -1,
            "wandb_project": args.default_project,
            "wandb_entity": args.default_entity,
        }
        override = overrides.get(model_dir, {})
        item = build_model_manifest_item(
            model_dir=model_dir,
            defaults=defaults,
            override=override,
            run_id_prefix=args.run_id_prefix,
        )
        models.append(item)

    manifest = {
        "version": MANIFEST_VERSION,
        "generated_at": now_utc_iso(),
        "results_root": str(args.results_root),
        "models": sorted(models, key=lambda x: x["model_dir"]),
    }

    save_json_atomic(args.output_manifest, manifest)

    logging.info("Manifest written: %s", args.output_manifest)
    logging.info("Models included: %d", len(models))
    for item in models:
        logging.info(
            "model=%s dataset=%s cb_setting=%s run_id=%s",
            item["model_dir"],
            item["dataset"],
            item["cb_setting"],
            item["wandb_run_id"],
        )

    return 0


def parse_model_spec(raw: dict[str, Any], idx: int) -> ModelSpec:
    required = [
        "model_dir",
        "dataset",
        "cb_setting",
        "eval_split",
        "seed",
        "num_beams",
        "sid_levels",
        "wandb_project",
        "wandb_run_id",
        "wandb_run_name",
    ]

    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Manifest models[{idx}] missing required keys: {missing}")

    entity_raw = raw.get("wandb_entity")
    wandb_entity = str(entity_raw) if entity_raw not in (None, "") else None

    return ModelSpec(
        model_dir=str(raw["model_dir"]),
        dataset=str(raw["dataset"]),
        cb_setting=str(raw["cb_setting"]),
        eval_split=str(raw["eval_split"]),
        seed=int(raw["seed"]),
        num_beams=int(raw["num_beams"]),
        sid_levels=int(raw["sid_levels"]),
        wandb_project=str(raw["wandb_project"]),
        wandb_entity=wandb_entity,
        wandb_run_id=str(raw["wandb_run_id"]),
        wandb_run_name=str(raw["wandb_run_name"]),
    )


def load_manifest(path: Path) -> dict[str, Any]:
    raw = load_json(path)
    version = raw.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version={version}, expected {MANIFEST_VERSION}")

    models = raw.get("models")
    if not isinstance(models, list):
        raise ValueError("Manifest 'models' must be a list")

    return raw


def parse_models_from_manifest(manifest: dict[str, Any], model_filter: set[str]) -> list[ModelSpec]:
    seen: set[str] = set()
    specs: list[ModelSpec] = []

    for idx, model_raw in enumerate(manifest["models"]):
        if not isinstance(model_raw, dict):
            raise ValueError(f"Manifest models[{idx}] must be an object")
        spec = parse_model_spec(model_raw, idx)
        if model_filter and spec.model_dir not in model_filter:
            continue
        if spec.model_dir in seen:
            raise ValueError(f"Duplicate model_dir in manifest: {spec.model_dir}")
        seen.add(spec.model_dir)
        specs.append(spec)

    specs.sort(key=lambda m: m.model_dir)
    return specs


def normalize_upload_state(raw: dict[str, Any], model: ModelSpec) -> dict[str, Any]:
    processed_steps = sorted({int(step) for step in raw.get("processed_steps", [])})
    table_rows = raw.get("table_rows", [])
    if not isinstance(table_rows, list):
        table_rows = []

    failed_steps = raw.get("failed_steps", {})
    if not isinstance(failed_steps, dict):
        failed_steps = {}

    return {
        "version": 1,
        "model_dir": model.model_dir,
        "run_id": model.wandb_run_id,
        "run_name": model.wandb_run_name,
        "processed_steps": processed_steps,
        "failed_steps": failed_steps,
        "last_seen_step": raw.get("last_seen_step"),
        "last_update_time": raw.get("last_update_time", now_utc_iso()),
        "table_rows": table_rows,
    }


def state_file_for_model(state_dir: Path, model: ModelSpec) -> Path:
    suffix = stable_hash({"model_dir": model.model_dir, "run_id": model.wandb_run_id}, 12)
    base = sanitize_token(model.model_dir)
    return state_dir / f"{base}_{suffix}.json"


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


def build_row(ckpt: CheckpointInfo, metrics: dict[str, float], metrics_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_step": ckpt.step,
        "checkpoint_name": ckpt.name,
        "metrics_path": str(metrics_path),
    }
    for key in METRIC_KEYS:
        row[key] = float(metrics[key]) if key in metrics else None
    return row


def init_wandb_run(args: argparse.Namespace, model: ModelSpec):
    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in current environment") from exc

    run = wandb.init(
        project=model.wandb_project,
        entity=model.wandb_entity,
        id=model.wandb_run_id,
        name=model.wandb_run_name,
        resume=args.wandb_resume,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        config={
            "model_name": model.model_dir,
            "dataset": model.dataset,
            "cb_setting": model.cb_setting,
            "seed": model.seed,
            "num_beams": model.num_beams,
            "sid_levels": model.sid_levels,
            "eval_split": model.eval_split,
        },
        reinit=True,
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

    run.log(payload, step=ckpt.step)

    table = wandb_module.Table(columns=TABLE_COLUMNS)
    for row in sorted(table_rows, key=lambda item: int(item["checkpoint_step"])):
        table.add_data(*[row.get(column) for column in TABLE_COLUMNS])
    run.log({"eval/checkpoint_table": table}, step=ckpt.step)

    summary = build_summary_from_rows(table_rows)
    for key, value in summary.items():
        run.summary[key] = value


def discover_pending_checkpoints(
    results_root: Path, model: ModelSpec, processed: set[int]
) -> tuple[list[CheckpointInfo], list[CheckpointInfo]]:
    model_path = results_root / model.model_dir
    checkpoints = discover_checkpoint_dirs(model_path)

    pending: list[CheckpointInfo] = []
    for ckpt in checkpoints:
        if ckpt.step in processed:
            continue
        if not metrics_json_path(ckpt).is_file():
            continue
        pending.append(ckpt)

    return checkpoints, pending


def process_model_once(args: argparse.Namespace, model: ModelSpec) -> tuple[int, int]:
    state_file = state_file_for_model(args.state_dir, model)
    raw_state: dict[str, Any] = {}
    if state_file.is_file():
        raw_state = load_json(state_file)
    state = normalize_upload_state(raw_state, model)

    processed = {int(step) for step in state["processed_steps"]}

    checkpoints, pending = discover_pending_checkpoints(args.results_root, model, processed)
    state["last_seen_step"] = checkpoints[-1].step if checkpoints else None

    if not pending:
        state["last_update_time"] = now_utc_iso()
        save_json_atomic(state_file, state)
        return (0, 0)

    logging.info(
        "Model %s pending checkpoints: %s",
        model.model_dir,
        ", ".join(item.name for item in pending),
    )

    wandb_module = None
    wandb_run = None
    if not args.disable_wandb:
        wandb_module, wandb_run = init_wandb_run(args, model)

    processed_count = 0
    try:
        for ckpt in pending:
            metrics_path = metrics_json_path(ckpt)
            try:
                metrics = load_metrics(metrics_path)

                state["processed_steps"] = sorted({*processed, ckpt.step})
                processed.add(ckpt.step)

                failed_steps = state.get("failed_steps", {})
                failed_steps.pop(str(ckpt.step), None)
                state["failed_steps"] = failed_steps

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
                processed_count += 1
                logging.info("Processed %s/%s", model.model_dir, ckpt.name)
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
                logging.exception("Failed processing %s/%s", model.model_dir, ckpt.name)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return (processed_count, len(pending))


def parse_model_filter(raw_values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for raw in raw_values:
        for item in raw.split(","):
            value = item.strip()
            if value:
                parsed.add(value)
    return parsed


def cmd_upload(args: argparse.Namespace) -> int:
    model_filter = parse_model_filter(args.model_filter)
    lock_file = args.state_dir / ".upload.lock"

    with exclusive_lock(lock_file):
        while True:
            manifest = load_manifest(args.manifest_path)
            models = parse_models_from_manifest(manifest, model_filter)

            if not models:
                logging.info("No models to process (manifest + model filter).")

            total_processed = 0
            total_pending = 0

            for model in models:
                processed_count, pending_count = process_model_once(args, model)
                total_processed += processed_count
                total_pending += pending_count

            if args.once:
                if total_pending == 0:
                    logging.info("No pending checkpoints. Exit due to --once.")
                else:
                    logging.info(
                        "Completed one pass due to --once. processed=%d pending_seen=%d",
                        total_processed,
                        total_pending,
                    )
                break

            time.sleep(args.poll_interval_seconds)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest-based eval uploader for W&B")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-manifest", help="Generate manifest from results directory")
    prepare.add_argument("--results-root", default="./results", help="Results root containing model directories")
    prepare.add_argument(
        "--output-manifest",
        default="results/.wandb_eval_manifest.json",
        help="Output manifest path",
    )
    prepare.add_argument("--overrides", default="", help="Optional overrides JSON path")
    prepare.add_argument("--default-project", default="MIMIGenRec-Eval", help="Default W&B project")
    prepare.add_argument("--default-entity", default="", help="Default W&B entity")
    prepare.add_argument("--default-eval-split", default="test", help="Default eval split")
    prepare.add_argument("--run-id-prefix", default="eval", help="Run ID prefix for generated run IDs")
    prepare.add_argument("--log-level", default="INFO")

    upload = subparsers.add_parser("upload", help="Upload metrics based on manifest")
    upload.add_argument("--results-root", default="./results", help="Results root containing model directories")
    upload.add_argument(
        "--manifest-path",
        default="results/.wandb_eval_manifest.json",
        help="Manifest generated by prepare-manifest",
    )
    upload.add_argument("--state-dir", default="state/wandb_eval_uploader", help="Uploader state directory")
    upload.add_argument("--once", action="store_true", help="Process pending checkpoints once then exit")
    upload.add_argument("--poll-interval-seconds", type=int, default=60)
    upload.add_argument("--disable-wandb", action="store_true")
    upload.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    upload.add_argument("--wandb-resume", default=os.environ.get("WANDB_RESUME", "allow"))
    upload.add_argument("--wandb-job-type", default=os.environ.get("WANDB_JOB_TYPE", "eval"))
    upload.add_argument(
        "--model-filter",
        action="append",
        default=[],
        help="Optional model_dir filter (repeatable or comma-separated)",
    )
    upload.add_argument("--log-level", default="INFO")

    return parser


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    cwd = Path.cwd()

    if args.command == "prepare-manifest":
        args.results_root = resolve_path(args.results_root, cwd)
        args.output_manifest = resolve_path(args.output_manifest, cwd)
        args.overrides = resolve_path(args.overrides, cwd) if args.overrides else None

        if not args.results_root.is_dir():
            raise FileNotFoundError(f"results root not found: {args.results_root}")
        if args.overrides is not None and not args.overrides.is_file():
            raise FileNotFoundError(f"overrides file not found: {args.overrides}")

        return cmd_prepare_manifest(args)

    if args.command == "upload":
        args.results_root = resolve_path(args.results_root, cwd)
        args.manifest_path = resolve_path(args.manifest_path, cwd)
        args.state_dir = resolve_path(args.state_dir, cwd)

        if not args.results_root.is_dir():
            raise FileNotFoundError(f"results root not found: {args.results_root}")
        if not args.manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {args.manifest_path}")

        return cmd_upload(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
