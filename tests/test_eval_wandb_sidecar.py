from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_wandb_sidecar as sidecar


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_upload_applies_local_overrides_to_stale_manifest(tmp_path: Path, monkeypatch) -> None:
    model_dir = "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495"
    old_run_id = "eval-7496742c37888e35b425"
    new_run_id = f"{old_run_id}-rerun-20260323"
    state_dir = tmp_path / "state" / "wandb_eval_uploader"

    _write_json(
        tmp_path / "results" / model_dir / "checkpoint-333" / "metrics.json",
        {
            "HR@1": 0.1,
            "HR@3": 0.2,
            "HR@5": 0.3,
            "HR@10": 0.4,
            "NDCG@1": 0.1,
            "NDCG@3": 0.2,
            "NDCG@5": 0.3,
            "NDCG@10": 0.4,
        },
    )
    _write_json(
        tmp_path / "results" / ".wandb_eval_manifest.json",
        {
            "version": sidecar.MANIFEST_VERSION,
            "generated_at": "2026-03-28T00:00:00+00:00",
            "results_root": "./results",
            "models": [
                {
                    "model_dir": model_dir,
                    "dataset": "Instruments",
                    "cb_setting": "none",
                    "eval_split": "test",
                    "seed": 42,
                    "num_beams": 50,
                    "sid_levels": -1,
                    "wandb_project": "MIMIGenRec-Eval",
                    "wandb_entity": None,
                    "wandb_run_id": old_run_id,
                    "wandb_run_name": f"{model_dir}-eval",
                }
            ],
        },
    )
    _write_json(
        tmp_path / "config" / "wandb_eval_manifest_overrides.json",
        {
            "models": {
                model_dir: {
                    "wandb_run_id": new_run_id,
                    "wandb_run_name": f"{model_dir}-eval-rerun-20260323",
                }
            }
        },
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_wandb_sidecar.py",
            "upload",
            "--results-root",
            "./results",
            "--manifest-path",
            "./results/.wandb_eval_manifest.json",
            "--state-dir",
            str(state_dir),
            "--once",
            "--disable-wandb",
        ],
    )

    assert sidecar.main() == 0

    old_suffix = sidecar.stable_hash({"model_dir": model_dir, "run_id": old_run_id}, 12)
    new_suffix = sidecar.stable_hash({"model_dir": model_dir, "run_id": new_run_id}, 12)
    old_state = state_dir / f"{model_dir}_{old_suffix}.json"
    new_state = state_dir / f"{model_dir}_{new_suffix}.json"

    assert not old_state.exists()
    assert new_state.exists()
    assert sidecar.load_json(new_state)["run_id"] == new_run_id
