from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

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


def test_upload_persists_checkpoint_index_and_epoch_progress(tmp_path: Path, monkeypatch) -> None:
    model_dir = "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495"
    state_dir = tmp_path / "state" / "wandb_eval_uploader"

    for checkpoint_step in (266, 532, 798):
        _write_json(
            tmp_path / "results" / model_dir / f"checkpoint-{checkpoint_step}" / "metrics.json",
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
            "generated_at": "2026-04-01T00:00:00+00:00",
            "results_root": "./results",
            "models": [
                {
                    "model_dir": model_dir,
                    "dataset": "Instruments_grec",
                    "cb_setting": "none",
                    "eval_split": "test",
                    "seed": 42,
                    "num_beams": 50,
                    "sid_levels": -1,
                    "num_train_epochs": 2,
                    "wandb_project": "MIMIGenRec-Eval",
                    "wandb_entity": None,
                    "wandb_run_id": "eval-244685544728aa7e29a6",
                    "wandb_run_name": f"{model_dir}-eval",
                }
            ],
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

    state_suffix = sidecar.stable_hash({"model_dir": model_dir, "run_id": "eval-244685544728aa7e29a6"}, 12)
    state_path = state_dir / f"{model_dir}_{state_suffix}.json"
    state = sidecar.load_json(state_path)

    assert state["processed_steps"] == [266, 532, 798]
    assert [row["checkpoint_index"] for row in state["table_rows"]] == [1, 2, 3]
    assert [row["epoch_progress"] for row in state["table_rows"]] == [
        pytest.approx(266 / 798 * 2, rel=1e-6),
        pytest.approx(532 / 798 * 2, rel=1e-6),
        pytest.approx(2.0, rel=1e-6),
    ]


def test_init_wandb_run_passes_group_from_model_spec(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_wandb = types.SimpleNamespace()

    def fake_init(**kwargs):
        captured.update(kwargs)
        return object()

    fake_wandb.init = fake_init
    fake_wandb.define_metric = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    args = types.SimpleNamespace(
        wandb_resume="allow",
        wandb_job_type="eval",
        wandb_mode="offline",
    )
    model = types.SimpleNamespace(
        model_dir="Instruments-grec-grpo-prefix-qwen2.5-3b-qwen4B-4-256-from-sft495",
        variant="epoch",
        dataset="Instruments_grec",
        cb_setting="cb256",
        seed=42,
        num_beams=50,
        sid_levels=-1,
        num_train_epochs=2,
        eval_split="test",
        wandb_project="MIMIGenRec-Eval",
        wandb_entity=None,
        wandb_run_id="eval-a8fc2bd06d188292bd17-epoch-20260401",
        wandb_run_name="instruments-prefix-eval-epoch-20260401",
        wandb_group="epoch",
    )

    sidecar.init_wandb_run(args, model)

    assert captured["group"] == "epoch"


def test_parse_models_from_manifest_expands_ckpt_step_and_epoch_variants() -> None:
    manifest = {
        "version": sidecar.MANIFEST_VERSION,
        "generated_at": "2026-04-02T00:00:00+00:00",
        "results_root": "./results",
        "models": [
            {
                "model_dir": "Instruments-grec-grpo-prefix-qwen2.5-3b-qwen4B-4-256-from-sft495",
                "dataset": "Instruments_grec",
                "cb_setting": "cb256",
                "eval_split": "test",
                "seed": 42,
                "num_beams": 50,
                "sid_levels": -1,
                "num_train_epochs": 2,
                "wandb_project": "MIMIGenRec-Eval",
                "wandb_entity": None,
                "wandb_run_id": "eval-a8fc2bd06d188292bd17",
                "wandb_run_name": "instruments-prefix-eval",
            }
        ],
    }

    specs = sidecar.parse_models_from_manifest(
        manifest,
        model_filter=set(),
        upload_variants=sidecar.parse_upload_variants("ckpt_step,epoch"),
    )

    assert [(spec.variant, spec.wandb_group) for spec in specs] == [
        ("ckpt_step", "ckpt_step"),
        ("epoch", "epoch"),
    ]
    assert specs[0].wandb_run_id == "eval-a8fc2bd06d188292bd17"
    assert specs[1].wandb_run_id == "eval-a8fc2bd06d188292bd17-epoch"
    assert specs[1].wandb_run_name == "instruments-prefix-eval-epoch"


def test_init_wandb_run_surfaces_deleted_run_id_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_wandb = types.SimpleNamespace()

    def fake_init(**kwargs):
        raise RuntimeError(
            "Run initialization has timed out after 90.0 sec. Please try increasing the timeout."
        )

    fake_wandb.init = fake_init
    fake_wandb.define_metric = lambda *args, **kwargs: None

    log_path = (
        tmp_path
        / "wandb"
        / "run-20260411_130940-eval-7496742c37888e35b425"
        / "logs"
        / "debug-internal.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                '{"time":"2026-04-11T13:09:42.71467+08:00","level":"INFO","msg":"api: retrying HTTP error","status":409}',
                (
                    '{"errors":[{"message":"run eval-7496742c37888e35b425 was previously '
                    'created and deleted; try a new run id"}]}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    args = types.SimpleNamespace(
        wandb_resume="allow",
        wandb_job_type="eval",
        wandb_mode="online",
    )
    model = types.SimpleNamespace(
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        variant="ckpt_step",
        dataset="Instruments_grec",
        cb_setting="none",
        seed=42,
        num_beams=50,
        sid_levels=-1,
        num_train_epochs=2,
        eval_split="test",
        wandb_project="MIMIGenRec-Eval",
        wandb_entity=None,
        wandb_run_id="eval-7496742c37888e35b425",
        wandb_run_name="instruments-fixedhint-eval",
        wandb_group="ckpt_step",
    )

    with pytest.raises(RuntimeError, match="previously created and deleted") as exc_info:
        sidecar.init_wandb_run(args, model)

    assert "config/wandb_eval_manifest_overrides.json" in str(exc_info.value)
    assert "variants.ckpt_step.wandb_run_id" in str(exc_info.value)


def test_upload_defaults_to_dual_variants_with_legacy_epoch_override(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495"
    ckpt_run_id = "eval-7496742c37888e35b425"
    epoch_run_id = f"{ckpt_run_id}-epoch-20260402"
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
            "generated_at": "2026-04-02T00:00:00+00:00",
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
                    "num_train_epochs": 2,
                    "wandb_project": "MIMIGenRec-Eval",
                    "wandb_entity": None,
                    "wandb_run_id": ckpt_run_id,
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
                    "num_train_epochs": 2,
                    "wandb_group": "epoch",
                    "wandb_run_id": epoch_run_id,
                    "wandb_run_name": f"{model_dir}-eval-epoch-20260402",
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

    ckpt_suffix = sidecar.stable_hash({"model_dir": model_dir, "run_id": ckpt_run_id}, 12)
    epoch_suffix = sidecar.stable_hash({"model_dir": model_dir, "run_id": epoch_run_id}, 12)
    ckpt_state = sidecar.load_json(state_dir / f"{model_dir}_{ckpt_suffix}.json")
    epoch_state = sidecar.load_json(state_dir / f"{model_dir}_{epoch_suffix}.json")

    assert ckpt_state["run_id"] == ckpt_run_id
    assert epoch_state["run_id"] == epoch_run_id


def test_repo_overrides_rebind_deleted_ckpt_step_run_id() -> None:
    model_dir = "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495"
    manifest = {
        "version": sidecar.MANIFEST_VERSION,
        "generated_at": "2026-04-11T00:00:00+00:00",
        "results_root": "./results",
        "models": [
            {
                "model_dir": model_dir,
                "dataset": "Instruments_grec",
                "cb_setting": "none",
                "eval_split": "test",
                "seed": 42,
                "num_beams": 50,
                "sid_levels": -1,
                "num_train_epochs": None,
                "wandb_project": "MIMIGenRec-Eval",
                "wandb_entity": None,
                "wandb_run_id": "eval-7496742c37888e35b425",
                "wandb_run_name": f"{model_dir}-eval",
            }
        ],
    }

    overrides = sidecar.load_overrides(REPO_ROOT / "config" / "wandb_eval_manifest_overrides.json")
    merged_manifest = sidecar.apply_manifest_overrides(manifest, overrides)
    specs = sidecar.parse_models_from_manifest(
        merged_manifest,
        model_filter=set(),
        upload_variants=sidecar.parse_upload_variants("ckpt_step,epoch"),
    )

    specs_by_variant = {spec.variant: spec for spec in specs}
    assert specs_by_variant["ckpt_step"].wandb_run_id == "eval-7496742c37888e35b425-rerun-20260411"
    assert specs_by_variant["epoch"].wandb_run_id == "eval-7496742c37888e35b425-epoch-20260401"
