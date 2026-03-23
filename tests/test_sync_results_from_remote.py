from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_results_from_remote.sh"


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_results_archive(archive_path: Path, *, model_dir: str, checkpoint: str, hr10: float) -> None:
    staging_root = archive_path.parent / f"{archive_path.stem}_staging"
    metrics_path = staging_root / "GenRec" / "results" / model_dir / checkpoint / "metrics.json"
    manifest_path = staging_root / "GenRec" / "results" / ".wandb_eval_manifest.json"

    _write_file(metrics_path, '{"HR@10": %.4f, "NDCG@10": %.4f}\n' % (hr10, hr10 / 2))
    _write_file(
        manifest_path,
        (
            '{"version": 1, "generated_at": "2026-03-23T00:00:00+00:00", "results_root": "./results", '
            '"models": [{"model_dir": "%s", "dataset": "Instruments", "cb_setting": "none", '
            '"eval_split": "test", "seed": 42, "num_beams": 50, "sid_levels": -1, '
            '"wandb_project": "MIMIGenRec-Eval", "wandb_entity": null, '
            '"wandb_run_id": "eval-test", "wandb_run_name": "%s-eval"}]}\n'
        )
        % (model_dir, model_dir),
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(staging_root / "GenRec", arcname="GenRec")
    shutil.rmtree(staging_root)


def _build_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    home_dir = tmp_path / "home"
    downloads_dir = home_dir / "Downloads"
    local_repo_dir = tmp_path / "local_genrec"
    local_results_dir = local_repo_dir / "results"

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "LOCAL_GENREC_REPO_DIR": str(local_repo_dir),
            "LOCAL_GENREC_RESULTS_DIR": str(local_results_dir),
        }
    )
    return env, downloads_dir


def test_unpack_without_input_prefers_downloads_and_deletes_consumed_archive(tmp_path: Path):
    env, downloads_dir = _build_env(tmp_path)
    archive_path = downloads_dir / "results_sync_latest.tar.gz"
    _make_results_archive(
        archive_path,
        model_dir="Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
        checkpoint="checkpoint-3326",
        hr10=0.1186,
    )

    cwd_archive = REPO_ROOT / "results_sync_latest.tar.gz"
    if cwd_archive.exists():
        cwd_archive.unlink()

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "unpack"],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Using latest archive: {archive_path}" in result.stdout
    assert not archive_path.exists()
    assert (
        tmp_path
        / "local_genrec/results/Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495/checkpoint-3326/metrics.json"
    ).exists()


def test_unpack_only_deletes_archive_it_consumed(tmp_path: Path):
    env, downloads_dir = _build_env(tmp_path)
    old_archive = downloads_dir / "results_sync_old.tar.gz"
    new_archive = downloads_dir / "results_sync_new.tar.gz"
    _make_results_archive(
        old_archive,
        model_dir="Instruments-old-model",
        checkpoint="checkpoint-111",
        hr10=0.0500,
    )
    _make_results_archive(
        new_archive,
        model_dir="Instruments-new-model",
        checkpoint="checkpoint-222",
        hr10=0.0900,
    )

    os.utime(old_archive, (old_archive.stat().st_atime, old_archive.stat().st_mtime - 60))

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "unpack"],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not new_archive.exists()
    assert old_archive.exists()
    assert (tmp_path / "local_genrec/results/Instruments-new-model/checkpoint-222/metrics.json").exists()


def test_unpack_without_input_prefers_downloads_over_current_dir_archive(tmp_path: Path):
    env, downloads_dir = _build_env(tmp_path)
    downloads_archive = downloads_dir / "results_sync_downloads.tar.gz"
    cwd_archive = REPO_ROOT / "results_sync_current_test.tar.gz"

    _make_results_archive(
        downloads_archive,
        model_dir="Instruments-downloads-model",
        checkpoint="checkpoint-333",
        hr10=0.1100,
    )
    _make_results_archive(
        cwd_archive,
        model_dir="Instruments-current-model",
        checkpoint="checkpoint-444",
        hr10=0.2200,
    )

    try:
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "unpack"],
            check=False,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert f"Using latest archive: {downloads_archive}" in result.stdout
        assert not downloads_archive.exists()
        assert cwd_archive.exists()
        assert (tmp_path / "local_genrec/results/Instruments-downloads-model/checkpoint-333/metrics.json").exists()
    finally:
        if cwd_archive.exists():
            cwd_archive.unlink()
