from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_hint_research_bundle.sh"


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_fake_remote_layout(tmp_path: Path) -> dict[str, Path | str]:
    remote_root = tmp_path / "remote_home"
    repo_root = remote_root / "GenRec"
    raw_data_root = remote_root / "data"
    category = "Instruments"
    data_variant = "Instruments_grec_fake_variant"

    _write_file(
        repo_root
        / "output/jupyter-notebook/genrec-hint-cascade-artifacts/instruments_grec_beam16_hint_difficulty_table.csv",
        "sample_id,task_label\n0,sid\n",
    )
    _write_file(
        repo_root / "temp/rl_beam_hint/instruments_grec_beam_hint_cascade_20260314_summary.json",
        '{"beam_sizes":[16],"results":{}}',
    )
    _write_file(
        repo_root / "temp/rl_beam_hint/instruments_grec_beam_hint_cascade_20260314_details.json",
        '{"results":{"16":{"stages":{}}}}',
    )
    _write_file(
        repo_root / "data" / data_variant / "rl" / "train.json",
        '[{"extra_info":{"task":"task1_sid_sft"}}]',
    )
    _write_file(
        repo_root / "data" / data_variant / "id2sid.json",
        '{"1":["<a_1>","<b_2>"]}',
    )
    _write_file(
        repo_root / "data" / data_variant / "new_tokens.json",
        '["<a_1>","<b_2>"]',
    )
    _write_file(
        repo_root / "scripts/hint_research/genrec-hint-cascade-analysis-2.ipynb",
        '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}',
    )
    _write_file(raw_data_root / category / f"{category}.inter.json", '{"u1":["1","2","3"]}')
    _write_file(raw_data_root / category / f"{category}.item.json", '{"1":{"title":"A"}}')

    return {
        "repo_root": repo_root,
        "raw_data_root": raw_data_root,
        "category": category,
        "data_variant": data_variant,
    }


def _build_env(layout: dict[str, Path | str], bundle_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BUNDLE_REPO_ROOT": str(layout["repo_root"]),
            "BUNDLE_RAW_DATA_ROOT": str(layout["raw_data_root"]),
            "CATEGORY": str(layout["category"]),
            "DATA_VARIANT": str(layout["data_variant"]),
            "BUNDLE_NAME": bundle_name,
        }
    )
    return env


def test_pack_creates_research_bundle_archive_with_expected_files(tmp_path: Path):
    layout = _build_fake_remote_layout(tmp_path)
    env = _build_env(layout, bundle_name="test_bundle")
    archive_path = tmp_path / "research_bundle.tar.gz"

    subprocess.run(
        ["bash", str(SCRIPT_PATH), "pack", str(archive_path)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert archive_path.exists()
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())

    assert "test_bundle/bundle_manifest.json" in names
    assert (
        "test_bundle/GenRec/output/jupyter-notebook/genrec-hint-cascade-artifacts/"
        "instruments_grec_beam16_hint_difficulty_table.csv"
    ) in names
    assert "test_bundle/GenRec/temp/rl_beam_hint/instruments_grec_beam_hint_cascade_20260314_summary.json" in names
    assert "test_bundle/GenRec/data/Instruments_grec_fake_variant/rl/train.json" in names
    assert "test_bundle/raw_data/Instruments/Instruments.inter.json" in names


def test_unpack_extracts_bundle_into_target_directory(tmp_path: Path):
    layout = _build_fake_remote_layout(tmp_path)
    env = _build_env(layout, bundle_name="test_bundle")
    archive_path = tmp_path / "research_bundle.tar.gz"
    unpack_root = tmp_path / "unpacked"

    subprocess.run(
        ["bash", str(SCRIPT_PATH), "pack", str(archive_path)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["bash", str(SCRIPT_PATH), "unpack", str(archive_path), str(unpack_root)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    bundle_root = unpack_root / "test_bundle"
    assert (bundle_root / "bundle_manifest.json").exists()
    assert (
        bundle_root
        / "GenRec/output/jupyter-notebook/genrec-hint-cascade-artifacts/instruments_grec_beam16_hint_difficulty_table.csv"
    ).exists()
    assert (bundle_root / "raw_data/Instruments/Instruments.item.json").exists()


def test_pack_splits_large_bundle_into_configured_chunk_size_and_unpack_accepts_first_part(tmp_path: Path):
    layout = _build_fake_remote_layout(tmp_path)
    # Make the bundle large enough to force splitting under a small chunk size.
    _write_file(
        layout["repo_root"] / "scripts/hint_research/genrec-hint-cascade-analysis-2.ipynb",
        '{"cells":[' + ('"x"' * 4000) + '],"metadata":{},"nbformat":4,"nbformat_minor":5}',
    )
    env = _build_env(layout, bundle_name="test_bundle")
    env["CHUNK_SIZE"] = "512"

    archive_path = tmp_path / "research_bundle.tar.gz"
    unpack_root = tmp_path / "split_unpacked"

    subprocess.run(
        ["bash", str(SCRIPT_PATH), "pack", str(archive_path)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    part0 = Path(str(archive_path) + ".part.000")
    part1 = Path(str(archive_path) + ".part.001")
    assert part0.exists()
    assert part1.exists()
    assert part0.stat().st_size <= 512
    assert part1.stat().st_size <= 512

    subprocess.run(
        ["bash", str(SCRIPT_PATH), "unpack", str(part0), str(unpack_root)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    bundle_root = unpack_root / "test_bundle"
    assert (bundle_root / "bundle_manifest.json").exists()
    assert (bundle_root / "GenRec/scripts/hint_research/genrec-hint-cascade-analysis-2.ipynb").exists()
