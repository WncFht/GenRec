from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_pins_protobuf_below_major_7() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    protobuf_spec = next(dep for dep in dependencies if dep.startswith("protobuf"))

    assert "<7" in protobuf_spec, protobuf_spec
