#!/usr/bin/env python3
"""Build and resolve explicit evaluation profiles for checkpoint watchers."""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=(.+?)\s*$")
YAML_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
CHECKPOINT_RE = re.compile(r"^checkpoint-\d+$")
POSITIONAL_ARG_RE = re.compile(r"^\$(?:[0-9]+|[@*#?])$")
SUPPORTED_VARIANT_PREFIXES = (
    "Industrial_and_Scientific",
    "Instruments",
    "Games",
    "Arts",
)


def strip_inline_shell_comment(raw: str) -> str:
    quote_char = ""
    escaped = False
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote_char:
            if char == quote_char:
                quote_char = ""
            continue
        if char in {"'", '"'}:
            quote_char = char
            continue
        if char == "#":
            return raw[:index].rstrip()
    return raw.strip()


def strip_matching_quotes(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return raw


def find_matching_brace(text: str, start_index: int) -> int:
    depth = 1
    index = start_index
    while index < len(text):
        if text.startswith("${", index):
            depth += 1
            index += 2
            continue
        if text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def split_default_expression(content: str) -> tuple[str, str | None]:
    index = 0
    depth = 0
    while index < len(content):
        if content.startswith("${", index):
            depth += 1
            index += 2
            continue
        if content[index] == "}":
            depth = max(depth - 1, 0)
            index += 1
            continue
        if depth == 0 and content.startswith(":-", index):
            return (content[:index], content[index + 2 :])
        index += 1
    return (content, None)


def parse_shell_assignments(path: Path) -> dict[str, str]:
    raw_assignments: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name = match.group(1)
        raw_value = strip_inline_shell_comment(match.group(2))
        if not raw_value:
            continue
        raw_value = strip_matching_quotes(raw_value)
        if name in raw_assignments and POSITIONAL_ARG_RE.match(raw_value):
            continue
        raw_assignments[name] = raw_value
    return raw_assignments


def expand_shell_text(
    text: str,
    assignments: dict[str, str],
    cache: dict[str, str],
    stack: set[str],
) -> str:
    parts: list[str] = []
    index = 0
    while index < len(text):
        if not text.startswith("${", index):
            parts.append(text[index])
            index += 1
            continue
        end_index = find_matching_brace(text, index + 2)
        if end_index < 0:
            parts.append(text[index:])
            break
        content = text[index + 2 : end_index]
        variable_name, default_value = split_default_expression(content)
        variable_name = variable_name.strip()
        if variable_name in assignments:
            resolved = resolve_shell_assignment(variable_name, assignments, cache, stack)
            if not resolved and default_value is not None:
                resolved = expand_shell_text(default_value, assignments, cache, stack)
        elif default_value is not None:
            resolved = expand_shell_text(default_value, assignments, cache, stack)
        else:
            resolved = ""
        parts.append(resolved)
        index = end_index + 1
    return "".join(parts)


def resolve_shell_assignment(
    name: str,
    assignments: dict[str, str],
    cache: dict[str, str],
    stack: set[str],
) -> str:
    if name in cache:
        return cache[name]
    if name in stack:
        return ""
    stack.add(name)
    raw_value = assignments.get(name, "")
    resolved = expand_shell_text(raw_value, assignments, cache, stack)
    cache[name] = resolved
    stack.remove(name)
    return resolved


def resolve_all_shell_assignments(path: Path) -> dict[str, str]:
    raw_assignments = parse_shell_assignments(path)
    cache: dict[str, str] = {}
    for name in raw_assignments:
        resolve_shell_assignment(name, raw_assignments, cache, set())
    return cache


def parse_simple_yaml(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = YAML_KEY_RE.match(line)
        if match is None:
            continue
        key = match.group(1)
        value = strip_matching_quotes(match.group(2).strip())
        parsed[key] = value
    return parsed


def variant_from_dataset_key(key: str) -> str:
    for suffix in ("_train", "_valid", "_test"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def category_from_variant(variant: str) -> str:
    prefixes = (
        "Industrial_and_Scientific",
        "Instruments_grec_rlsidonly",
        "Instruments_grec",
        "Instruments_mimionerec",
        "Games_grec",
        "Arts_grec",
        "Instruments",
        "Games",
        "Arts",
    )
    for prefix in prefixes:
        if variant.startswith(prefix):
            return prefix
    return variant


def is_supported_variant(variant: str) -> bool:
    return any(variant.startswith(prefix) for prefix in SUPPORTED_VARIANT_PREFIXES)


def normalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    datasets = {
        name: manifest["datasets"][name]
        for name in sorted(manifest.get("datasets", {}).keys())
    }
    aliases = {
        name: {
            "dataset_variant": manifest["aliases"][name]["dataset_variant"],
            "sources": sorted(set(manifest["aliases"][name].get("sources", []))),
        }
        for name in sorted(manifest.get("aliases", {}).keys())
    }
    return {
        "version": 1,
        "datasets": datasets,
        "aliases": aliases,
    }


def register_dataset(manifest: dict[str, Any], variant: str) -> None:
    if not is_supported_variant(variant):
        return
    datasets = manifest["datasets"]
    if variant in datasets:
        return
    datasets[variant] = {
        "category": category_from_variant(variant),
        "test_data_path": f"{variant}/sft/test.json",
        "index_path": f"{variant}/id2sid.json",
    }


def register_alias(
    manifest: dict[str, Any],
    alias: str,
    variant: str,
    *,
    source: str,
) -> None:
    cleaned_alias = alias.strip()
    if not cleaned_alias:
        return
    if "$" in cleaned_alias:
        return
    register_dataset(manifest, variant)
    aliases = manifest["aliases"]
    existing = aliases.get(cleaned_alias)
    if existing is None:
        aliases[cleaned_alias] = {
            "dataset_variant": variant,
            "sources": [source],
        }
        return
    if existing["dataset_variant"] != variant:
        raise ValueError(
            f"alias {cleaned_alias!r} maps to both {existing['dataset_variant']!r} and {variant!r}"
        )
    existing.setdefault("sources", []).append(source)


def basename_or_none(path_text: str | None) -> str | None:
    if not path_text:
        return None
    return Path(path_text).name or None


def model_root_from_model_path(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if CHECKPOINT_RE.match(path.name):
        return path.parent.name or None
    return path.name or None


def dataset_variant_from_test_path(test_path: str | None) -> str | None:
    if not test_path:
        return None
    parts = Path(test_path).parts
    if len(parts) < 3:
        return None
    if parts[-2:] != ("sft", "test.json"):
        return None
    return parts[-3]


def load_dataset_manifest_entries(data_root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {"datasets": {}, "aliases": {}, "_dataset_keys": {}}
    dataset_info_path = data_root / "dataset_info.json"
    if dataset_info_path.is_file():
        payload = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        for entry_name, entry in payload.items():
            file_name = str(entry.get("file_name", ""))
            parts = Path(file_name).parts
            if not parts:
                continue
            variant = parts[0]
            manifest["_dataset_keys"][entry_name] = variant
            register_dataset(manifest, variant)
            dataset_entry = manifest["datasets"][variant]
            if entry_name.endswith("_train"):
                dataset_entry["train_entry"] = entry_name
            if entry_name.endswith("_valid"):
                dataset_entry["valid_entry"] = entry_name
    for child in sorted(data_root.iterdir(), key=lambda item: item.name) if data_root.is_dir() else []:
        if not child.is_dir():
            continue
        register_dataset(manifest, child.name)
    return manifest


def collect_yaml_aliases(repo_root: Path, manifest: dict[str, Any]) -> None:
    examples_root = repo_root / "examples" / "train_full"
    if not examples_root.is_dir():
        return
    for path in sorted(examples_root.rglob("*.yaml")):
        payload = parse_simple_yaml(path)
        dataset_name = payload.get("eval_dataset") or payload.get("dataset")
        if not dataset_name:
            continue
        variant = manifest.get("_dataset_keys", {}).get(dataset_name) or variant_from_dataset_key(dataset_name)
        if not is_supported_variant(variant):
            continue
        register_dataset(manifest, variant)
        output_dir = basename_or_none(payload.get("output_dir"))
        if output_dir:
            register_alias(
                manifest,
                output_dir,
                variant,
                source=f"{path.relative_to(repo_root)}#yaml_output_dir",
            )
        run_name = payload.get("run_name")
        if run_name:
            register_alias(
                manifest,
                run_name,
                variant,
                source=f"{path.relative_to(repo_root)}#yaml_run_name",
            )


def collect_shell_aliases(repo_root: Path, manifest: dict[str, Any]) -> None:
    hope_root = repo_root / "hope"
    if not hope_root.is_dir():
        return
    for path in sorted(hope_root.rglob("*.sh")):
        payload = resolve_all_shell_assignments(path)
        variant = (
            payload.get("DATA_VARIANT_DEFAULT")
            or dataset_variant_from_test_path(payload.get("TEST_DATA_PATH"))
            or dataset_variant_from_test_path(payload.get("TEST_DATA_PATH_DEFAULT"))
        )
        if not variant:
            continue
        if not is_supported_variant(variant):
            continue
        register_dataset(manifest, variant)
        source_prefix = str(path.relative_to(repo_root))
        for alias_value, alias_kind in (
            (basename_or_none(payload.get("OUTPUT_DIR_DEFAULT")), "output_dir_default"),
            (basename_or_none(payload.get("OUTPUT_DIR")), "output_dir"),
            (payload.get("RUN_NAME_DEFAULT"), "run_name_default"),
            (payload.get("RUN_NAME"), "run_name"),
            (basename_or_none(payload.get("SFT_ROOT")), "sft_root"),
            (basename_or_none(payload.get("SFT_ROOT_DEFAULT")), "sft_root_default"),
        ):
            if not alias_value:
                continue
            register_alias(
                manifest,
                alias_value,
                variant,
                source=f"{source_prefix}#{alias_kind}",
            )


def apply_overrides(overrides_path: Path, manifest: dict[str, Any]) -> None:
    if not overrides_path.is_file():
        return
    payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    for alias, variant in payload.get("aliases", {}).items():
        if not isinstance(alias, str) or not isinstance(variant, str):
            continue
        register_alias(
            manifest,
            alias,
            variant,
            source=f"{overrides_path.name}#manual_override",
        )


def build_manifest(repo_root: Path, data_root: Path, overrides_path: Path | None = None) -> dict[str, Any]:
    manifest = load_dataset_manifest_entries(data_root)
    collect_yaml_aliases(repo_root, manifest)
    collect_shell_aliases(repo_root, manifest)
    if overrides_path is not None:
        apply_overrides(overrides_path, manifest)
    return normalize_manifest(manifest)


@lru_cache(maxsize=8)
def load_manifest_cached(
    repo_root_text: str,
    data_root_text: str,
    manifest_path_text: str,
    overrides_path_text: str,
) -> dict[str, Any]:
    repo_root = Path(repo_root_text)
    data_root = Path(data_root_text)
    manifest_path = Path(manifest_path_text)
    overrides_path = Path(overrides_path_text)
    if manifest_path.is_file():
        return normalize_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    return build_manifest(repo_root, data_root, overrides_path)


def resolve_profile(
    repo_root: Path,
    data_root: Path,
    model_name: str,
    *,
    manifest_path: Path,
    overrides_path: Path,
) -> dict[str, Path | str] | None:
    manifest = load_manifest_cached(
        str(repo_root.resolve()),
        str(data_root.resolve()),
        str(manifest_path.resolve()),
        str(overrides_path.resolve()),
    )
    alias_entry = manifest.get("aliases", {}).get(model_name)
    if alias_entry is None:
        return None
    dataset_variant = str(alias_entry["dataset_variant"])
    dataset_entry = manifest["datasets"].get(dataset_variant)
    if dataset_entry is None:
        return None
    return {
        "category": str(dataset_entry["category"]),
        "test_data_path": data_root / str(dataset_entry["test_data_path"]),
        "index_path": data_root / str(dataset_entry["index_path"]),
        "data_profile": f"manifest:dataset_variant={dataset_variant};alias={model_name}",
        "cb_width": "n/a",
    }


def write_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and resolve explicit evaluation profiles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest", help="Build the manifest file from repo metadata")
    build.add_argument("--repo-root", default=".")
    build.add_argument("--data-root", default="./data")
    build.add_argument("--output", default="./data/eval_profile_manifest.json")
    build.add_argument("--overrides", default="./data/eval_profile_overrides.json")

    resolve = subparsers.add_parser("resolve", help="Resolve a model/output name to an eval profile")
    resolve.add_argument("--repo-root", default=".")
    resolve.add_argument("--data-root", default="./data")
    resolve.add_argument("--manifest", default="./data/eval_profile_manifest.json")
    resolve.add_argument("--overrides", default="./data/eval_profile_overrides.json")
    resolve.add_argument("--model-name", required=True)
    resolve.add_argument("--format", choices=("json", "tsv"), default="json")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()

    if args.command == "build-manifest":
        manifest = build_manifest(repo_root, data_root, Path(args.overrides).expanduser().resolve())
        write_manifest(Path(args.output).expanduser().resolve(), manifest)
        return 0

    if args.command == "resolve":
        profile = resolve_profile(
            repo_root,
            data_root,
            args.model_name,
            manifest_path=Path(args.manifest).expanduser().resolve(),
            overrides_path=Path(args.overrides).expanduser().resolve(),
        )
        if profile is None:
            return 1
        if args.format == "json":
            serializable = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in profile.items()
            }
            print(json.dumps(serializable, ensure_ascii=True))
            return 0
        print(
            "\t".join(
                [
                    str(profile["category"]),
                    str(profile["test_data_path"]),
                    str(profile["index_path"]),
                    str(profile["data_profile"]),
                    str(profile["cb_width"]),
                ]
            )
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
