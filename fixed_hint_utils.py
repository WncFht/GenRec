from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_rl_beam_hint import extract_sid_tokens


def build_hint_text(ground_truth: str, hint_depth: int) -> str:
    return "".join(extract_sid_tokens(ground_truth)[: max(hint_depth, 0)])


def build_prompt_with_hint(example: dict[str, Any], formatter) -> str:
    prompt_text = formatter(example["prompt"])
    return f"{prompt_text}{example.get('oracle_hint_text', '')}"


def load_fixed_hint_depth_map(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_fixed_hint_depth_to_example(
    example: dict[str, Any],
    hint_map: dict[str, Any],
    cap_depth: int | None = None,
    unsolved_depth: int | None = None,
) -> dict[str, Any]:
    source_index = example.get("extra_info", {}).get("index")
    if source_index is None:
        raise KeyError("Missing extra_info.index for fixed hint lookup.")

    key = str(source_index)
    if key not in hint_map.get("hint_depth_by_index", {}):
        raise KeyError(f"Missing fixed hint depth for index={source_index}.")

    mapped_depth = int(hint_map["hint_depth_by_index"][key])
    effective_unsolved_depth = (
        int(unsolved_depth)
        if unsolved_depth is not None
        else int(hint_map.get("default_unsolved_depth", mapped_depth))
    )
    oracle_hint_unsolved = int(source_index) in {int(index) for index in hint_map.get("unsolved_indices", [])}
    if oracle_hint_unsolved:
        mapped_depth = effective_unsolved_depth
    if cap_depth is not None:
        mapped_depth = min(mapped_depth, int(cap_depth))

    enriched = dict(example)
    enriched["oracle_hint_depth"] = mapped_depth
    enriched["oracle_hint_text"] = build_hint_text(example["reward_model"]["ground_truth"], mapped_depth)
    enriched["oracle_hint_unsolved"] = oracle_hint_unsolved
    return enriched


def group_examples_by_hint_depth(examples: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[int(example["oracle_hint_depth"])].append(example)
    return {hint_depth: grouped[hint_depth] for hint_depth in sorted(grouped)}


def group_generation_inputs_by_hint_depth(inputs: list[dict[str, Any]]) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, example in enumerate(inputs):
        grouped[int(example["oracle_hint_depth"])].append((index, example))
    return {hint_depth: grouped[hint_depth] for hint_depth in sorted(grouped)}
