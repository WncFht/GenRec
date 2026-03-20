import pytest

from analyze_rl_beam_hint import build_fixed_hint_depth_map_from_details
from fixed_hint_utils import (
    apply_fixed_hint_depth_to_example,
    build_fixed_hint_sample_key,
    build_hint_text,
    build_prompt_with_hint,
    group_examples_by_hint_depth,
    group_generation_inputs_by_hint_depth,
)


def test_build_fixed_hint_depth_map_uses_task_and_index_sample_keys():
    details_payload = {
        "results": {
            "16": {
                "stages": {
                    "base": {
                        "rows": [
                            {"sample_id": 0, "source_index": 100, "group": {"rule_hit_any": False}},
                            {"sample_id": 1, "source_index": 100, "group": {"rule_hit_any": True}},
                            {"sample_id": 2, "source_index": 101, "group": {"rule_hit_any": False}},
                        ]
                    },
                    "hint_1": {
                        "rows": [
                            {"sample_id": 0, "source_index": 100, "group": {"rule_hit_any": True}},
                            {"sample_id": 2, "source_index": 101, "group": {"rule_hit_any": False}},
                        ]
                    },
                    "hint_2": {
                        "rows": [
                            {"sample_id": 2, "source_index": 101, "group": {"rule_hit_any": False}},
                        ]
                    },
                }
            }
        }
    }
    samples = [
        {"extra_info": {"task": "task1_sid_sft", "index": 100}},
        {"extra_info": {"task": "task5_title_desc2sid", "index": 100}},
        {"extra_info": {"task": "task4_hisTitle2sid", "index": 101}},
    ]

    fixed_map = build_fixed_hint_depth_map_from_details(
        details_payload,
        beam_size=16,
        unsolved_depth=3,
        samples=samples,
    )

    assert fixed_map["beam_size"] == 16
    assert fixed_map["sample_key_type"] == "extra_info.task+index"
    assert fixed_map["hint_depth_by_sample_key"] == {
        build_fixed_hint_sample_key("task1_sid_sft", 100): 1,
        build_fixed_hint_sample_key("task4_hisTitle2sid", 101): 3,
        build_fixed_hint_sample_key("task5_title_desc2sid", 100): 0,
    }
    assert fixed_map["unsolved_sample_keys"] == [build_fixed_hint_sample_key("task4_hisTitle2sid", 101)]


def test_apply_fixed_hint_depth_to_example_uses_task_and_index_and_respects_cap():
    hint_map = {
        "sample_key_type": "extra_info.task+index",
        "hint_depth_by_sample_key": {
            build_fixed_hint_sample_key("task1_sid_sft", 100): 0,
            build_fixed_hint_sample_key("task4_hisTitle2sid", 102): 3,
        },
        "unsolved_sample_keys": [build_fixed_hint_sample_key("task4_hisTitle2sid", 102)],
        "default_unsolved_depth": 3,
    }
    example = {
        "extra_info": {"task": "task4_hisTitle2sid", "index": 102},
        "reward_model": {"ground_truth": "<a_1><b_2><c_3><d_4>"},
    }

    enriched = apply_fixed_hint_depth_to_example(example, hint_map, cap_depth=2)

    assert enriched["oracle_hint_depth"] == 2
    assert enriched["oracle_hint_text"] == "<a_1><b_2>"
    assert enriched["oracle_hint_unsolved"] is True


def test_apply_fixed_hint_depth_to_example_supports_legacy_index_only_maps():
    hint_map = {
        "hint_depth_by_index": {"100": 0, "101": 2, "102": 3},
        "unsolved_indices": [102],
        "default_unsolved_depth": 3,
    }
    example = {
        "extra_info": {"index": 102},
        "reward_model": {"ground_truth": "<a_1><b_2><c_3><d_4>"},
    }

    enriched = apply_fixed_hint_depth_to_example(example, hint_map, cap_depth=2)

    assert enriched["oracle_hint_depth"] == 2
    assert enriched["oracle_hint_text"] == "<a_1><b_2>"
    assert enriched["oracle_hint_unsolved"] is True


def test_build_hint_text_and_group_examples_by_hint_depth():
    assert build_hint_text("<a_1><b_2><c_3><d_4>", 3) == "<a_1><b_2><c_3>"

    grouped = group_examples_by_hint_depth(
        [
            {"sample_id": 0, "oracle_hint_depth": 0},
            {"sample_id": 1, "oracle_hint_depth": 2},
            {"sample_id": 2, "oracle_hint_depth": 2},
            {"sample_id": 3, "oracle_hint_depth": 1},
        ]
    )

    assert list(grouped) == [0, 1, 2]
    assert [example["sample_id"] for example in grouped[2]] == [1, 2]


def test_apply_fixed_hint_depth_requires_known_index():
    hint_map = {"hint_depth_by_index": {}, "unsolved_indices": [], "default_unsolved_depth": 3}
    example = {"extra_info": {}, "reward_model": {"ground_truth": "<a_1><b_2>"}}

    with pytest.raises(KeyError):
        apply_fixed_hint_depth_to_example(example, hint_map)


def test_apply_fixed_hint_depth_requires_task_for_composite_key_map():
    hint_map = {
        "sample_key_type": "extra_info.task+index",
        "hint_depth_by_sample_key": {build_fixed_hint_sample_key("task4_hisTitle2sid", 0): 1},
        "unsolved_sample_keys": [],
        "default_unsolved_depth": 3,
    }
    example = {"extra_info": {"index": 0}, "reward_model": {"ground_truth": "<a_1><b_2>"}}

    with pytest.raises(KeyError):
        apply_fixed_hint_depth_to_example(example, hint_map)


def test_build_prompt_with_hint_appends_hint_text_after_formatted_prompt():
    example = {
        "prompt": [{"role": "user", "content": "Predict next item."}],
        "oracle_hint_text": "<a_1><b_2>",
    }

    prompt_text = build_prompt_with_hint(example, formatter=lambda prompt: "PROMPT::")

    assert prompt_text == "PROMPT::<a_1><b_2>"


def test_group_generation_inputs_by_hint_depth_preserves_original_indices():
    grouped = group_generation_inputs_by_hint_depth(
        [
            {"oracle_hint_depth": 2, "sample_id": 10},
            {"oracle_hint_depth": 0, "sample_id": 11},
            {"oracle_hint_depth": 2, "sample_id": 12},
        ]
    )

    assert list(grouped) == [0, 2]
    assert grouped[0][0][0] == 1
    assert grouped[0][0][1]["sample_id"] == 11
    assert [item[1]["sample_id"] for item in grouped[2]] == [10, 12]
