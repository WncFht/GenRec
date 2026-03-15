import pytest

from analyze_rl_beam_hint import (
    aggregate_run_summary,
    aggregate_stage_summary,
    build_group_pattern,
    combine_hint_with_suffixes,
    convert_legacy_details_to_stage_rows,
    discover_reusable_cache,
    extract_sid_tokens,
    summarize_group,
)


def test_extract_sid_tokens_returns_ordered_sid_tokens():
    assert extract_sid_tokens("<a_12><b_34><c_56>") == ["<a_12>", "<b_34>", "<c_56>"]


def test_summarize_group_uses_rule_and_prefix_rewards():
    metrics = summarize_group(
        completions=[
            "<a_1><b_2><c_3>",
            "<a_1><b_2><c_9>",
            "<a_1><b_9><c_9>",
            "<a_9><b_9><c_9>",
        ],
        ground_truth="<a_1><b_2><c_3>",
    )

    assert metrics["rule_hit_count"] == 1
    assert metrics["rule_hit_any"] is True
    assert metrics["prefix_rewards"] == [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
    assert metrics["prefix_max"] == 1.0
    assert metrics["prefix_mean"] == pytest.approx(0.5)
    assert metrics["prefix_min"] == 0.0


def test_combine_hint_with_suffixes_restores_full_sid_prediction():
    assert combine_hint_with_suffixes("<a_7>", ["<b_8><c_9>", ""]) == ["<a_7><b_8><c_9>", "<a_7>"]


def test_build_group_pattern_compacts_match_length_counts():
    assert build_group_pattern([0, 0, 1, 1, 3, 4]) == "0:2|1:2|3:1|4:1"


def test_summarize_group_tracks_full_and_suffix_prefix_patterns():
    metrics = summarize_group(
        completions=[
            "<a_1><b_2><c_3><d_4>",
            "<a_1><b_2><x_9><d_4>",
            "<a_1><x_9><c_3><d_4>",
            "<x_9><b_2><c_3><d_4>",
        ],
        ground_truth="<a_1><b_2><c_3><d_4>",
        hint_depth=1,
    )

    assert metrics["gt_len"] == 4
    assert metrics["full_prefix_match_lens"] == [4, 2, 1, 0]
    assert metrics["full_group_pattern"] == "0:1|1:1|2:1|4:1"
    assert metrics["suffix_prefix_match_lens"] == [3, 1, 0, 0]
    assert metrics["suffix_group_pattern"] == "0:2|1:1|3:1"
    assert metrics["suffix_prefix_min"] == 0.0


def test_convert_legacy_details_to_stage_rows_reuses_base_and_hint1_rows():
    samples = [
        {"reward_model": {"ground_truth": "<a_1><b_2><c_3><d_4>"}},
        {"reward_model": {"ground_truth": "<a_5><b_6><c_7><d_8>"}},
    ]
    legacy_details = {
        "8": {
            "base_rows": [
                {
                    "sample_id": 0,
                    "source_index": 10,
                    "hint_text": "",
                    "group": {
                        "rule_rewards": [0.0, 0.0],
                        "rule_hit_count": 0,
                        "rule_hit_any": False,
                        "prefix_rewards": [0.25, 0.0],
                        "prefix_max": 0.25,
                        "prefix_mean": 0.125,
                        "prefix_min": 0.0,
                    },
                }
            ],
            "hinted_rows": {
                "0": {
                    "sample_id": 0,
                    "source_index": 10,
                    "hint_text": "<a_1>",
                    "group": {
                        "rule_rewards": [1.0, 0.0],
                        "rule_hit_count": 1,
                        "rule_hit_any": True,
                        "prefix_rewards": [1.0, 0.5],
                        "prefix_max": 1.0,
                        "prefix_mean": 0.75,
                        "prefix_min": 0.5,
                    },
                }
            },
        }
    }

    stages = convert_legacy_details_to_stage_rows(legacy_details, beam_size=8, samples=samples)

    assert list(stages) == ["base", "hint_1"]
    assert stages["base"][0]["hint_depth"] == 0
    assert stages["base"][0]["group"]["full_prefix_match_lens"] == [1, 0]
    assert stages["hint_1"][0]["hint_depth"] == 1
    assert stages["hint_1"][0]["group"]["suffix_prefix_match_lens"] == [3, 1]


def test_aggregate_stage_summary_tracks_patterns_and_cumulative_hits():
    rows = [
        {
            "sample_id": 0,
            "group": {
                "rule_hit_any": True,
                "rule_hit_count": 1,
                "full_prefix_match_lens": [4, 2],
                "suffix_prefix_match_lens": [3, 1],
                "full_group_pattern": "2:1|4:1",
                "suffix_group_pattern": "1:1|3:1",
            },
        },
        {
            "sample_id": 1,
            "group": {
                "rule_hit_any": False,
                "rule_hit_count": 0,
                "full_prefix_match_lens": [1, 0],
                "suffix_prefix_match_lens": [0, 0],
                "full_group_pattern": "0:1|1:1",
                "suffix_group_pattern": "0:2",
            },
        },
    ]

    summary = aggregate_stage_summary(
        rows=rows,
        total_samples=5,
        input_subset_size=2,
        previous_cumulative_hit_count=2,
        hint_depth=1,
        beam_size=8,
    )

    assert summary["stage_rule_hit_sample_count"] == 1
    assert summary["stage_rule_hit_sample_rate_within_input"] == pytest.approx(0.5)
    assert summary["remaining_subset_size"] == 1
    assert summary["cumulative_rule_hit_sample_count"] == 3
    assert summary["cumulative_rule_hit_sample_rate"] == pytest.approx(0.6)
    assert summary["full_prefix_match_hist_all_beams"] == {4: 1, 2: 1, 1: 1, 0: 1}
    assert summary["suffix_prefix_match_hist_all_beams"] == {3: 1, 1: 1, 0: 2}
    assert summary["group_pattern_counts_full"] == {"2:1|4:1": 1, "0:1|1:1": 1}


def test_discover_reusable_cache_picks_matching_latest_summary(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    good_summary = cache_dir / "good_summary.json"
    good_details = cache_dir / "good_details.json"
    good_summary.write_text(
        '{"model_path":"m","data_dir":"d","index_path":"i","num_samples":2,"beam_sizes":[8,16],"hint_depth":1}',
        encoding="utf-8",
    )
    good_details.write_text('{"8":{"base_rows":[],"hinted_rows":{}},"16":{"base_rows":[],"hinted_rows":{}}}', encoding="utf-8")

    bad_summary = cache_dir / "bad_summary.json"
    bad_details = cache_dir / "bad_details.json"
    bad_summary.write_text(
        '{"model_path":"other","data_dir":"d","index_path":"i","num_samples":2,"beam_sizes":[8,16],"hint_depth":1}',
        encoding="utf-8",
    )
    bad_details.write_text('{"8":{"base_rows":[],"hinted_rows":{}}}', encoding="utf-8")

    cache_paths = discover_reusable_cache(
        cache_dir=cache_dir,
        model_path="m",
        data_dir="d",
        index_path="i",
        beam_sizes=[8, 16],
        num_samples=2,
        offset=0,
        max_samples=None,
    )

    assert cache_paths == {"summary_path": good_summary, "details_path": good_details}


def test_aggregate_run_summary_tracks_hint_recovery_on_rule_misses():
    base_rows = [
        {
            "sample_id": 0,
            "group": {
                "rule_hit_any": True,
                "rule_hit_count": 1,
                "prefix_max": 1.0,
                "prefix_mean": 0.6,
                "prefix_min": 0.0,
                "prefix_rewards": [1.0, 0.2],
            },
        },
        {
            "sample_id": 1,
            "group": {
                "rule_hit_any": False,
                "rule_hit_count": 0,
                "prefix_max": 1.0 / 3.0,
                "prefix_mean": 1.0 / 6.0,
                "prefix_min": 0.0,
                "prefix_rewards": [1.0 / 3.0, 0.0],
            },
        },
        {
            "sample_id": 2,
            "group": {
                "rule_hit_any": False,
                "rule_hit_count": 0,
                "prefix_max": 0.0,
                "prefix_mean": 0.0,
                "prefix_min": 0.0,
                "prefix_rewards": [0.0, 0.0],
            },
        },
    ]
    hinted_rows = {
        1: {
            "sample_id": 1,
            "group": {
                "rule_hit_any": True,
                "rule_hit_count": 2,
                "prefix_max": 1.0,
                "prefix_mean": 0.75,
                "prefix_min": 0.5,
                "prefix_rewards": [1.0, 0.5],
            },
        },
        2: {
            "sample_id": 2,
            "group": {
                "rule_hit_any": False,
                "rule_hit_count": 0,
                "prefix_max": 1.0 / 3.0,
                "prefix_mean": 1.0 / 6.0,
                "prefix_min": 0.0,
                "prefix_rewards": [1.0 / 3.0, 0.0],
            },
        },
    }

    summary = aggregate_run_summary(base_rows, hinted_rows)

    assert summary["num_samples"] == 3
    assert summary["rule_hit_sample_count"] == 1
    assert summary["rule_hit_sample_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["rule_hit_total_count"] == 1
    assert summary["prefix_per_sample_max_avg"] == pytest.approx((1.0 + 1.0 / 3.0 + 0.0) / 3.0)
    assert summary["miss_subset_size"] == 2
    assert summary["hint_rule_hit_sample_count_within_miss"] == 1
    assert summary["hint_rule_hit_sample_rate_within_miss"] == pytest.approx(0.5)
    assert summary["hint_rule_hit_total_count_within_miss"] == 2
