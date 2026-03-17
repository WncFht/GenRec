from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.hint_research.analyze_hint_subtree_relation import (
    build_node_stats,
    build_report_html,
)


def make_enriched_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "task_label": "sid",
                "effective_hint_depth": 0,
                "final_unsolved": False,
                "token_d1": "<a_1>",
                "parent_d1": "ROOT",
                "subtree_d1": 10.0,
                "token_d2": "<b_1>",
                "parent_d2": "<a_1>",
                "subtree_d2": 3.0,
                "token_d3": "<c_1>",
                "parent_d3": "<a_1><b_1>",
                "subtree_d3": 1.0,
                "token_d4": "<d_1>",
                "parent_d4": "<a_1><b_1><c_1>",
                "subtree_d4": 1.0,
            },
            {
                "sample_id": "s2",
                "task_label": "sid",
                "effective_hint_depth": 2,
                "final_unsolved": False,
                "token_d1": "<a_1>",
                "parent_d1": "ROOT",
                "subtree_d1": 10.0,
                "token_d2": "<b_1>",
                "parent_d2": "<a_1>",
                "subtree_d2": 3.0,
                "token_d3": "<c_1>",
                "parent_d3": "<a_1><b_1>",
                "subtree_d3": 1.0,
                "token_d4": "<d_1>",
                "parent_d4": "<a_1><b_1><c_1>",
                "subtree_d4": 1.0,
            },
            {
                "sample_id": "s3",
                "task_label": "sid",
                "effective_hint_depth": 4,
                "final_unsolved": True,
                "token_d1": "<a_1>",
                "parent_d1": "ROOT",
                "subtree_d1": 10.0,
                "token_d2": "<b_1>",
                "parent_d2": "<a_1>",
                "subtree_d2": 3.0,
                "token_d3": "<c_1>",
                "parent_d3": "<a_1><b_1>",
                "subtree_d3": 1.0,
                "token_d4": "<d_1>",
                "parent_d4": "<a_1><b_1><c_1>",
                "subtree_d4": 1.0,
            },
            {
                "sample_id": "h1",
                "task_label": "hisTitle2sid",
                "effective_hint_depth": 1,
                "final_unsolved": False,
                "token_d1": "<a_1>",
                "parent_d1": "ROOT",
                "subtree_d1": 10.0,
                "token_d2": "<b_2>",
                "parent_d2": "<a_1>",
                "subtree_d2": 7.0,
                "token_d3": "<c_2>",
                "parent_d3": "<a_1><b_2>",
                "subtree_d3": 1.0,
                "token_d4": "<d_2>",
                "parent_d4": "<a_1><b_2><c_2>",
                "subtree_d4": 1.0,
            },
        ]
    )


def test_build_node_stats_aggregates_hint_strength_and_stage_local_rates() -> None:
    node_stats = build_node_stats(make_enriched_frame())

    depth1_all = node_stats[
        (node_stats["task_label"] == "all")
        & (node_stats["depth"] == 1)
        & (node_stats["node_key"] == "<a_1>")
    ].iloc[0]
    assert depth1_all["sample_count"] == 4
    assert depth1_all["subtree_size"] == 10.0
    assert depth1_all["mean_effective_hint_depth"] == 1.75
    assert depth1_all["need_hint_rate"] == 0.75
    assert depth1_all["hint2_plus_rate"] == 0.50
    assert depth1_all["hint3_plus_rate"] == 0.25
    assert depth1_all["unsolved_rate"] == 0.25
    assert depth1_all["stage_local_rate"] == 0.75
    assert depth1_all["stage_local_name"] == "base_to_need_hint"

    depth2_sid = node_stats[
        (node_stats["task_label"] == "sid")
        & (node_stats["depth"] == 2)
        & (node_stats["node_key"] == "<a_1><b_1>")
    ].iloc[0]
    assert depth2_sid["sample_count"] == 3
    assert depth2_sid["subtree_size"] == 3.0
    assert depth2_sid["stage_local_eligible_count"] == 2
    assert depth2_sid["stage_local_rate"] == 1.0
    assert depth2_sid["stage_local_name"] == "hint1_to_need_hint2_plus"

    depth2_his = node_stats[
        (node_stats["task_label"] == "hisTitle2sid")
        & (node_stats["depth"] == 2)
        & (node_stats["node_key"] == "<a_1><b_2>")
    ].iloc[0]
    assert depth2_his["sample_count"] == 1
    assert depth2_his["subtree_size"] == 7.0
    assert depth2_his["stage_local_eligible_count"] == 1
    assert depth2_his["stage_local_rate"] == 0.0


def test_build_report_html_references_plots_and_summary() -> None:
    summary = {
        "overall": {
            "strongest_positive_correlation": {
                "task_label": "sid",
                "depth": 2,
                "metric": "mean_effective_hint_depth",
                "pearson": 0.61,
            }
        }
    }
    plot_paths = {
        "scatter": [
            Path("hint_subtree_relation_plots/scatter_all_depth1.png"),
            Path("hint_subtree_relation_plots/scatter_sid_depth2.png"),
        ],
        "bins": [Path("hint_subtree_relation_plots/bin_lines_all_depth1.png")],
        "heatmap": [Path("hint_subtree_relation_plots/correlation_heatmap_pearson.png")],
        "outliers": [Path("hint_subtree_relation_plots/outliers_sid_depth2.png")],
    }

    html = build_report_html(summary=summary, plot_paths=plot_paths)

    assert "提示强度 vs 子树规模" in html
    assert "关键结论" in html
    assert "任务概览" in html
    assert "指标词典" in html
    assert "图册筛选" in html
    assert 'data-task="sid"' in html
    assert "当前筛选条件下没有可显示的图表" in html
    assert "横轴是 log1p(子树规模)" in html
    assert "每个点代表一个节点" in html
    assert "position: sticky" not in html
    assert "scatter_all_depth1.png" in html
    assert "correlation_heatmap_pearson.png" in html
    assert "strongest_positive_correlation" in html
