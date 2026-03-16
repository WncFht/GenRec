from __future__ import annotations

import pandas as pd

from scripts.hint_research.explore_local_hint_bundle import (
    build_leaf_distinctiveness_frame,
    build_parent_title_family_rows,
    build_parent_pathology_rows,
)


def test_build_parent_pathology_rows_tracks_dominant_unsolved_leaf():
    frame = pd.DataFrame(
        [
            {
                "parent_d4": "<a><b><c>",
                "token_d4": "<d_1>",
                "final_unsolved": True,
                "effective_hint_depth": 4,
                "task_label": "sid",
            },
            {
                "parent_d4": "<a><b><c>",
                "token_d4": "<d_1>",
                "final_unsolved": True,
                "effective_hint_depth": 4,
                "task_label": "sid",
            },
            {
                "parent_d4": "<a><b><c>",
                "token_d4": "<d_2>",
                "final_unsolved": False,
                "effective_hint_depth": 3,
                "task_label": "hisTitle2sid",
            },
            {
                "parent_d4": "<x><y><z>",
                "token_d4": "<w_1>",
                "final_unsolved": True,
                "effective_hint_depth": 4,
                "task_label": "sid",
            },
        ]
    )

    rows = build_parent_pathology_rows(frame)

    assert rows[0]["parent_d4"] == "<a><b><c>"
    assert rows[0]["total_case_count"] == 3
    assert rows[0]["unsolved_case_count"] == 2
    assert rows[0]["dominant_unsolved_leaf"] == "<d_1>"
    assert rows[0]["dominant_unsolved_leaf_count"] == 2
    assert rows[0]["dominant_unsolved_leaf_share"] == 1.0


def test_build_leaf_distinctiveness_frame_marks_generic_leaf_as_less_distinctive():
    branch_items = pd.DataFrame(
        [
            {"leaf_token": "<d_1>", "title": "Walker Williams bourbon strap"},
            {"leaf_token": "<d_2>", "title": "Walker Williams bourbon strap skull"},
            {"leaf_token": "<d_3>", "title": "Walker Williams bourbon strap cross"},
        ]
    )
    leaf_stats = pd.DataFrame(
        [
            {"leaf_token": "<d_1>", "unsolved_case_count": 3, "total_case_count": 3},
            {"leaf_token": "<d_2>", "unsolved_case_count": 0, "total_case_count": 2},
            {"leaf_token": "<d_3>", "unsolved_case_count": 0, "total_case_count": 2},
        ]
    )

    distinctiveness = build_leaf_distinctiveness_frame(branch_items, leaf_stats)
    generic = distinctiveness[distinctiveness["leaf_token"] == "<d_1>"].iloc[0]
    red = distinctiveness[distinctiveness["leaf_token"] == "<d_2>"].iloc[0]
    blue = distinctiveness[distinctiveness["leaf_token"] == "<d_3>"].iloc[0]

    assert generic["unique_term_count"] == 0
    assert red["unique_term_count"] == 1
    assert blue["unique_term_count"] == 1
    assert generic["mean_jaccard_to_siblings"] > red["mean_jaccard_to_siblings"]
    assert generic["mean_jaccard_to_siblings"] > blue["mean_jaccard_to_siblings"]


def test_build_parent_title_family_rows_scores_near_duplicate_family_higher():
    branch_items = pd.DataFrame(
        [
            {"parent_d4": "<a><b><c>", "leaf_token": "<d_1>", "title": "Walker Williams bourbon strap red"},
            {"parent_d4": "<a><b><c>", "leaf_token": "<d_2>", "title": "Walker Williams bourbon strap blue"},
            {"parent_d4": "<x><y><z>", "leaf_token": "<w_1>", "title": "Clamp meter industrial tool"},
            {"parent_d4": "<x><y><z>", "leaf_token": "<w_2>", "title": "Guitar picks assorted colors"},
        ]
    )

    family_rows = pd.DataFrame(build_parent_title_family_rows(branch_items))
    similar = family_rows[family_rows["parent_d4"] == "<a><b><c>"].iloc[0]
    mixed = family_rows[family_rows["parent_d4"] == "<x><y><z>"].iloc[0]

    assert similar["leaf_count"] == 2
    assert similar["mean_pairwise_jaccard"] > mixed["mean_pairwise_jaccard"]
    assert "bourbon" in similar["shared_terms"]
