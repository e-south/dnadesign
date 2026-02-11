"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_stage_a_yield_bias_plot.py

Stage-A yield/bias plot behaviors for labels and layout cues.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

from dnadesign.densegen.src.viz.plotting import _build_stage_a_yield_bias_figure


def test_stage_a_yield_bias_labels_and_ticks() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "tier_scheme": "pct_0.1_1_9",
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": 1.0,
        "eligible_score_hist": [
            {
                "regulator": "lexA",
                "pwm_consensus": "AATT",
                "pwm_consensus_iupac": "WWTT",
                "edges": [0.0, 2.0, 4.0],
                "counts": [1, 1],
                "tier0_score": 4.0,
                "tier1_score": 2.0,
                "tier2_score": 1.0,
                "generated": 1000,
                "candidates_with_hit": 500,
                "eligible_raw": 250,
                "eligible_unique": 200,
                "retained": 120,
                "selection_pool_size_final": 200,
                "diversity": {
                    "core_entropy": {
                        "top_candidates": {"values": [0.05, 0.1, 0.15, 0.2], "n": 2},
                        "diversified_candidates": {"values": [0.1, 0.2, 0.3, 0.4], "n": 2},
                    }
                },
                "padding_audit": None,
                "mining_audit": None,
            },
            {
                "regulator": "cpxR",
                "pwm_consensus": "CCGG",
                "pwm_consensus_iupac": "SSGG",
                "edges": [0.0, 1.0, 2.0],
                "counts": [1, 1],
                "tier0_score": 2.0,
                "tier1_score": 1.0,
                "tier2_score": 0.5,
                "generated": 800,
                "candidates_with_hit": 400,
                "eligible_raw": 200,
                "eligible_unique": 160,
                "retained": 90,
                "selection_pool_size_final": 160,
                "diversity": {
                    "core_entropy": {
                        "top_candidates": {"values": [0.0, 0.05, 0.1, 0.2], "n": 2},
                        "diversified_candidates": {"values": [0.0, 0.1, 0.2, 0.3], "n": 2},
                    }
                },
                "padding_audit": None,
                "mining_audit": None,
            },
        ],
    }
    pool_df = pd.DataFrame(
        {
            "tf": ["lexA", "lexA", "cpxR", "cpxR"],
            "tfbs": ["AAAAAA", "AAAAAAA", "CCCCCC", "CCCCCCC"],
            "best_hit_score": [3.5, 2.2, 1.8, 1.2],
        }
    )

    fig, axes_left, axes_right, _ = _build_stage_a_yield_bias_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        assert axes_left[-1].get_xlabel() == "Stage"
        labels = [tick.get_text() for tick in axes_left[-1].get_xticklabels() if tick.get_text()]
        assert labels == ["Generated", "Eligible", "Unique core", "Selection pool", "Retained"]
        header_axes = [ax for ax in fig.axes if ax.get_label() == "header"]
        assert not header_axes
        assert axes_left[0].get_title() == "Stepwise sequence yield"
        assert axes_right[0].get_title() == "Core positional entropy (top vs diversified)"
        assert axes_left[0].yaxis.get_offset_text().get_text() == ""
        assert axes_left[-1].xaxis.label.get_size() == pytest.approx(axes_right[-1].xaxis.label.get_size())
        left_x_ticks = [tick.get_size() for tick in axes_left[-1].get_xticklabels() if tick.get_text()]
        right_x_ticks = [tick.get_size() for tick in axes_right[-1].get_xticklabels() if tick.get_text()]
        right_y_ticks = [tick.get_size() for tick in axes_right[-1].get_yticklabels() if tick.get_text()]
        assert left_x_ticks and right_x_ticks
        assert min(left_x_ticks) == pytest.approx(max(left_x_ticks))
        assert min(right_x_ticks) == pytest.approx(max(right_x_ticks))
        assert right_y_ticks
        assert left_x_ticks[0] < right_y_ticks[0]
        left_pos = axes_left[0].get_position()
        right_pos = axes_right[0].get_position()
        assert left_pos.width > right_pos.width
        assert (right_pos.x0 - left_pos.x1) <= 0.10
        stage_annotation_sizes = [
            text.get_fontsize() for text in axes_left[0].texts if "\n" in text.get_text() and "%" in text.get_text()
        ]
        assert stage_annotation_sizes
        assert stage_annotation_sizes[0] >= 7.2
        entropy_labels = [text for text in fig.texts if text.get_text() == "Entropy (bits)"]
        assert len(entropy_labels) == 1
        x_label_size = axes_right[-1].xaxis.label.get_size()
        assert entropy_labels[0].get_fontsize() == pytest.approx(x_label_size)
    finally:
        fig.clf()


def test_stage_a_yield_bias_requires_iupac_consensus() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "tier_scheme": "pct_0.1_1_9",
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": 1.0,
        "eligible_score_hist": [
            {
                "regulator": "lexA",
                "pwm_consensus": "AATT",
                "edges": [0.0, 2.0, 4.0],
                "counts": [1, 1],
                "tier0_score": 4.0,
                "tier1_score": 2.0,
                "tier2_score": 1.0,
                "generated": 1000,
                "candidates_with_hit": 500,
                "eligible_raw": 250,
                "eligible_unique": 200,
                "retained": 120,
                "selection_pool_size_final": 200,
                "diversity": {
                    "core_entropy": {
                        "top_candidates": {"values": [0.05, 0.1, 0.15, 0.2], "n": 2},
                        "diversified_candidates": {"values": [0.1, 0.2, 0.3, 0.4], "n": 2},
                    }
                },
                "padding_audit": None,
                "mining_audit": None,
            }
        ],
    }
    pool_df = pd.DataFrame(
        {
            "tf": ["lexA"],
            "tfbs": ["AAAAAA"],
            "best_hit_score": [3.5],
        }
    )
    with pytest.raises(ValueError, match="pwm_consensus_iupac"):
        _build_stage_a_yield_bias_figure(
            input_name="demo_input",
            pool_df=pool_df,
            sampling=sampling,
            style={},
        )
