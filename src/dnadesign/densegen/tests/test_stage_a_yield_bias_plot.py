"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_yield_bias_plot.py

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
                "selection_pool_source": "eligible_unique",
                "diversity": {
                    "core_entropy": {
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
                "selection_pool_source": "eligible_unique",
                "diversity": {
                    "core_entropy": {
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
        assert labels == ["Generated", "Eligible", "Unique core", "MMR pool", "Retained"]
        assert axes_left[0].get_title() == "Stepwise sequence yield"
        assert axes_right[0].get_title() == "Diversified sequences: core positional entropy"
        assert axes_left[0].yaxis.get_offset_text().get_text() == ""
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
                "selection_pool_source": "eligible_unique",
                "diversity": {
                    "core_entropy": {
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
