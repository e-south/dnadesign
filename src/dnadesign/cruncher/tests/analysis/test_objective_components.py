"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_objective_components.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.objective import compute_objective_components


def test_objective_components_basic() -> None:
    df = pd.DataFrame(
        {
            "phase": ["draw", "draw"],
            "combined_score_final": [1.0, 2.0],
            "score_tfA": [0.2, 0.4],
            "score_tfB": [0.1, 0.5],
            "sequence": ["AC", "GT"],
        }
    )
    result = compute_objective_components(
        df,
        ["tfA", "tfB"],
        top_k=1,
        overlap_total_bp_median=2.0,
        dsdna_canonicalize=False,
    )
    assert result["best_score_final"] == 2.0
    assert result["top_k_median_final"] == 2.0
    assert result["overlap_total_bp_median"] == 2.0
    assert "worst_tf_frequency" in result
    assert result["unique_fraction_canonical"] is None


def test_objective_components_learning_metrics() -> None:
    df = pd.DataFrame(
        {
            "phase": ["draw", "draw", "draw", "draw"],
            "chain": [0, 0, 1, 1],
            "draw": [0, 1, 0, 1],
            "combined_score_final": [0.1, 0.15, 0.2, 0.18],
            "score_tfA": [0.2, 0.3, 0.4, 0.3],
            "score_tfB": [0.1, 0.2, 0.3, 0.2],
            "sequence": ["AA", "AT", "TA", "TT"],
        }
    )

    result = compute_objective_components(
        df,
        ["tfA", "tfB"],
        top_k=1,
        overlap_total_bp_median=2.0,
        dsdna_canonicalize=False,
        early_stop={"enabled": True, "patience": 1, "min_delta": 0.02},
    )

    learning = result.get("learning") or {}
    assert learning["best_score_draw"] == 0
    assert learning["best_score_chain"] == 1
    assert learning["last_improvement_draw"] == 1
    assert learning["plateau_draws"] == 0
    early_stop = learning.get("early_stop") or {}
    assert early_stop["enabled"] is True
    assert early_stop["per_chain"]["1"]["early_stop_draw"] == 1


def test_objective_components_early_stop_requires_unique_successes() -> None:
    df = pd.DataFrame(
        {
            "phase": ["draw", "draw", "draw"],
            "chain": [0, 0, 0],
            "draw": [0, 1, 2],
            "combined_score_final": [0.2, 0.2, 0.2],
            "score_tfA": [0.2, 0.2, 0.2],
            "score_tfB": [0.2, 0.2, 0.2],
            "sequence": ["AA", "AA", "AA"],
        }
    )

    result = compute_objective_components(
        df,
        ["tfA", "tfB"],
        top_k=1,
        dsdna_canonicalize=False,
        early_stop={
            "enabled": True,
            "patience": 1,
            "min_delta": 0.0,
            "require_min_unique": True,
            "min_unique": 2,
            "success_min_per_tf_norm": 0.5,
        },
    )

    early_stop = (result.get("learning") or {}).get("early_stop") or {}
    assert early_stop.get("eligible") is False
    assert early_stop.get("earliest_draw") is None
