"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_objective_components.py

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
