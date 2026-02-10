"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_diagnostics_overlap.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis import diagnostics


def test_diagnostics_uses_overlap_summary(monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("compute_overlap_tables should not be called when overlap_summary is provided")

    monkeypatch.setattr(diagnostics, "compute_overlap_tables", _boom)

    sequences_df = pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA"],
            "phase": ["draw", "draw"],
            "score_tf1": [0.2, 0.3],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["ACGT"],
            "score_tf1": [0.2],
            "id": ["elite-1"],
        }
    )
    hits_df = pd.DataFrame(
        {
            "elite_id": ["elite-1"],
            "tf": ["tf1"],
            "best_start": [0],
            "best_strand": ["+"],
            "pwm_width": [4],
            "best_core_seq": ["ACGT"],
        }
    )
    overlap_summary = {"overlap_rate_median": 0.5, "overlap_total_bp_median": 1.0}

    result = diagnostics.summarize_sampling_diagnostics(
        trace_idata=None,
        sequences_df=sequences_df,
        elites_df=elites_df,
        elites_hits_df=hits_df,
        tf_names=["tf1"],
        optimizer=None,
        optimizer_stats=None,
        mode="sample",
        optimizer_kind="gibbs_anneal",
        sample_meta={"top_k": 1},
        trace_required=False,
        overlap_summary=overlap_summary,
    )
    elites_metrics = result["metrics"]["elites"]
    assert elites_metrics["overlap_rate_median"] == 0.5
    assert elites_metrics["overlap_total_bp_median"] == 1.0
