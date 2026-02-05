"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_auto_opt_scorecard_metric.py

Validates auto-opt ranking uses the configured scorecard metric.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.sample.auto_opt import AutoOptCandidate, _rank_auto_opt_candidates


def _candidate(*, pilot_score: float | None, top_k_median: float | None) -> AutoOptCandidate:
    return AutoOptCandidate(
        kind="pt",
        length=10,
        budget=10,
        cooling_boost=1.0,
        move_profile="balanced",
        swap_prob=None,
        ladder_size=None,
        move_probs=None,
        move_probs_label=None,
        run_dir=Path("."),
        run_dirs=[Path(".")],
        best_score=top_k_median,
        top_k_median_final=top_k_median,
        best_score_final=top_k_median,
        top_k_ci_low=None,
        top_k_ci_high=None,
        rhat=None,
        ess=None,
        unique_fraction=None,
        balance_median=None,
        diversity=None,
        improvement=None,
        acceptance_b=None,
        acceptance_m=None,
        acceptance_mh=None,
        swap_rate=None,
        status="ok",
        quality="ok",
        warnings=[],
        diagnostics={},
        pilot_score=pilot_score,
        median_relevance_raw=None,
        mean_pairwise_distance=None,
        unique_successes=None,
    )


def test_auto_opt_ranking_uses_mmr_scorecard() -> None:
    candidate_a = _candidate(pilot_score=0.5, top_k_median=0.9)
    candidate_b = _candidate(pilot_score=0.6, top_k_median=0.1)
    ranked = _rank_auto_opt_candidates([candidate_a, candidate_b])
    assert ranked[0].pilot_score == 0.6
