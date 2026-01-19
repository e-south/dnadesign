"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_auto_opt_selection.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnadesign.cruncher.app.sample_workflow import (
    AutoOptCandidate,
    _aggregate_candidate_runs,
    _assess_candidate_quality,
    _best_score_final_from_sequences,
    _bootstrap_seed,
    _bootstrap_top_k_ci,
    _build_final_sample_cfg,
    _draw_scores_from_sequences,
    _pooled_bootstrap_seed,
    _select_auto_opt_candidate,
    _top_k_median_from_scores,
    _validate_auto_opt_candidates,
    _write_auto_opt_best_marker,
)
from dnadesign.cruncher.config.schema_v2 import (
    AutoOptConfig,
    AutoOptLengthConfig,
    InitConfig,
    SampleBudgetConfig,
    SampleConfig,
)


def _candidate(tmp_path: Path, name: str, **overrides) -> AutoOptCandidate:
    run_dir = tmp_path / name
    run_dir.mkdir()
    payload = {
        "kind": "gibbs",
        "length": 10,
        "budget": 200,
        "cooling_boost": 1.0,
        "move_profile": "balanced",
        "swap_prob": None,
        "ladder_size": None,
        "move_probs": None,
        "move_probs_label": None,
        "run_dir": run_dir,
        "run_dirs": [run_dir],
        "best_score": 1.0,
        "top_k_median_final": 1.0,
        "best_score_final": 1.0,
        "top_k_ci_low": 0.9,
        "top_k_ci_high": 1.1,
        "rhat": 1.01,
        "ess": 50.0,
        "unique_fraction": 0.5,
        "balance_median": 0.5,
        "diversity": 0.1,
        "improvement": 0.1,
        "acceptance_b": 0.4,
        "acceptance_m": 0.4,
        "acceptance_mh": 0.4,
        "swap_rate": None,
        "status": "ok",
        "quality": "ok",
        "warnings": [],
        "diagnostics": {},
    }
    payload.update(overrides)
    if "top_k_median_final" not in overrides and payload.get("best_score") is not None:
        payload["top_k_median_final"] = payload["best_score"]
    if "best_score_final" not in overrides and payload.get("best_score") is not None:
        payload["best_score_final"] = payload["best_score"]
    return AutoOptCandidate(**payload)


def test_auto_opt_prefers_shortest_length(tmp_path: Path) -> None:
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=True, prefer_shortest=True))
    cand_short = _candidate(tmp_path, "short", length=10, best_score=1.0)
    cand_long = _candidate(tmp_path, "long", length=12, best_score=10.0, balance_median=0.9)

    winner = _select_auto_opt_candidate([cand_short, cand_long], auto_cfg)
    assert winner.length == 10


def test_auto_opt_length_ranking_prefers_best_score(tmp_path: Path) -> None:
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=True, prefer_shortest=False))
    cand_short = _candidate(tmp_path, "short", length=10, best_score=1.0, balance_median=0.9)
    cand_long = _candidate(tmp_path, "long", length=12, best_score=10.0, balance_median=0.2)

    winner = _select_auto_opt_candidate([cand_short, cand_long], auto_cfg)
    assert winner.length == 12


def test_auto_opt_prefers_best_score_over_quality(tmp_path: Path) -> None:
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=False))
    cand_ok = _candidate(tmp_path, "ok", balance_median=0.2, best_score=1.0, quality="ok", status="ok")
    cand_warn = _candidate(
        tmp_path,
        "warn",
        balance_median=0.9,
        best_score=5.0,
        quality="warn",
        status="warn",
    )

    winner = _select_auto_opt_candidate([cand_warn, cand_ok], auto_cfg)
    assert winner.best_score == 5.0


def test_auto_opt_scorecard_not_blocked_by_trace_metrics(tmp_path: Path) -> None:
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=False))
    candidate = _candidate(
        tmp_path,
        "no_block",
        rhat=2.5,
        ess=2.0,
        unique_fraction=0.5,
        balance_median=0.6,
        diversity=0.2,
        acceptance_b=0.4,
        acceptance_m=0.4,
    )

    notes = _assess_candidate_quality(candidate, auto_cfg, mode="optimize")
    assert candidate.quality == "ok"
    assert notes == []


def test_auto_opt_all_fail_allowed_when_requested(tmp_path: Path) -> None:
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=False))
    candidate = _candidate(
        tmp_path,
        "fail",
        status="fail",
        quality="fail",
        best_score=None,
        balance_median=None,
        diversity=None,
        unique_fraction=None,
    )

    winner = _select_auto_opt_candidate([candidate], auto_cfg, allow_fail=True)
    assert winner.status == "fail"


def test_auto_opt_requires_ok_candidates_unless_allow_warn(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, "warn", status="warn", quality="warn", best_score=1.0)
    viable, allow_fail = _validate_auto_opt_candidates([candidate], allow_warn=True)
    assert viable == [candidate]
    assert allow_fail is False


def test_auto_opt_missing_diagnostics_requires_allow_warn(tmp_path: Path) -> None:
    candidate = _candidate(
        tmp_path,
        "fail",
        status="fail",
        quality="fail",
        best_score=None,
        top_k_median_final=None,
        best_score_final=None,
        top_k_ci_low=None,
        top_k_ci_high=None,
        balance_median=None,
        diversity=None,
        unique_fraction=None,
    )
    with pytest.raises(ValueError, match="failed catastrophic checks"):
        _validate_auto_opt_candidates([candidate], allow_warn=False)


def test_auto_opt_aggregate_uses_best_run_dir(tmp_path: Path) -> None:
    run_a = _candidate(tmp_path, "a", best_score=1.0, balance_median=0.2)
    run_b = _candidate(tmp_path, "b", best_score=2.0, balance_median=0.1)
    run_c = _candidate(tmp_path, "c", best_score=2.0, balance_median=0.5)
    agg = _aggregate_candidate_runs([run_a, run_b, run_c], budget=200, scorecard_top_k=5)
    assert agg.run_dir == run_c.run_dir


def test_auto_opt_final_applies_cooling_boost() -> None:
    sample_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=10, draws=10, restarts=2),
        init=InitConfig(kind="random", length=10),
    )

    final_cfg, notes = _build_final_sample_cfg(sample_cfg, kind="gibbs", cooling_boost=2.0)
    assert final_cfg.optimizers.gibbs.beta_schedule.beta == (2.0, 40.0)
    assert any("boosted cooling" in note for note in notes)


def test_auto_opt_best_marker_written(tmp_path: Path) -> None:
    payload = {
        "selected_candidate": {"kind": "gibbs", "length": 10, "pilot_run": "pilot_1"},
        "final_sample_run": "sample_1",
        "config_summary": "optimizer=gibbs",
    }
    marker_path = _write_auto_opt_best_marker(tmp_path, payload, run_group="lexA-cpxR")
    assert marker_path.exists()
    data = json.loads(marker_path.read_text())
    assert data["selected_candidate"]["pilot_run"] == "pilot_1"


def test_best_score_final_from_sequences_draw_phase() -> None:
    df = pd.DataFrame(
        {
            "phase": ["tune", "draw", "draw"],
            "combined_score_final": [1.0, 2.0, 1.5],
        }
    )
    assert _best_score_final_from_sequences(df) == 2.0


def test_best_score_final_missing_column() -> None:
    df = pd.DataFrame({"phase": ["draw"], "score": [1.0]})
    assert _best_score_final_from_sequences(df) is None


def test_top_k_median_bootstrap_deterministic() -> None:
    scores = _draw_scores_from_sequences(
        pd.DataFrame(
            {
                "phase": ["tune", "draw", "draw", "draw", "draw", "draw", "draw"],
                "combined_score_final": [10.0, 1.0, 2.0, 3.0, 1.5, 2.5, 0.5],
            }
        )
    )
    top_k = _top_k_median_from_scores(scores, 3)
    rng = np.random.default_rng(1234)
    ci = _bootstrap_top_k_ci(scores, k=3, rng=rng, n_boot=200)
    rng_repeat = np.random.default_rng(1234)
    ci_repeat = _bootstrap_top_k_ci(scores, k=3, rng=rng_repeat, n_boot=200)
    assert top_k == 2.5
    assert ci is not None
    assert ci[0] <= top_k <= ci[1]
    assert ci == ci_repeat


def test_bootstrap_seed_deterministic_across_run_dirs(tmp_path: Path) -> None:
    manifest = {
        "seed": 123,
        "sequence_length": 10,
        "regulator_set": {"tfs": ["lexA", "cpxR"]},
        "auto_opt": {"attempt": 1, "candidate": "gibbs", "budget": 200, "replicate": 0},
    }
    seed_a = _bootstrap_seed(manifest=manifest, run_dir=tmp_path / "run_a", kind="gibbs")
    seed_b = _bootstrap_seed(manifest=manifest, run_dir=tmp_path / "run_b", kind="gibbs")
    assert seed_a == seed_b


def test_pooled_bootstrap_seed_independent_of_order() -> None:
    manifest_a = {"seed": 1, "sequence_length": 10, "auto_opt": {"replicate": 0}}
    manifest_b = {"seed": 2, "sequence_length": 10, "auto_opt": {"replicate": 1}}
    seed_ab = _pooled_bootstrap_seed(manifests=[manifest_a, manifest_b], kind="gibbs", length=10, budget=200)
    seed_ba = _pooled_bootstrap_seed(manifests=[manifest_b, manifest_a], kind="gibbs", length=10, budget=200)
    assert seed_ab == seed_ba
