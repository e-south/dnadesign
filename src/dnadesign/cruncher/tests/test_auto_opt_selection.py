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

from dnadesign.cruncher.app.sample_workflow import (
    AutoOptCandidate,
    _assess_candidate_quality,
    _build_final_sample_cfg,
    _select_auto_opt_candidate,
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
        "run_dir": run_dir,
        "run_dirs": [run_dir],
        "best_score": 1.0,
        "rhat": 1.01,
        "ess": 50.0,
        "unique_fraction": 0.5,
        "balance_median": 0.5,
        "diversity": 0.1,
        "improvement": 0.1,
        "acceptance_b": 0.4,
        "acceptance_m": 0.4,
        "swap_rate": None,
        "status": "ok",
        "quality": "ok",
        "warnings": [],
        "diagnostics": {},
    }
    payload.update(overrides)
    return AutoOptCandidate(**payload)


def test_auto_opt_prefers_shortest_length(tmp_path: Path) -> None:
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=True, prefer_shortest=True))
    cand_short = _candidate(tmp_path, "short", length=10, best_score=1.0)
    cand_long = _candidate(tmp_path, "long", length=12, best_score=10.0, balance_median=0.9)

    winner = _select_auto_opt_candidate([cand_short, cand_long], auto_cfg)
    assert winner.length == 10


def test_auto_opt_length_ranking_prefers_balance(tmp_path: Path) -> None:
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=True, prefer_shortest=False))
    cand_short = _candidate(tmp_path, "short", length=10, best_score=1.0, balance_median=0.9)
    cand_long = _candidate(tmp_path, "long", length=12, best_score=10.0, balance_median=0.2)

    winner = _select_auto_opt_candidate([cand_short, cand_long], auto_cfg)
    assert winner.length == 10


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
    assert any("rhat=" in note for note in notes)
    assert any("ess=" in note for note in notes)


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


def test_auto_opt_final_applies_cooling_boost() -> None:
    sample_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=10, draws=10, restarts=2),
        init=InitConfig(kind="random", length=10),
    )

    final_cfg, notes = _build_final_sample_cfg(sample_cfg, kind="gibbs", cooling_boost=2.0)
    assert final_cfg.optimizers.gibbs.beta_schedule.beta == (0.1, 1.0)
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
