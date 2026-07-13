"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_scoring.py

Tests for the stress-study response metric metastudy scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    PolicySpec,
    SfxiEvidenceFrame,
    SfxiSourceProvenance,
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation.candidates import (
    build_top_candidate_table,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation.recompute import (
    assert_canonical_sfxi_recompute,
    validate_canonical_sfxi_recompute,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation.scoring import (
    off_state_logic_level,
    score_sfxi_evidence,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation.selection import (
    select_top_rows,
)


def test_off_state_logic_level_uses_only_setpoint_zero_states() -> None:
    v_hat = np.asarray(
        [
            [0.1, 0.8, 0.3, 0.9],
            [0.6, 0.4, 0.2, 0.7],
        ]
    )
    setpoint = np.asarray([0.0, 1.0, 0.0, 1.0])

    assert np.allclose(off_state_logic_level(v_hat, setpoint), [0.2, 0.4])


def test_logic_gate_ranks_effect_only_after_gate() -> None:
    evidence = _sfxi_evidence_frame()
    policy = PolicySpec(
        id="gate055_effect",
        label="Gate logic>=0.55, then effect",
        kind="logic_gate",
        gamma=1.0,
        logic_gate=0.55,
    )

    scored = score_sfxi_evidence(evidence, policy)

    assert scored.iloc[0]["id"] == "passes-high-effect"
    assert scored.iloc[1]["id"] == "passes-low-effect"
    assert scored.iloc[-1]["id"] == "fails-high-effect"
    assert bool(scored.iloc[-1]["eligible"]) is False


def test_top_rows_never_include_logic_gate_ineligible_rows() -> None:
    evidence = _sfxi_evidence_frame()
    policy = PolicySpec(
        id="gate055_effect",
        label="Gate logic>=0.55, then effect",
        kind="logic_gate",
        gamma=1.0,
        logic_gate=0.95,
    )

    scored = score_sfxi_evidence(evidence, policy)

    assert select_top_rows(scored, top_k=6).empty


def test_top_rows_preserve_competition_rank_ties_at_the_selection_boundary() -> None:
    scored = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "score": [1.0, 1.0, 1.0, 0.5],
            "eligible": [True, True, True, True],
        }
    )

    selected = select_top_rows(scored, top_k=2)

    assert selected["id"].tolist() == ["a", "b", "c"]
    assert selected["rank"].tolist() == [1, 1, 1]


def test_top_candidate_table_never_invents_ineligible_fill_rows() -> None:
    evidence = _sfxi_evidence_frame()
    policy = PolicySpec(
        id="gate055_effect",
        label="Gate logic>=0.55, then effect",
        kind="logic_gate",
        gamma=1.0,
        logic_gate=0.95,
    )
    scored = {policy.id: {evidence.target_view.id: score_sfxi_evidence(evidence, policy)}}

    table = build_top_candidate_table((policy,), (evidence,), scored, top_k=6)

    assert table.empty


def test_target_view_rejects_masks_outside_the_canonical_sfxi_domain() -> None:
    with pytest.raises(ValueError, match="binary mask"):
        _sfxi_evidence_frame(setpoint=(0.0, 2.0, 0.0, 1.0))


def test_canonical_sfxi_recompute_mismatch_is_a_blocking_error() -> None:
    validation = {
        "matches_canonical_ledger": False,
        "max_abs_error": 1.0e-4,
        "per_selection_view": [{"selection_view_id": "ethanol", "max_abs_error": 1.0e-4}],
    }

    with pytest.raises(RuntimeError, match="canonical SFXI recomputation mismatch"):
        assert_canonical_sfxi_recompute(validation)


def test_canonical_sfxi_recompute_rejects_nonfinite_score_differences() -> None:
    evidence = _sfxi_evidence_frame()
    evidence.predictions["pred__score_selected"] = np.nan
    canonical = score_sfxi_evidence(
        evidence,
        PolicySpec(id="sfxi_beta1_gamma1", label="Canonical SFXI", kind="multiplicative"),
    )
    canonical["score"] = np.nan

    with pytest.raises(ValueError, match="non-finite score differences"):
        validate_canonical_sfxi_recompute((evidence,), {evidence.target_view.id: canonical})


def _sfxi_evidence_frame(
    *,
    setpoint: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> SfxiEvidenceFrame:
    target_view = StressTargetView(
        id="ethanol",
        label="Synthetic",
        target_mask=setpoint,
    )
    y_hat = np.asarray(
        [
            [0.05, 0.9, 0.05, 0.9, 0.0, 2.0, 0.0, 2.0],
            [0.2, 0.75, 0.2, 0.75, 0.0, 1.0, 0.0, 1.0],
            [0.8, 0.2, 0.8, 0.2, 0.0, 3.0, 0.0, 3.0],
        ],
        dtype=float,
    )
    predictions = pd.DataFrame(
        {
            "id": ["passes-high-effect", "passes-low-effect", "fails-high-effect"],
            "sequence": ["A", "C", "G"],
            "pred__y_hat_model": list(y_hat),
            "pred__score_selected": [0.0, 0.0, 0.0],
            "sel__rank_competition": [1, 2, 3],
            "sel__is_selected": [True, False, False],
            "obj__logic_fidelity": [0.0, 0.0, 0.0],
            "obj__effect_scaled": [0.0, 0.0, 0.0],
        }
    )
    return SfxiEvidenceFrame(
        source=SfxiSourceProvenance(
            source_id="test-sfxi-ethanol",
            source_campaign_slug="test-sfxi-ethanol",
            expected_run_id="r0",
            target_view_id="ethanol",
        ),
        target_view=target_view,
        predictions=predictions,
        y_hat=y_hat,
        denom=8.0,
        run_id="r0",
    )
