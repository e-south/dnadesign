"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_comparison_panel.py

Tests for response metric metastudy policy-comparison candidate panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    comparison_panel as panel_eval,
)


def test_comparison_panel_keeps_roles_explicit_and_non_synthesis() -> None:
    candidates = _candidate_rows()
    recommendation = {"comparison_policy_id": "tradeoff_logic0p85"}

    panel = panel_eval.build_policy_comparison_panel(
        candidates,
        recommendation=recommendation,
        observed_label_ids={"shared-1"},
        per_target_view=1,
    )

    assert {
        "canonical_sfxi_high_effect",
        "comparison_shape_effect",
        "high_logic_lower_effect",
        "off_state_logic_penalized",
        "canonical_sfxi_shared_overlap",
        "target_view_specific_comparison",
    } <= set(panel["panel_role"])
    assert set(panel["selection_posture"]) == {"metric_comparison_not_synthesis"}
    assert panel[["panel_role", "policy_id", "selection_view_id", "id"]].duplicated().sum() == 0
    assert panel.loc[panel["id"] == "shared-1", "observed_label_member"].any()


def test_comparison_panel_fails_fast_when_required_policy_is_missing() -> None:
    candidates = _candidate_rows()

    try:
        panel_eval.build_policy_comparison_panel(
            candidates[candidates["policy_id"] != "lexicographic_logic_effect"],
            recommendation={"comparison_policy_id": "tradeoff_logic0p85"},
        )
    except ValueError as exc:
        assert "lexicographic_logic_effect" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("expected missing-policy failure")


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policies = (
        "sfxi_beta1_gamma1",
        "tradeoff_logic0p85",
        "lexicographic_logic_effect",
        "off_state_logic_eta2_beta2_gamma05",
    )
    for policy_id in policies:
        for target_view_id in ("ethanol", "ciprofloxacin", "and"):
            for idx, candidate_id in enumerate(("shared-1", f"{target_view_id}-{policy_id}-1"), start=1):
                rows.append(
                    {
                        "policy_id": policy_id,
                        "selection_view_id": target_view_id,
                        "rank": idx,
                        "id": candidate_id,
                        "sequence": "ACGT",
                        "score": 1.0 / idx,
                        "logic_fidelity": 0.6 - (idx * 0.05),
                        "effect_scaled": 0.4 + (idx * 0.02),
                        "off_state_logic_level": 0.1 * idx,
                        "v00": 0.1,
                        "v10": 0.8,
                        "v01": 0.2,
                        "v11": 0.9,
                        "selection_view_count": 3 if candidate_id == "shared-1" else 1,
                    }
                )
    return pd.DataFrame(rows)
