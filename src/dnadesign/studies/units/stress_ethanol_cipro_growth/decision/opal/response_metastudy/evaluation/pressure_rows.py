"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/pressure_rows.py

Reusable evaluators for adversarial pressure-test rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.policies import CANONICAL_SFXI_POLICY_ID


def upper_bound_status(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "indeterminate"
    return "fail" if value > threshold else "pass"


def model_support_row(summary: dict[str, object], *, minimum: float) -> dict[str, object]:
    value = float(summary["weakest_target_view_median_score_spearman"])
    finite = np.isfinite(value)
    status = "indeterminate" if not finite else ("pass" if value >= float(minimum) else "fail")
    interpretation = (
        "Held-out target-view ordering is undefined."
        if not finite
        else (
            "Held-out target-view ordering clears the provisional review guardrail."
            if status == "pass"
            else "Held-out target-view ordering is too weak to promote a metric policy."
        )
    )
    return pressure_row(
        agent="model_validation",
        check_id="held_out_target_view_ordering",
        status=status,
        severity="blocker",
        premise=(
            "A scalarizer cannot support a next-build decision when the vec8 predictor does not preserve "
            "held-out target-view ordering."
        ),
        evidence=(
            f"weakest_target_view_median_score_spearman={value:.3f}; "
            f"target_views={summary.get('target_view_median_score_spearman')}"
        ),
        threshold=f">= {float(minimum):.2f} provisional review guardrail",
        interpretation=interpretation,
        action="Keep policy promotion and synthesis paused; expand measured support or revise the predictive model.",
    )


def setpoint_support_rows(
    support: pd.DataFrame,
    *,
    logic_threshold: float,
    minimum_count: int,
) -> list[dict[str, object]]:
    selected = support[np.isclose(support["logic_threshold"], float(logic_threshold))]
    rows: list[dict[str, object]] = []
    for _, value in selected.sort_values("selection_view_id", kind="mergesort").iterrows():
        count = int(value["candidate_count"])
        selection_view_id = str(value["selection_view_id"])
        rows.append(
            pressure_row(
                agent="candidate_support",
                check_id=f"setpoint_support_{selection_view_id}",
                status="pass" if count >= minimum_count else "fail",
                severity="high",
                premise="The predicted candidate surface must contain enough response shapes near the setpoint.",
                evidence=(
                    f"selection_view_id={selection_view_id}; candidates_at_logic_{logic_threshold:.2f}={count}; "
                    f"max_logic={float(value['max_logic_fidelity']):.3f}"
                ),
                threshold=f"candidate_count >= {minimum_count} at logic_fidelity >= {logic_threshold:.2f}",
                interpretation=(
                    "A scalarizer cannot recover a response shape that is absent from the predicted candidate surface."
                ),
                action="Treat missing shape support as a model/data limitation before tuning selection weights.",
            )
        )
    return rows


def effect_dominance_rows(pairwise: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    canonical = pairwise[
        (pairwise["policy_id"] == CANONICAL_SFXI_POLICY_ID) & (pairwise["metric"] == "within_selection_view")
    ]
    for target_view_id in sorted(canonical["selection_view_a"].unique()):
        logic = _within_corr(canonical, target_view_id, "logic_fidelity")
        effect = _within_corr(canonical, target_view_id, "effect_scaled")
        finite = np.isfinite(logic) and np.isfinite(effect)
        status = "indeterminate" if not finite else ("fail" if abs(effect) > abs(logic) else "pass")
        if not finite:
            interpretation = (
                "Score-component dominance is indeterminate because one or both correlations are not finite."
            )
        elif status == "fail":
            interpretation = "Canonical SFXI is effect-dominated for this target view."
        else:
            interpretation = "Canonical SFXI is not effect-dominated for this target view."
        rows.append(
            pressure_row(
                agent="metric",
                check_id=f"canonical_sfxi_effect_dominance_{target_view_id}",
                status=status,
                severity="medium",
                premise="The selected score should not track effect more strongly than target-shape fidelity.",
                evidence=f"score_logic_pearson={logic:.3f}; score_effect_pearson={effect:.3f}",
                threshold="abs(score-effect correlation) <= abs(score-logic correlation)",
                interpretation=interpretation,
                action="Use this row to check whether beta/gamma tuning changes the dominant term.",
            )
        )
    return rows


def pressure_row(
    *,
    agent: str,
    check_id: str,
    status: str,
    severity: str,
    premise: str,
    evidence: str,
    threshold: str,
    interpretation: str,
    action: str,
) -> dict[str, object]:
    return {
        "agent": agent,
        "check_id": check_id,
        "status": status,
        "severity": severity,
        "premise": premise,
        "evidence": evidence,
        "threshold": threshold,
        "interpretation": interpretation,
        "action": action,
    }


def _within_corr(frame: pd.DataFrame, target_view_id: str, target: str) -> float:
    row = frame[(frame["selection_view_a"] == target_view_id) & (frame["selection_view_b"] == target)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["pearson"])
