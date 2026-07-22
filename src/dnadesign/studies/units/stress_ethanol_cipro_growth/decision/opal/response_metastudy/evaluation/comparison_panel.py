"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/comparison_panel.py

Policy-comparison candidate-panel construction for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ..core.policies import CANONICAL_SFXI_POLICY_ID

REQUIRED_PANEL_POLICIES = (
    CANONICAL_SFXI_POLICY_ID,
    "lexicographic_logic_effect",
    "off_state_logic_eta2_beta2_gamma05",
)

ROLE_DESCRIPTIONS = {
    "canonical_sfxi_high_effect": "Canonical SFXI high-effect rows retained as a negative-control view.",
    "comparison_shape_effect": "Shape-ceiling comparison rows used to inspect logic/effect tradeoffs.",
    "high_logic_lower_effect": "Logic-first rows that test whether stronger shape fidelity loses effect.",
    "off_state_logic_penalized": "Rows selected when predicted logic level in OFF states is explicitly penalized.",
    "canonical_sfxi_shared_overlap": "Canonical SFXI rows reused by multiple target views for collapse diagnosis.",
    "target_view_specific_comparison": "Shape-ceiling comparison rows not shared across target views.",
}


def build_policy_comparison_panel(
    candidates: pd.DataFrame,
    *,
    recommendation: dict[str, object],
    observed_label_ids: Iterable[str] = (),
    per_target_view: int = 3,
) -> pd.DataFrame:
    if per_target_view < 1:
        raise ValueError(f"per_target_view must be >= 1; got {per_target_view}.")
    comparison_policy_id = str(recommendation.get("comparison_policy_id") or "")
    required = (*REQUIRED_PANEL_POLICIES, comparison_policy_id)
    _assert_required_policies(candidates, required)

    observed = {str(value) for value in observed_label_ids}
    panels = [
        _role_rows(
            candidates,
            role="canonical_sfxi_high_effect",
            policy_id=CANONICAL_SFXI_POLICY_ID,
            per_target_view=per_target_view,
        ),
        _role_rows(
            candidates,
            role="comparison_shape_effect",
            policy_id=comparison_policy_id,
            per_target_view=per_target_view,
        ),
        _role_rows(
            candidates,
            role="high_logic_lower_effect",
            policy_id="lexicographic_logic_effect",
            per_target_view=per_target_view,
            sort_columns=["logic_fidelity", "effect_scaled", "id"],
            ascending=[False, True, True],
        ),
        _role_rows(
            candidates,
            role="off_state_logic_penalized",
            policy_id="off_state_logic_eta2_beta2_gamma05",
            per_target_view=per_target_view,
            sort_columns=["rank", "id"],
            ascending=[True, True],
        ),
        _shared_overlap_rows(
            candidates,
            role="canonical_sfxi_shared_overlap",
            policy_id=CANONICAL_SFXI_POLICY_ID,
        ),
        _target_view_specific_rows(
            candidates,
            role="target_view_specific_comparison",
            policy_id=comparison_policy_id,
        ),
    ]
    panel = pd.concat([frame for frame in panels if not frame.empty], ignore_index=True)
    if panel.empty:
        return _empty_panel()
    panel = panel.drop_duplicates(["panel_role", "policy_id", "selection_view_id", "id"], keep="first")
    panel["panel_role_description"] = panel["panel_role"].map(ROLE_DESCRIPTIONS)
    panel["observed_label_member"] = panel["id"].astype(str).isin(observed)
    panel["selection_posture"] = "metric_comparison_not_synthesis"
    panel["claim_boundary"] = "Predicted metric-behavior probe; not a synthesis handoff or biological result."
    panel["panel_rank"] = panel.groupby(["panel_role", "selection_view_id"], sort=False).cumcount() + 1
    return panel[
        [
            "panel_role",
            "panel_role_description",
            "selection_posture",
            "claim_boundary",
            "policy_id",
            "selection_view_id",
            "panel_rank",
            "rank",
            "id",
            "sequence",
            "score",
            "logic_fidelity",
            "effect_scaled",
            "off_state_logic_level",
            "selection_view_count",
            "observed_label_member",
            "v00",
            "v10",
            "v01",
            "v11",
        ]
    ]


def _role_rows(
    candidates: pd.DataFrame,
    *,
    role: str,
    policy_id: str,
    per_target_view: int,
    sort_columns: list[str] | None = None,
    ascending: list[bool] | None = None,
) -> pd.DataFrame:
    data = candidates[candidates["policy_id"] == policy_id].copy()
    if data.empty:
        return _empty_panel()
    sort_columns = sort_columns or ["rank", "id"]
    ascending = ascending or [True, True]
    rows = (
        data.sort_values(["selection_view_id", *sort_columns], ascending=[True, *ascending], kind="mergesort")
        .groupby("selection_view_id", sort=False)
        .head(per_target_view)
        .copy()
    )
    rows["panel_role"] = role
    return rows


def _shared_overlap_rows(candidates: pd.DataFrame, *, role: str, policy_id: str) -> pd.DataFrame:
    rows = candidates[(candidates["policy_id"] == policy_id) & (candidates["selection_view_count"] >= 2)].copy()
    if rows.empty:
        return _empty_panel()
    rows = rows.sort_values(
        ["selection_view_count", "id", "selection_view_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    rows["panel_role"] = role
    return rows


def _target_view_specific_rows(candidates: pd.DataFrame, *, role: str, policy_id: str) -> pd.DataFrame:
    rows = candidates[(candidates["policy_id"] == policy_id) & (candidates["selection_view_count"] == 1)].copy()
    if rows.empty:
        return _empty_panel()
    rows = rows.sort_values(
        ["selection_view_id", "rank", "id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    rows["panel_role"] = role
    return rows


def _assert_required_policies(candidates: pd.DataFrame, required: Iterable[str]) -> None:
    present = set(candidates["policy_id"].astype(str)) if "policy_id" in candidates.columns else set()
    missing = sorted({policy_id for policy_id in required if policy_id} - present)
    if missing:
        raise ValueError(f"Policy-comparison panel missing required policy rows: {', '.join(missing)}")


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame()
