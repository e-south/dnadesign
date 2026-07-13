"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/metric_behavior.py

Metric-behavior probes for SFXI policy review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..core.contracts import PolicySpec, SfxiEvidenceFrame
from .selection import select_top_rows


def build_denominator_sensitivity(
    policies: Iterable[PolicySpec],
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    scored: dict[str, dict[str, pd.DataFrame]],
    *,
    factors: tuple[float, ...],
    top_k: int,
) -> pd.DataFrame:
    if not factors:
        raise ValueError("factors must contain at least one denominator scale factor.")
    rows: list[dict[str, object]] = []
    for factor in factors:
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"denominator factors must be positive and finite; got {factor}.")
    for policy in policies:
        for evidence in sfxi_evidence:
            frame = scored[policy.id][evidence.target_view.id]
            _assert_metric_columns(frame)
            for factor in factors:
                rescored = _rescore_with_denominator_factor(policy, frame, evidence=evidence, factor=float(factor))
                top = select_top_rows(rescored, top_k=top_k)
                rows.append(
                    {
                        "policy_id": policy.id,
                        "selection_view_id": evidence.target_view.id,
                        "denominator_factor": float(factor),
                        "effective_topk": int(len(top)),
                        "unique_topk": int(top["id"].nunique()) if not top.empty else 0,
                        "median_logic_fidelity": _median(top, "logic_fidelity"),
                        "median_effect_scaled": _median(top, "effect_scaled"),
                        "median_off_state_logic_level": _median(top, "off_state_logic_level"),
                        "top_ids": ",".join(top["id"].astype(str).tolist()) if not top.empty else "",
                        "interpretation_boundary": "recomputed_from_predicted_effect_raw",
                    }
                )
    return pd.DataFrame(rows)


def _rescore_with_denominator_factor(
    policy: PolicySpec,
    frame: pd.DataFrame,
    *,
    evidence: SfxiEvidenceFrame,
    factor: float,
) -> pd.DataFrame:
    data = frame.copy()
    effect = np.clip(data["effect_raw"].astype(float).to_numpy() / (float(evidence.denom) * factor), 0.0, 1.0)
    logic = data["logic_fidelity"].astype(float).to_numpy()
    off_state_level = data["off_state_logic_level"].astype(float).to_numpy()
    if policy.kind == "multiplicative":
        score = np.power(logic, policy.beta) * np.power(effect, policy.gamma)
        eligible = np.ones(len(data), dtype=bool)
        sort_cols = ["eligible", "score", "id"]
        ascending = [False, False, True]
    elif policy.kind == "off_state_logic_penalty":
        penalty = np.power(
            np.clip(1.0 - off_state_level, 0.0, 1.0),
            policy.off_state_logic_eta,
        )
        score = np.power(logic, policy.beta) * np.power(effect, policy.gamma) * penalty
        eligible = np.ones(len(data), dtype=bool)
        sort_cols = ["eligible", "score", "id"]
        ascending = [False, False, True]
    elif policy.kind == "logic_gate":
        if policy.logic_gate is None:
            raise ValueError(f"{policy.id}: logic_gate policy requires a threshold.")
        eligible = logic >= float(policy.logic_gate)
        score = np.where(eligible, np.power(effect, policy.gamma), np.nan)
        sort_cols = ["eligible", "score", "logic_fidelity", "id"]
        ascending = [False, False, False, True]
    elif policy.kind == "lexicographic":
        eligible = np.ones(len(data), dtype=bool)
        score = logic
        sort_cols = ["eligible", "logic_fidelity", "effect_scaled", "id"]
        ascending = [False, False, False, True]
    else:
        raise ValueError(f"Unsupported policy kind: {policy.kind}")
    data["score"] = score
    data["effect_scaled"] = effect
    data["eligible"] = eligible
    return data.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _assert_metric_columns(frame: pd.DataFrame) -> None:
    missing = {"id", "effect_raw", "logic_fidelity", "off_state_logic_level"} - set(frame.columns)
    if missing:
        raise ValueError(f"denominator sensitivity missing required columns: {', '.join(sorted(missing))}")


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame[column].median())
