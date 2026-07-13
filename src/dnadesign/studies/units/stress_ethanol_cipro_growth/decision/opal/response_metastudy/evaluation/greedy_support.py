"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/greedy_support.py

Grouped evidence for, but not allocation of, greedy next-build choices.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import beta


def build_greedy_support_evidence(
    model_screen: pd.DataFrame,
    retrospective_enrichment: pd.DataFrame,
    *,
    primary_reduction_id: str,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Summarize held-out enrichment without converting it into slot counts."""

    if not 0.5 < confidence_level < 1.0:
        raise ValueError("greedy-support confidence_level must be between 0.5 and 1.0.")
    required_screen = {
        "representation_id",
        "model_id",
        "promotion_eligible",
        "all_target_view_metrics_finite",
        "weakest_required_ordering_spearman",
        "median_channel_spearman",
    }
    required_enrichment = {
        "representation_id",
        "model_id",
        "selection_view_id",
        "reader_experiment_id",
        "selection_defined",
        "selected_true_percentile",
        "beats_group_median",
    }
    _require_columns(model_screen, required_screen, context="model screen")
    _require_columns(retrospective_enrichment, required_enrichment, context="retrospective enrichment")

    eligible = model_screen.loc[
        model_screen["representation_id"].astype(str).eq(primary_reduction_id)
        & model_screen["promotion_eligible"].astype(bool)
        & model_screen["all_target_view_metrics_finite"].astype(bool)
    ]
    if eligible.empty:
        raise ValueError(f"no eligible model screen exists for primary reduction {primary_reduction_id!r}.")
    best = eligible.sort_values(
        ["weakest_required_ordering_spearman", "median_channel_spearman", "model_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    evidence = retrospective_enrichment.loc[
        retrospective_enrichment["representation_id"].astype(str).eq(primary_reduction_id)
        & retrospective_enrichment["model_id"].astype(str).eq(str(best["model_id"]))
        & retrospective_enrichment["selection_defined"].astype(bool)
    ].copy()
    if evidence.empty:
        raise ValueError("best primary model has no defined grouped enrichment rows.")

    records: list[dict[str, object]] = []
    for selection_view_id, rows in evidence.groupby("selection_view_id", sort=True):
        successes = int(rows["beats_group_median"].astype(bool).sum())
        groups = int(len(rows))
        fraction = successes / groups
        lower, upper = _clopper_pearson(successes, groups, confidence_level=confidence_level)
        records.append(
            {
                "selection_view_id": str(selection_view_id),
                "representation_id": primary_reduction_id,
                "model_id": str(best["model_id"]),
                "held_out_group_count": groups,
                "groups_beating_median": successes,
                "fraction_beating_group_median": fraction,
                "fraction_ci_low": lower,
                "fraction_ci_high": upper,
                "confidence_method": "clopper_pearson_exact",
                "confidence_level": confidence_level,
                "median_selected_true_percentile": float(rows["selected_true_percentile"].median()),
                "evidence_posture": _posture(fraction),
                "allocation_boundary": "descriptive_only_no_slot_assignment",
            }
        )
    return pd.DataFrame.from_records(records).sort_values("selection_view_id", kind="mergesort").reset_index(drop=True)


def _clopper_pearson(successes: int, trials: int, *, confidence_level: float) -> tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("exact interval requires 0 <= successes <= trials and trials > 0.")
    alpha = 1.0 - confidence_level
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def _posture(fraction: float) -> str:
    if fraction > 0.5:
        return "descriptive_above_half"
    if fraction == 0.5:
        return "descriptive_equal_half"
    return "descriptive_below_half"


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


__all__ = ["build_greedy_support_evidence"]
