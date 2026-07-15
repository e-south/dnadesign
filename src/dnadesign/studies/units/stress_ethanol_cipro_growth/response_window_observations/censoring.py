"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/censoring.py

Validate and summarize Reader component-bound provenance before label publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import VALUE_COLUMNS

BOUND_KINDS = frozenset({"exact", "lower", "upper", "indeterminate"})


class ResponseWindowCensoringError(ValueError):
    """Raised when component-bound provenance is missing or inconsistent."""


def validated_censor_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Require complete Reader provenance and return normalized component columns."""

    required = {
        f"{component}_{suffix}"
        for component in VALUE_COLUMNS
        for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
    }
    if missing := sorted(required - set(frame.columns)):
        raise ResponseWindowCensoringError(f"response measurements are missing censor provenance: {missing}")
    result = frame.copy()
    for component in VALUE_COLUMNS:
        policy_column = f"{component}_has_policy_clipping"
        overflow_column = f"{component}_has_instrument_overflow"
        bound_column = f"{component}_bound_kind"
        policy = _strict_booleans(result[policy_column], field=policy_column)
        overflow = _strict_booleans(result[overflow_column], field=overflow_column)
        bounds = result[bound_column].astype(str)
        if unknown := sorted(set(bounds) - BOUND_KINDS):
            raise ResponseWindowCensoringError(
                f"response measurement field {bound_column!r} contains unsupported values: {unknown}"
            )
        affected = policy | overflow
        if not affected.eq(bounds.ne("exact")).all():
            raise ResponseWindowCensoringError(
                f"response measurement provenance for {component!r} disagrees with its bound kind."
            )
        result[policy_column] = policy
        result[overflow_column] = overflow
        result[bound_column] = bounds
    return result


def bounded_component_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one observable row per non-exact primary contribution component."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "design_id",
                "reader_experiment_id",
                "component",
                "bound_kind",
                "has_policy_clipping",
                "has_instrument_overflow",
                "included_in_label",
            ]
        )
    normalized = validated_censor_provenance(frame)
    identity = [
        column
        for column in ("candidate_id", "design_id", "reader_experiment_id", "included_in_label")
        if column in normalized.columns
    ]
    rows: list[pd.DataFrame] = []
    for component in VALUE_COLUMNS:
        bound_column = f"{component}_bound_kind"
        selected = normalized.loc[normalized[bound_column].ne("exact"), identity].copy()
        if selected.empty:
            continue
        selected["component"] = component
        selected["bound_kind"] = normalized.loc[selected.index, bound_column]
        selected["has_policy_clipping"] = normalized.loc[selected.index, f"{component}_has_policy_clipping"]
        selected["has_instrument_overflow"] = normalized.loc[selected.index, f"{component}_has_instrument_overflow"]
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*identity, "component"])


def bounded_label_blockers(contributions: pd.DataFrame) -> tuple[str, ...]:
    """Block exact-label publication for any included candidate with bounded inputs."""

    bounded = bounded_component_rows(contributions)
    if bounded.empty:
        return ()
    if "included_in_label" in bounded.columns:
        bounded = bounded.loc[bounded["included_in_label"].astype(bool)]
    blockers: list[str] = []
    for candidate_id, rows in bounded.groupby("candidate_id", sort=True):
        components = ", ".join(sorted(rows["component"].astype(str).unique()))
        blockers.append(
            f"{candidate_id}: primary components [{components}] are bounded; "
            "exact-label publication requires an explicit censor-aware policy"
        )
    return tuple(blockers)


def bounded_primary_summary(contributions: pd.DataFrame) -> dict[str, int]:
    """Return compact preview counts without interpreting censored values."""

    bounded = bounded_component_rows(contributions)
    if bounded.empty:
        return {
            "bounded_primary_candidate_count": 0,
            "bounded_primary_contribution_count": 0,
            "bounded_primary_component_count": 0,
            "bounded_primary_label_candidate_count": 0,
        }
    contribution_columns = ["candidate_id", "design_id", "reader_experiment_id"]
    included = (
        bounded.loc[bounded["included_in_label"].astype(bool)] if "included_in_label" in bounded.columns else bounded
    )
    return {
        "bounded_primary_candidate_count": int(bounded["candidate_id"].nunique()),
        "bounded_primary_contribution_count": int(len(bounded[contribution_columns].drop_duplicates())),
        "bounded_primary_component_count": len(bounded),
        "bounded_primary_label_candidate_count": int(included["candidate_id"].nunique()),
    }


def _strict_booleans(values: pd.Series, *, field: str) -> pd.Series:
    if values.isna().any() or not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ResponseWindowCensoringError(f"response measurement field {field!r} must contain booleans.")
    return values.astype(bool)


__all__ = [
    "BOUND_KINDS",
    "ResponseWindowCensoringError",
    "bounded_component_rows",
    "bounded_label_blockers",
    "bounded_primary_summary",
    "validated_censor_provenance",
]
