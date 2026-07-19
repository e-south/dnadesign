"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/three_axis_scatter_data.py

Contract validation and deterministic data sampling for three-axis scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

THREE_AXIS_SCATTER_ADAPTER = "three_axis_scatter_v1"


def resolve_three_axis_interactive_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return the interactive three-axis adapter declaration."""

    interactive = _mapping(contract.get("interactive"))
    if interactive.get("adapter") != THREE_AXIS_SCATTER_ADAPTER:
        raise ValueError(f"Interactive scatter requires adapter {THREE_AXIS_SCATTER_ADAPTER!r}.")
    required = {
        "adapter",
        "score_column",
        "score_label",
        "prediction_sample_limit",
        "sampling_method",
    }
    missing = sorted(required - set(interactive))
    if missing:
        raise ValueError(f"Three-axis scatter adapter is missing fields: {missing}.")
    if not str(interactive["score_column"]).strip() or not str(interactive["score_label"]).strip():
        raise ValueError("Three-axis scatter score column and label must be non-empty.")
    limit = interactive["prediction_sample_limit"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Three-axis scatter prediction_sample_limit must be a positive integer.")
    if interactive["sampling_method"] != "sha256_id_v1":
        raise ValueError("Three-axis scatter sampling_method must be 'sha256_id_v1'.")
    return interactive


def require_finite_three_axis_rows(rows: pd.DataFrame, *, columns: Sequence[str]) -> None:
    """Reject non-numeric or non-finite figure coordinates and scores."""

    try:
        values = rows.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Three-axis scatter coordinates and scores must be finite numeric values.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Three-axis scatter coordinates and scores must be finite numeric values.")


def sample_notebook_three_axis_rows(
    rows: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Retain complete evidence layers and deterministically thin only the background pool."""

    interactive = resolve_three_axis_interactive_contract(contract)
    view = _mapping(contract["view"])
    record_column = str(view["record_kind_column"])
    selection_column = str(view["selection_column"])
    prediction_value = str(view["prediction_value"])
    is_prediction = rows[record_column].astype(str).eq(prediction_value)
    is_selected = rows[selection_column].fillna(False).astype(bool)
    background = rows.loc[is_prediction & ~is_selected]
    evidence = rows.loc[~is_prediction | is_selected]
    limit = int(interactive["prediction_sample_limit"])
    if len(background) > limit:
        identities = background["id"].astype(str)
        order = identities.map(_stable_identity_digest).sort_values(kind="mergesort").index[:limit]
        background = background.loc[order]
    sampled = pd.concat([background, evidence], axis=0).sort_index(kind="mergesort").reset_index(drop=True)
    sampled.attrs.update(rows.attrs)
    sampled.attrs["complete_background_count"] = int((is_prediction & ~is_selected).sum())
    sampled.attrs["displayed_background_count"] = int(len(background))
    return sampled


def _stable_identity_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "THREE_AXIS_SCATTER_ADAPTER",
    "require_finite_three_axis_rows",
    "resolve_three_axis_interactive_contract",
    "sample_notebook_three_axis_rows",
]
