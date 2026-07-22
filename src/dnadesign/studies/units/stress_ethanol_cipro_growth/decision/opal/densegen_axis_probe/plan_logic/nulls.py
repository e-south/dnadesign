"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/plan_logic/nulls.py

Null-label provenance helpers for the DenseGen motif QA probe.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.constants import ACTIVE_LABEL_FAMILY_ID, NULL_ORACLE_ID
from .label_families import TF_FAMILY_COUNT_COLUMNS, TF_FAMILY_PRESENCE_COLUMNS


def null_provenance_payload(
    labels: pd.DataFrame,
    null_labels: pd.DataFrame,
    *,
    seed: int,
    label_family_id: str = ACTIVE_LABEL_FAMILY_ID,
    active_label_families: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return manifest-ready provenance for the global quality-ok permutation null."""
    _require_columns(labels, ["id", "quality_flag"])
    _require_columns(null_labels, ["id", "quality_flag"])
    ok_mask = labels["quality_flag"].astype(str).eq("ok")
    ok_ids = labels.loc[ok_mask, "id"].astype(str)
    null_ok_ids = null_labels.loc[null_labels["quality_flag"].astype(str).eq("ok"), "id"].astype(str)
    if sorted(ok_ids.tolist()) != sorted(null_ok_ids.tolist()):
        raise ValueError("null labels do not preserve the quality-ok permutation universe")
    before_balance = _balance(labels.loc[ok_mask])
    after_balance = _balance(null_labels.loc[null_labels["id"].astype(str).isin(set(ok_ids))])
    unchanged = _unchanged_assignments(labels, null_labels, ok_ids=ok_ids.tolist())
    return {
        "schema_version": "stress_ethanol_cipro_growth.densegen_null_provenance.v1",
        "null_label_source_id": NULL_ORACLE_ID,
        "label_family_id": str(label_family_id),
        "active_label_families": list(active_label_families or (label_family_id,)),
        "strategy": "global_quality_ok_permutation",
        "seed": int(seed),
        "permutation_universe": {
            "quality_flag": "ok",
            "row_count": int(len(ok_ids)),
            "id_column": "id",
        },
        "class_balance_before": before_balance,
        "class_balance_after": after_balance,
        "tf_count_sums_before": _numeric_column_sums(labels.loc[ok_mask], TF_FAMILY_COUNT_COLUMNS),
        "tf_count_sums_after": _numeric_column_sums(
            null_labels.loc[null_labels["id"].astype(str).isin(set(ok_ids))],
            TF_FAMILY_COUNT_COLUMNS,
        ),
        "tf_presence_sums_before": _numeric_column_sums(labels.loc[ok_mask], TF_FAMILY_PRESENCE_COLUMNS),
        "tf_presence_sums_after": _numeric_column_sums(
            null_labels.loc[null_labels["id"].astype(str).isin(set(ok_ids))],
            TF_FAMILY_PRESENCE_COLUMNS,
        ),
        "unchanged_assignment_count": int(unchanged),
    }


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"label frame missing null-provenance column(s): {missing}")


def _balance(frame: pd.DataFrame) -> dict[str, int]:
    if "axis_class" not in frame.columns:
        return {}
    counts = frame["axis_class"].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _numeric_column_sums(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    return {
        column: int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
        for column in columns
        if column in frame.columns
    }


def _unchanged_assignments(labels: pd.DataFrame, null_labels: pd.DataFrame, *, ok_ids: list[str]) -> int:
    if "logic4" not in labels.columns or "logic4" not in null_labels.columns:
        return 0
    original = labels.set_index(labels["id"].astype(str))["logic4"]
    permuted = null_labels.set_index(null_labels["id"].astype(str))["logic4"]
    count = 0
    for candidate_id in ok_ids:
        if _same_vector(original.get(candidate_id), permuted.get(candidate_id)):
            count += 1
    return count


def _same_vector(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    return list(left) == list(right)
