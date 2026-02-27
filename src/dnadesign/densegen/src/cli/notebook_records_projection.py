"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_records_projection.py

Curated records projection helpers for DenseGen notebook preview and export.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

_CURATED_RECORD_COLUMNS = [
    "id",
    "sequence",
    "source",
    "densegen__run_id",
    "densegen__plan",
    "densegen__input_name",
    "densegen__length",
    "densegen__compression_ratio",
    "densegen__gc_total",
    "densegen__gc_core",
    "densegen__required_regulators",
    "densegen__used_tf_counts",
    "densegen__sampling_library_hash",
    "densegen__sampling_library_index",
    "densegen__pad_used",
    "densegen__pad_end",
    "densegen__pad_bases",
    "densegen__pad_literal",
    "densegen__parts_detail",
]


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "as_py"):
        try:
            value = value.as_py()
        except Exception:
            return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    return value


def _coerce_list_of_dicts(raw_value: Any) -> list[dict[str, Any]]:
    value = _normalize_value(raw_value)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    items: list[dict[str, Any]] = []
    for item in value:
        item = _normalize_value(item)
        if isinstance(item, dict):
            items.append(dict(item))
    return items


def _build_parts_detail(used_tfbs_detail: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _coerce_list_of_dicts(used_tfbs_detail)]


def build_records_preview_table(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=_CURATED_RECORD_COLUMNS)

    df = records_df.copy()
    if "densegen__parts_detail" not in df.columns:
        df["densegen__parts_detail"] = [
            _build_parts_detail(row.get("densegen__used_tfbs_detail")) for _, row in df.iterrows()
        ]
    if "densegen__pad_literal" not in df.columns:
        df["densegen__pad_literal"] = None

    available_cols = [name for name in _CURATED_RECORD_COLUMNS if name in df.columns]
    projected = df.loc[:, available_cols].copy()
    return projected
