"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/record_values.py

Record value coercion helpers for DenseGen tables.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

_LEGACY_USED_TFBS_KEY_ALIASES = {
    "tf": "regulator",
    "tfbs": "sequence",
    "stage_a_tfbs_core": "core_sequence",
    "stage_a_best_hit_score": "score_best_hit_raw",
    "stage_a_rank_within_regulator": "rank_among_mined_positive",
    "stage_a_selection_rank": "rank_among_selected",
    "stage_a_tier": "tier",
    "stage_a_fimo_start": "matched_start",
    "stage_a_fimo_stop": "matched_stop",
    "stage_a_fimo_strand": "matched_strand",
    "stage_a_selection_score_norm": "score_relative_to_theoretical_max",
}


def as_py_value(value):
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def coerce_list(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    value = as_py_value(value)
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                parsed = json.loads(text)
            except Exception:
                return []
            if isinstance(parsed, list):
                return list(parsed)
            return []
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def _is_nullish(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_used_tfbs_entry(item: dict, *, entry_index: int) -> dict:
    normalized = {str(key): as_py_value(value) for key, value in dict(item).items()}
    for legacy_key, canonical_key in _LEGACY_USED_TFBS_KEY_ALIASES.items():
        if canonical_key not in normalized or _is_nullish(normalized.get(canonical_key)):
            legacy_value = normalized.get(legacy_key)
            if not _is_nullish(legacy_value):
                normalized[canonical_key] = legacy_value

    if _is_nullish(normalized.get("part_index")):
        part_kind = str(normalized.get("part_kind") or "tfbs").strip().lower()
        if part_kind == "tfbs":
            normalized["part_index"] = int(entry_index)
    return normalized


def normalize_used_tfbs_entries(items: list[dict]) -> list[dict]:
    return [_normalize_used_tfbs_entry(item, entry_index=index) for index, item in enumerate(items)]


def coerce_list_of_dicts(value) -> list[dict]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    value = as_py_value(value)
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                value = json.loads(text)
            except Exception as exc:
                raise ValueError(f"Failed to parse JSON list field: {text[:80]}") from exc
    if isinstance(value, (list, tuple, np.ndarray)):
        items = []
        for item in list(value):
            item = as_py_value(item)
            if not isinstance(item, dict):
                raise ValueError("Expected list of dicts; found non-dict entries.")
            items.append(item)
        return normalize_used_tfbs_entries(items)
    raise ValueError(f"Expected list of dicts; got {type(value).__name__}.")


def require_list(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError("Expected list data or JSON-encoded list.") from exc
        if isinstance(parsed, list):
            return list(parsed)
        raise ValueError("Expected JSON list data.")
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    raise ValueError(f"Expected list data, got {type(value).__name__}.")


def require_list_of_dicts(value) -> list[dict]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError("used_tfbs_detail must be a list of dicts or JSON.") from exc
        if isinstance(parsed, list):
            if any(not isinstance(item, dict) for item in parsed):
                raise ValueError("used_tfbs_detail JSON list must contain dicts.")
            return normalize_used_tfbs_entries(list(parsed))
        raise ValueError("used_tfbs_detail JSON must decode to a list.")
    if isinstance(value, (list, np.ndarray)):
        items = list(value)
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("used_tfbs_detail list must contain dicts.")
        return normalize_used_tfbs_entries(items)
    raise ValueError(f"used_tfbs_detail must be list[dict], got {type(value).__name__}.")
