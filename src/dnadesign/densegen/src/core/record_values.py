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
        return items
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
            return list(parsed)
        raise ValueError("used_tfbs_detail JSON must decode to a list.")
    if isinstance(value, (list, np.ndarray)):
        items = list(value)
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("used_tfbs_detail list must contain dicts.")
        return items
    raise ValueError(f"used_tfbs_detail must be list[dict], got {type(value).__name__}.")
