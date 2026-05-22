from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:  # pragma: no cover - optional for non-dashboard contexts
    pl = None


def _deep_as_py(x: Any) -> Any:
    try:
        if hasattr(x, "as_py"):
            return x.as_py()
        if hasattr(x, "to_pylist"):
            return x.to_pylist()
    except Exception:
        pass
    if pl is not None and isinstance(x, pl.Series):
        return [_deep_as_py(v) for v in x.to_list()]
    if isinstance(x, np.ndarray):
        return [_deep_as_py(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Series):
        return [_deep_as_py(v) for v in x.to_list()]
    if isinstance(x, dict):
        return {k: _deep_as_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_deep_as_py(v) for v in x]
    return x


def _coerce_mapping(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _coerce_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        val = str(value).strip()
    except Exception:
        return None
    return val or None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _coerce_float_list(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.tolist()


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "t", "1", "yes"}:
            return True
        if val in {"false", "f", "0", "no"}:
            return False
    return None


def _normalize_value_wrapper(value: Any, *, require_numeric: bool) -> dict | None:
    wrapper = _coerce_mapping(value)
    if wrapper is None:
        return None
    if "value" not in wrapper:
        return None
    dtype_val = _coerce_non_empty_str(wrapper.get("dtype"))
    if dtype_val is None:
        return None
    schema_val = wrapper.get("schema")
    schema_map = _coerce_mapping(schema_val) if schema_val is not None else None
    raw_value = _deep_as_py(wrapper.get("value"))
    if require_numeric:
        numeric = _coerce_float_list(raw_value)
        if numeric is None:
            return None
        raw_value = numeric
    out = {"value": raw_value, "dtype": dtype_val}
    if schema_map is not None:
        out["schema"] = schema_map
    return out


def _normalize_label_entry(entry_map: Mapping[str, Any]) -> dict | None:
    round_val = entry_map.get("observed_round", entry_map.get("r", entry_map.get("round")))
    r_int = _coerce_int(round_val)
    if r_int is None:
        return None
    y_wrap = _normalize_value_wrapper(entry_map.get("y_obs"), require_numeric=True)
    if y_wrap is None:
        return None
    return {
        "kind": "label",
        "observed_round": r_int,
        "ts": entry_map.get("ts"),
        "src": entry_map.get("src"),
        "y_obs": y_wrap,
    }


def _normalize_pred_entry(entry_map: Mapping[str, Any]) -> dict | None:
    round_val = entry_map.get("as_of_round", entry_map.get("r", entry_map.get("round")))
    r_int = _coerce_int(round_val)
    if r_int is None:
        return None
    run_id = entry_map.get("run_id")
    if run_id is None:
        return None
    y_pred_wrap = _normalize_value_wrapper(entry_map.get("y_pred"), require_numeric=False)
    if y_pred_wrap is None:
        return None
    y_space = _coerce_non_empty_str(entry_map.get("y_space"))
    if y_space is None:
        return None
    pred: dict[str, Any] = {
        "kind": "pred",
        "as_of_round": r_int,
        "run_id": str(run_id),
        "ts": entry_map.get("ts"),
        "y_pred": y_pred_wrap,
        "y_space": y_space,
    }
    objective = entry_map.get("objective")
    if isinstance(objective, Mapping):
        pred["objective"] = dict(objective)
    metrics = entry_map.get("metrics")
    if isinstance(metrics, Mapping):
        pred["metrics"] = dict(metrics)
    selection = entry_map.get("selection")
    if isinstance(selection, Mapping):
        pred["selection"] = dict(selection)
    return pred
