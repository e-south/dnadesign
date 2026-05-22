from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

import numpy as np

from ...core.utils import OpalError
from ._coercion import (
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _coerce_mapping,
    _deep_as_py,
    _normalize_label_entry,
    _normalize_pred_entry,
)


def normalize_hist_cell(cell: Any) -> List[Dict[str, Any]]:
    """
    Normalize a 'label_hist' cell into the current list-of-dicts schema.

    Be permissive on container types to tolerate different Parquet round-trips.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except Exception:
            return []

    cell = _deep_as_py(cell)
    if isinstance(cell, dict):
        cell = [cell]
    elif isinstance(cell, tuple):
        cell = list(cell)
    elif not isinstance(cell, list):
        return []
    out: List[Dict[str, Any]] = []
    for e in cell:
        if isinstance(e, str):
            try:
                e = json.loads(e)
            except Exception:
                continue
        entry_map = _coerce_mapping(_deep_as_py(e))
        if entry_map is None:
            continue
        kind = entry_map.get("kind")
        if kind is None:
            normalized = _normalize_label_entry(entry_map)
            if normalized is not None:
                out.append(normalized)
            continue
        kind_str = str(kind).strip().lower()
        if kind_str == "label":
            normalized = _normalize_label_entry(entry_map)
            if normalized is not None:
                out.append(normalized)
        elif kind_str == "pred":
            normalized = _normalize_pred_entry(entry_map)
            if normalized is not None:
                out.append(normalized)
    return out


def parse_hist_cell_strict(cell: Any) -> List[Dict[str, Any]]:
    """
    Strict validation for label_hist: ensure every entry has required keys and finite values.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except Exception as e:
            raise OpalError(f"label_hist JSON parse failed: {e}")

    cell = _deep_as_py(cell)

    if isinstance(cell, dict):
        cell = [cell]
    elif isinstance(cell, tuple):
        cell = list(cell)
    elif not isinstance(cell, list):
        raise OpalError("label_hist cell must be a list or dict.")

    out: List[Dict[str, Any]] = []
    for e in cell:
        if isinstance(e, str):
            try:
                e = json.loads(e)
            except Exception as exc:
                raise OpalError(f"label_hist entry JSON parse failed: {exc}") from exc
        entry_map = _coerce_mapping(_deep_as_py(e))
        if entry_map is None:
            raise OpalError("label_hist entries must be dicts.")

        kind = entry_map.get("kind")
        if kind is None or str(kind).strip().lower() == "label":
            normalized = _normalize_label_entry(entry_map)
            if normalized is None:
                raise OpalError("label_hist label entry missing required keys.")
            out.append(normalized)
            continue

        kind_str = str(kind).strip().lower()
        if kind_str != "pred":
            raise OpalError(f"label_hist entry has unknown kind: {kind_str!r}")

        normalized = _normalize_pred_entry(entry_map)
        if normalized is None:
            raise OpalError("label_hist pred entry missing required keys.")

        y_space_val = normalized.get("y_space")
        if not isinstance(y_space_val, str) or not y_space_val.strip():
            raise OpalError("label_hist pred entry y_space must be a non-empty string.")

        objective = entry_map.get("objective")
        if not isinstance(objective, Mapping):
            raise OpalError("label_hist pred entry missing objective mapping.")
        if not objective.get("name"):
            raise OpalError("label_hist pred entry objective missing name.")
        params_val = objective.get("params")
        if params_val is not None and not isinstance(params_val, Mapping):
            raise OpalError("label_hist pred entry objective.params must be a mapping or null.")

        metrics = entry_map.get("metrics")
        if not isinstance(metrics, Mapping):
            raise OpalError("label_hist pred entry missing metrics mapping.")
        if "score" not in metrics:
            raise OpalError("label_hist pred entry metrics missing score.")
        score_val = _coerce_float(metrics.get("score"))
        if score_val is None:
            raise OpalError("label_hist pred entry metrics.score must be finite.")

        for key in ("logic_fidelity", "effect_scaled", "effect_raw"):
            if key in metrics and _coerce_float(metrics.get(key)) is None:
                raise OpalError(f"label_hist pred entry metrics.{key} must be finite.")

        selection = entry_map.get("selection")
        if not isinstance(selection, Mapping):
            raise OpalError("label_hist pred entry missing selection mapping.")
        rank_val = selection.get("rank")
        if rank_val is None or _coerce_int(rank_val) is None:
            raise OpalError("label_hist pred entry selection.rank must be an int.")
        top_k_val = _coerce_bool(selection.get("top_k"))
        if top_k_val is None:
            raise OpalError("label_hist pred entry selection.top_k must be a bool.")

        normalized["objective"] = dict(objective)
        normalized["metrics"] = dict(metrics)
        normalized["selection"] = dict(selection)
        out.append(normalized)
    return out
