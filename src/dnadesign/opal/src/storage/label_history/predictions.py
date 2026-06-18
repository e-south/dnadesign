"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/label_history/predictions.py

Storage helpers for predictions OPAL storage label history.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ...core.utils import OpalError
from ._coercion import _coerce_bool, _coerce_float, _coerce_int, _coerce_mapping, _deep_as_py


def append_predictions_from_arrays(
    history: Any,
    df: pd.DataFrame,
    *,
    ids: List[str],
    y_hat: np.ndarray,
    as_of_round: int,
    run_id: str,
    objective: Dict[str, Any],
    metrics_by_name: Dict[str, List[float]],
    selection_rank: np.ndarray,
    selection_top_k: np.ndarray,
    ts: str | None = None,
) -> pd.DataFrame:
    objective = _coerce_mapping(_deep_as_py(objective))
    if objective is None:
        raise OpalError("append_predictions_from_arrays requires objective mapping.")
    if not objective.get("name"):
        raise OpalError("append_predictions_from_arrays requires objective.name.")
    params_val = objective.get("params")
    if params_val is not None and not isinstance(params_val, dict):
        from collections.abc import Mapping

        if not isinstance(params_val, Mapping):
            raise OpalError("append_predictions_from_arrays requires objective.params mapping or null.")
    if params_val is not None:
        params_val = _coerce_mapping(_deep_as_py(params_val))
        if not params_val:
            params_val = None
    objective = {**objective, "params": params_val}

    lh = history.label_hist_col()
    out = df.copy()
    if lh not in out.columns:
        out[lh] = None
    if y_hat.shape[0] != len(ids):
        raise OpalError("append_predictions_from_arrays length mismatch: ids vs y_hat")
    if selection_rank.shape[0] != len(ids) or selection_top_k.shape[0] != len(ids):
        raise OpalError("append_predictions_from_arrays length mismatch: selection arrays")
    if "score" not in (metrics_by_name or {}):
        raise OpalError("append_predictions_from_arrays requires metrics_by_name['score'].")

    for key, values in (metrics_by_name or {}).items():
        if len(values) != len(ids):
            raise OpalError(f"append_predictions_from_arrays metrics length mismatch for '{key}'")

    y_hat_arr = np.asarray(y_hat, dtype=float)
    if not np.all(np.isfinite(y_hat_arr)):
        raise OpalError("append_predictions_from_arrays received non-finite y_hat values.")

    metrics_by_name = metrics_by_name or {}
    for key, values in metrics_by_name.items():
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise OpalError(f"append_predictions_from_arrays received non-finite metrics for '{key}'.")

    ranks_arr = np.asarray(selection_rank)
    selected_arr = np.asarray(selection_top_k)
    ts_val = ts or pd.Timestamp.now("UTC").isoformat()

    hist_map: Dict[str, List[Dict[str, Any]]] = {}
    for _id, hist_cell in out[["id", lh]].itertuples(index=False, name=None):
        _id_str = str(_id)
        try:
            hist_map[_id_str] = history.parse_hist_cell_strict(hist_cell)
        except OpalError as exc:
            raise OpalError(f"Malformed label history for id={_id_str}: {exc}") from exc

    for i, _id in enumerate(ids):
        _id = str(_id)
        cur = hist_map.get(_id, [])
        cur = [e for e in cur if not (e.get("kind") == "pred" and int(e.get("as_of_round", -1)) == int(as_of_round))]

        rank_val = _coerce_int(ranks_arr[i])
        if rank_val is None:
            raise OpalError(f"Prediction selection rank invalid for id={_id}.")
        top_k_val = _coerce_bool(selected_arr[i])
        if top_k_val is None:
            raise OpalError(f"Prediction selection top_k invalid for id={_id}.")

        metrics_entry: dict[str, Any] = {}
        for key, values in metrics_by_name.items():
            val = _coerce_float(values[i])
            if val is None:
                raise OpalError(f"Prediction metric '{key}' invalid for id={_id}.")
            metrics_entry[key] = val

        y_vec = y_hat_arr[i, :].tolist()
        entry = {
            "kind": "pred",
            "as_of_round": int(as_of_round),
            "run_id": str(run_id),
            "ts": ts_val,
            "y_pred": {
                "value": y_vec,
                "dtype": "vector",
                "schema": {"length": int(len(y_vec))},
            },
            "y_space": "objective",
            "objective": dict(objective or {}),
            "metrics": metrics_entry,
            "selection": {"rank": rank_val, "top_k": bool(top_k_val)},
        }
        cur.append(entry)
        hist_map[_id] = cur

    out[lh] = out["id"].astype(str).map(hist_map.get)
    return out
