"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/label_history/dashboard.py

Storage helpers for dashboard OPAL storage label history.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from ._coercion import _coerce_mapping, _deep_as_py
from .parsing import normalize_hist_cell, parse_hist_cell_strict


def _record_parse_error(
    *,
    errors: list[dict],
    row_id: str,
    message: str,
    sample: Any | None = None,
) -> None:
    if len(errors) >= 5:
        return
    errors.append(
        {
            "id": row_id,
            "error": message,
            "sample": (repr(sample)[:240] if sample is not None else None),
        }
    )


def parse_label_hist_cell_for_dashboard(
    cell: Any,
    *,
    row_id: str,
    y_col_name: str,
    errors: list[dict],
) -> List[Dict[str, Any]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    entries = normalize_hist_cell(cell)
    if not entries:
        try:
            entries = parse_hist_cell_strict(cell)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist parse failed: {exc}",
                sample=cell,
            )
            return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") != "label":
            continue
        r_val = entry.get("observed_round")
        if r_val is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="label_hist entry missing observed_round",
                sample=entry,
            )
            continue
        try:
            r_int = int(r_val)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist entry round is not int: {exc}",
                sample=entry,
            )
            continue
        y_wrap = _coerce_mapping(entry.get("y_obs"))
        if y_wrap is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="label_hist entry missing y_obs wrapper",
                sample=entry,
            )
            continue
        if "value" not in y_wrap:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="label_hist entry y_obs missing value",
                sample=y_wrap,
            )
            continue
        y_val = _deep_as_py(y_wrap.get("value"))
        try:
            y_list = [float(v) for v in np.asarray(y_val, dtype=float).ravel().tolist()]
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist entry 'y' not numeric: {exc}",
                sample=y_val,
            )
            continue
        out.append(
            {
                "observed_round": r_int,
                "label_src": entry.get("src"),
                "label_ts": entry.get("ts"),
                y_col_name: y_list,
            }
        )
    return out


def parse_pred_hist_cell_for_dashboard(
    cell: Any,
    *,
    row_id: str,
    errors: list[dict],
) -> List[Dict[str, Any]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    entries = normalize_hist_cell(cell)
    if not entries:
        try:
            entries = parse_hist_cell_strict(cell)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"label_hist parse failed: {exc}",
                sample=cell,
            )
            return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("kind") != "pred":
            continue
        as_of_round = entry.get("as_of_round")
        run_id = entry.get("run_id")
        if as_of_round is None or run_id is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing as_of_round or run_id",
                sample=entry,
            )
            continue
        try:
            round_int = int(as_of_round)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"pred entry as_of_round not int: {exc}",
                sample=entry,
            )
            continue
        y_wrap = _coerce_mapping(entry.get("y_pred"))
        if y_wrap is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing y_pred wrapper",
                sample=entry,
            )
            continue
        if "value" not in y_wrap:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry y_pred missing value",
                sample=y_wrap,
            )
            continue

        pred_value = _deep_as_py(y_wrap.get("value"))
        pred_dtype = y_wrap.get("dtype")
        try:
            pred_y_hat = [float(v) for v in np.asarray(pred_value, dtype=float).ravel().tolist()]
        except Exception:
            pred_y_hat = None
        try:
            pred_value_json = json.dumps(pred_value, ensure_ascii=True)
        except Exception:
            pred_value_json = repr(pred_value)

        metrics = _coerce_mapping(entry.get("metrics") or {})
        selection = _coerce_mapping(entry.get("selection") or {})
        score_val = metrics.get("score") if metrics is not None else None
        if score_val is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing metrics.score",
                sample=entry,
            )
            continue
        try:
            score_val = float(score_val)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"pred entry metrics.score not float: {exc}",
                sample=entry,
            )
            continue

        rank_val = selection.get("rank") if selection is not None else None
        top_k_val = selection.get("top_k") if selection is not None else None
        if rank_val is None or top_k_val is None:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message="pred entry missing selection.rank/top_k",
                sample=entry,
            )
            continue
        try:
            rank_val = int(rank_val)
        except Exception as exc:
            _record_parse_error(
                errors=errors,
                row_id=row_id,
                message=f"pred entry selection.rank not int: {exc}",
                sample=entry,
            )
            continue
        top_k_val = bool(top_k_val)

        objective = _coerce_mapping(entry.get("objective") or {})
        out.append(
            {
                "as_of_round": round_int,
                "run_id": str(run_id),
                "pred_ts": entry.get("ts"),
                "pred_y_hat": pred_y_hat,
                "pred_y_value": pred_value_json,
                "pred_y_dtype": str(pred_dtype) if pred_dtype is not None else None,
                "pred_score": score_val,
                "pred_logic_fidelity": metrics.get("logic_fidelity") if metrics else None,
                "pred_effect_scaled": metrics.get("effect_scaled") if metrics else None,
                "pred_effect_raw": metrics.get("effect_raw") if metrics else None,
                "pred_rank": rank_val,
                "pred_top_k": top_k_val,
                "pred_objective_name": objective.get("name") if objective else None,
                "pred_objective_params": objective.get("params") if objective else None,
            }
        )
    return out
