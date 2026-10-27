"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/writebacks.py

Storage helpers for writebacks OPAL storage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import LEDGER_SCHEMA_VERSION
from .. import __version__ as OPAL_VERSION


# ---------------------------
# Small JSON helpers
# ---------------------------
def _none_if_empty_mapping(m: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return None for {}, otherwise the mapping unchanged."""
    return None if isinstance(m, dict) and len(m) == 0 else m


def _none_if_empty_seq(s: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    """Return None for empty sequences, else a list copy. Leave non-sequences unchanged."""
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return None if len(s) == 0 else list(s)
    # already JSON-like; caller decides
    return s  # type: ignore[return-value]


# ---------------------------
# Selection emit payload
# ---------------------------
@dataclass(frozen=True)
class SelectionViewEmit:
    selection_view_id: str
    objective_name: str
    selection_name: str
    score: np.ndarray
    score_ref: str
    selection_score: np.ndarray
    ranks_competition: np.ndarray
    selected_bool: np.ndarray
    top_k: int
    diagnostics: Dict[str, Any]
    uncertainty: Optional[np.ndarray] = None
    uncertainty_ref: Optional[str] = None


# ---------------------------
# Labels → canonical events
# ---------------------------
def build_label_events(
    *,
    ids: List[str],
    sequences: List[Optional[str]],
    y_obs: List[Sequence[float]],
    observed_round: int,
    src: str,
    note: Optional[str],
) -> pd.DataFrame:
    if not (len(ids) == len(y_obs) == len(sequences)):
        raise ValueError("Label events length mismatch")
    n = len(ids)
    return pd.DataFrame(
        {
            "event": ["label"] * n,
            "observed_round": [int(observed_round)] * n,
            "id": [str(i) for i in ids],
            "sequence": sequences,
            "y_obs": [list(map(float, y)) for y in y_obs],
            "src": [src] * n,
            "note": [note] * n,
        }
    )


# ---------------------------
# Run predictions (per candidate)
# ---------------------------
def build_run_pred_events(
    run_id: str,
    as_of_round: int,
    ids: list[str],
    sequences: list[str | None],
    y_hat_model: np.ndarray,
    y_dim: int,
    selection_views: Sequence[SelectionViewEmit],
    score_channels: Optional[dict[str, np.ndarray]] = None,
    uncertainty_channels: Optional[dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    n = len(ids)
    y_hat_model_arr = np.asarray(y_hat_model, dtype=float)
    if y_hat_model_arr.ndim != 2 or y_hat_model_arr.shape[0] != n:
        raise ValueError("y_hat_model must be a 2D array with one row per id.")
    if not np.all(np.isfinite(y_hat_model_arr)):
        raise ValueError("y_hat_model must be finite in run_pred events.")

    if len(sequences) != n:
        raise ValueError("sequences length mismatch in run_pred events.")

    def _prepare_channel_arrays(
        channels: Optional[dict[str, np.ndarray]],
        *,
        channel_kind: str,
    ) -> dict[str, np.ndarray]:
        if not channels:
            return {}
        prepared: dict[str, np.ndarray] = {}
        for name in sorted(channels.keys()):
            arr = np.asarray(channels[name], dtype=float).reshape(-1)
            if arr.size != n:
                raise ValueError(f"{channel_kind} channel '{name}' length mismatch in run_pred events.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{channel_kind} channel '{name}' must be finite in run_pred events.")
            prepared[str(name)] = arr
        return prepared

    score_channel_arrays = _prepare_channel_arrays(score_channels, channel_kind="score")
    uncertainty_channel_arrays = _prepare_channel_arrays(uncertainty_channels, channel_kind="uncertainty")

    prepared_views: list[dict[str, Any]] = []
    seen_view_ids: set[str] = set()
    for view in selection_views:
        view_id = str(view.selection_view_id).strip()
        if not view_id or view_id in seen_view_ids:
            raise ValueError(f"selection view ids must be unique and non-empty; observed {view_id!r}.")
        seen_view_ids.add(view_id)
        arrays = {
            "score": np.asarray(view.score, dtype=float).reshape(-1),
            "selection_score": np.asarray(view.selection_score, dtype=float).reshape(-1),
            "rank": np.asarray(view.ranks_competition, dtype=int).reshape(-1),
            "selected": np.asarray(view.selected_bool, dtype=bool).reshape(-1),
        }
        if any(arr.size != n for arr in arrays.values()):
            raise ValueError(f"selection view {view_id!r} arrays must match {n} prediction rows.")
        if not np.all(np.isfinite(arrays["score"])) or not np.all(np.isfinite(arrays["selection_score"])):
            raise ValueError(f"selection view {view_id!r} scores must be finite.")
        uncertainty = None
        if view.uncertainty is not None:
            uncertainty = np.asarray(view.uncertainty, dtype=float).reshape(-1)
            if uncertainty.size != n or not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
                raise ValueError(f"selection view {view_id!r} uncertainty must be finite, non-negative, and aligned.")
        if (uncertainty is None) != (view.uncertainty_ref is None):
            raise ValueError(f"selection view {view_id!r} uncertainty and uncertainty_ref must be paired.")
        diagnostics: dict[str, np.ndarray] = {}
        for name, value in sorted((view.diagnostics or {}).items()):
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.size != n or not np.issubdtype(arr.dtype, np.number):
                continue
            diagnostics[str(name)] = arr.astype(float).reshape(-1)
        prepared_views.append(
            {
                "view": view,
                "view_id": view_id,
                "arrays": arrays,
                "uncertainty": uncertainty,
                "diagnostics": diagnostics,
            }
        )

    def _row_channel_payload(channels: dict[str, np.ndarray], idx: int) -> list[dict[str, Any]]:
        if not channels:
            return []
        return [{"name": name, "value": float(arr[idx])} for name, arr in channels.items()]

    def _row_selection_payload(idx: int) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for prepared in prepared_views:
            view = prepared["view"]
            arrays = prepared["arrays"]
            uncertainty = prepared["uncertainty"]
            diagnostics = prepared["diagnostics"]
            payload.append(
                {
                    "selection_view_id": prepared["view_id"],
                    "objective_name": str(view.objective_name),
                    "selection_name": str(view.selection_name),
                    "score": float(arrays["score"][idx]),
                    "score_ref": str(view.score_ref),
                    "selection_score": float(arrays["selection_score"][idx]),
                    "rank_competition": int(arrays["rank"][idx]),
                    "is_selected": bool(arrays["selected"][idx]),
                    "top_k": int(view.top_k),
                    "uncertainty": (None if uncertainty is None else float(uncertainty[idx])),
                    "uncertainty_ref": view.uncertainty_ref,
                    "diagnostics": [{"name": name, "value": float(arr[idx])} for name, arr in diagnostics.items()],
                }
            )
        return payload

    rows: Dict[str, list] = {
        "event": ["run_pred"] * n,
        "run_id": [run_id] * n,
        "as_of_round": [int(as_of_round)] * n,
        "id": [str(x) for x in ids],
        "sequence": sequences,
        "pred__y_dim": [int(y_dim)] * n,
        "pred__y_hat_model": [list(map(float, row)) for row in y_hat_model_arr],
        "pred__score_channels": [_row_channel_payload(score_channel_arrays, i) for i in range(n)],
        "pred__uncertainty_channels": [_row_channel_payload(uncertainty_channel_arrays, i) for i in range(n)],
        "pred__selection_views": [_row_selection_payload(i) for i in range(n)],
    }

    return pd.DataFrame(rows)


# ---------------------------
# Run meta (single-row)
# ---------------------------
# signature: add a new argument
def build_run_meta_event(
    *,
    run_id: str,
    as_of_round: int,
    model_name: str,
    model_params: Dict[str, Any],
    y_ops: list[dict],
    x_transform_name: str,
    x_transform_params: Dict[str, Any],
    y_ingest_transform_name: str,
    y_ingest_transform_params: Dict[str, Any],
    objective_defs: list[dict[str, Any]],
    selection_view_defs: list[dict[str, Any]],
    stats_n_train: int,
    stats_n_scored: int,
    pred_rows_df: pd.DataFrame,
    artifact_paths_and_hashes: Dict[str, tuple[str, str]],
) -> pd.DataFrame:
    if not objective_defs:
        raise ValueError("objective_defs must contain at least one selection view objective.")
    if not selection_view_defs:
        raise ValueError("selection_view_defs must contain at least one selection view.")

    return pd.DataFrame(
        {
            "event": ["run_meta"],
            "run_id": [run_id],
            "as_of_round": [int(as_of_round)],
            "model__name": [model_name],
            "model__params": [_none_if_empty_mapping(model_params)],
            "training__y_ops": [_none_if_empty_seq(y_ops)],
            "x_transform__name": [x_transform_name],
            "x_transform__params": [_none_if_empty_mapping(x_transform_params)],
            "y_ingest__name": [y_ingest_transform_name],
            "y_ingest__params": [_none_if_empty_mapping(y_ingest_transform_params)],
            "objective__defs_json": [json.dumps(objective_defs or [], separators=(",", ":"), ensure_ascii=True)],
            "selection_views__defs_json": [
                json.dumps(selection_view_defs or [], separators=(",", ":"), ensure_ascii=True)
            ],
            "stats__n_train": [int(stats_n_train)],
            "stats__n_scored": [int(stats_n_scored)],
            "artifacts": [_none_if_empty_mapping(artifact_paths_and_hashes)],
            "pred__preview": [pred_rows_df.head(5).to_dict(orient="records")],
            "schema__version": [LEDGER_SCHEMA_VERSION],
            "opal__version": [OPAL_VERSION],
        }
    )
