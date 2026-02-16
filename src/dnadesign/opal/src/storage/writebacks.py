"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/writebacks.py

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
class SelectionEmit:
    ranks_competition: np.ndarray  # (n,) int
    selected_bool: np.ndarray  # (n,) bool
    diagnostics: Optional[Dict[str, Any]] = None


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
    selected_score: np.ndarray,
    selected_score_ref: str,
    y_dim: int,
    obj_diagnostics: dict[str, Any],
    sel_emit: SelectionEmit,
    selected_uncertainty: Optional[np.ndarray] = None,
    selected_uncertainty_ref: Optional[str] = None,
    selection_score: Optional[np.ndarray] = None,
    score_channels: Optional[dict[str, np.ndarray]] = None,
    uncertainty_channels: Optional[dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    n = len(ids)
    y_hat_model_arr = np.asarray(y_hat_model, dtype=float)
    if y_hat_model_arr.ndim != 2 or y_hat_model_arr.shape[0] != n:
        raise ValueError("y_hat_model must be a 2D array with one row per id.")
    if not np.all(np.isfinite(y_hat_model_arr)):
        raise ValueError("y_hat_model must be finite in run_pred events.")

    selected_score_arr = np.asarray(selected_score, dtype=float).reshape(-1)
    if selected_score_arr.size != n:
        raise ValueError("selected_score length mismatch in run_pred events.")
    if not np.all(np.isfinite(selected_score_arr)):
        raise ValueError("selected_score must be finite in run_pred events.")
    selected_score_ref_value = str(selected_score_ref).strip()
    if not selected_score_ref_value:
        raise ValueError("selected_score_ref is required in run_pred events.")

    if len(sequences) != n:
        raise ValueError("sequences length mismatch in run_pred events.")

    ranks_competition = np.asarray(sel_emit.ranks_competition).reshape(-1)
    selected_bool = np.asarray(sel_emit.selected_bool).reshape(-1)
    if ranks_competition.size != n or selected_bool.size != n:
        raise ValueError("selection emit length mismatch in run_pred events.")

    selection_score_arr = None
    if selection_score is not None:
        selection_score_arr = np.asarray(selection_score, dtype=float).reshape(-1)
        if selection_score_arr.size != n:
            raise ValueError("selection_score length mismatch in run_pred events.")
        if not np.all(np.isfinite(selection_score_arr)):
            raise ValueError("selection_score must be finite in run_pred events.")

    selected_uncertainty_arr = None
    if selected_uncertainty is not None:
        selected_uncertainty_arr = np.asarray(selected_uncertainty, dtype=float).reshape(-1)
        if selected_uncertainty_arr.size != n:
            raise ValueError("selected_uncertainty length mismatch in run_pred events.")
        if not np.all(np.isfinite(selected_uncertainty_arr)):
            raise ValueError("selected_uncertainty must be finite in run_pred events.")
        if np.any(selected_uncertainty_arr < 0.0):
            raise ValueError("selected_uncertainty must be non-negative in run_pred events.")
    selected_uncertainty_ref_value = None
    if selected_uncertainty_ref is not None:
        selected_uncertainty_ref_value = str(selected_uncertainty_ref).strip() or None
    if (selected_uncertainty_arr is None) != (selected_uncertainty_ref_value is None):
        raise ValueError(
            "selected_uncertainty and selected_uncertainty_ref must both be provided together in run_pred events."
        )

    # Row-level diagnostics subset only (avoid run-level constants/summaries here)
    ROW_DIAG_KEYS = (
        "logic_fidelity",
        "effect_raw",
        "effect_scaled",
        "clip_lo_mask",
        "clip_hi_mask",
    )

    def _flatten_row_diagnostics(diag: Dict[str, Any], n_rows: int) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for k in ROW_DIAG_KEYS:
            if k in diag:
                v = np.asarray(diag[k])
                if v.ndim == 0:
                    out[f"obj__{k}"] = [float(v)] * n_rows
                else:
                    vv = v.reshape(-1)
                    if vv.size != n_rows:
                        raise ValueError(
                            (
                                f"objective diagnostic '{k}' length mismatch in run_pred events: "
                                f"got {vv.size}, expected {n_rows}."
                            )
                        )
                    out[f"obj__{k}"] = vv.astype(float).tolist()
        return out

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

    def _row_channel_payload(channels: dict[str, np.ndarray], idx: int) -> list[dict[str, Any]]:
        if not channels:
            return []
        return [{"name": name, "value": float(arr[idx])} for name, arr in channels.items()]

    rows: Dict[str, list] = {
        "event": ["run_pred"] * n,
        "run_id": [run_id] * n,
        "as_of_round": [int(as_of_round)] * n,
        "id": [str(x) for x in ids],
        "sequence": sequences,
        "pred__y_dim": [int(y_dim)] * n,
        "pred__y_hat_model": [list(map(float, row)) for row in y_hat_model_arr],
        "pred__score_selected": list(map(float, selected_score_arr)),
        "pred__score_ref": [selected_score_ref_value] * n,
        "pred__selection_score": [
            float(selection_score_arr[i]) if selection_score_arr is not None else None for i in range(n)
        ],
        "pred__uncertainty_selected": [
            float(selected_uncertainty_arr[i]) if selected_uncertainty_arr is not None else None for i in range(n)
        ],
        "pred__uncertainty_ref": [selected_uncertainty_ref_value] * n,
        "pred__score_channels": [_row_channel_payload(score_channel_arrays, i) for i in range(n)],
        "pred__uncertainty_channels": [_row_channel_payload(uncertainty_channel_arrays, i) for i in range(n)],
        "sel__rank_competition": ranks_competition.astype(int).tolist(),
        "sel__is_selected": selected_bool.astype(bool).tolist(),
    }

    rows.update(_flatten_row_diagnostics(obj_diagnostics or {}, n))

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
    objective_name: str,
    objective_params: Dict[str, Any],
    objective_defs: list[dict[str, Any]],
    selection_name: str,
    selection_params: Dict[str, Any],
    selection_score_ref: str,
    selection_uncertainty_ref: Optional[str],
    selection_objective_mode: str,
    sel_tie_handling: str,
    stats_n_train: int,
    stats_n_scored: int,
    unc_mean_sd: Optional[np.ndarray],
    pred_rows_df: pd.DataFrame,
    artifact_paths_and_hashes: Dict[str, tuple[str, str]],
    objective_summary_stats: Dict[str, Any] | None,
) -> pd.DataFrame:
    # Convenience mirrors of denominator info (ergonomic, avoid round_ctx.json reads)
    denom_value = (
        (objective_summary_stats or {}).get("denom_used") if isinstance(objective_summary_stats, dict) else None
    )
    denom_percentile = None
    if isinstance(objective_summary_stats, dict) and objective_summary_stats.get("denom_percentile") is not None:
        denom_percentile = int(objective_summary_stats["denom_percentile"])
    elif isinstance(objective_params, dict):
        scaling_cfg = objective_params.get("scaling")
        if isinstance(scaling_cfg, dict) and scaling_cfg.get("percentile") is not None:
            denom_percentile = int(scaling_cfg["percentile"])
    if denom_percentile is not None and not (1 <= int(denom_percentile) <= 100):
        raise ValueError("objective denominator percentile must be an integer in [1, 100].")

    selection_score_ref_value = str(selection_score_ref).strip()
    if not selection_score_ref_value:
        raise ValueError("selection_score_ref is required in run_meta events.")
    selection_uncertainty_ref_value = None
    if selection_uncertainty_ref is not None:
        selection_uncertainty_ref_value = str(selection_uncertainty_ref).strip()
        if not selection_uncertainty_ref_value:
            raise ValueError("selection_uncertainty_ref must be a non-empty string when provided.")

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
            "objective__name": [objective_name],
            "objective__params": [_none_if_empty_mapping(objective_params)],
            "objective__defs_json": [json.dumps(objective_defs or [], separators=(",", ":"), ensure_ascii=True)],
            "objective__summary_stats": [_none_if_empty_mapping(objective_summary_stats)],
            # Mirrors for easy access
            "objective__denom_value": [float(denom_value) if denom_value is not None else None],
            "objective__denom_percentile": [int(denom_percentile) if denom_percentile is not None else None],
            "selection__name": [selection_name],
            "selection__params": [_none_if_empty_mapping(selection_params)],
            "selection__score_ref": [selection_score_ref_value],
            "selection__uncertainty_ref": [selection_uncertainty_ref_value],
            "selection__objective": [selection_objective_mode],
            "selection__tie_handling": [sel_tie_handling],
            "stats__n_train": [int(stats_n_train)],
            "stats__n_scored": [int(stats_n_scored)],
            "stats__unc_mean_sd_targets": [float(np.nanmean(unc_mean_sd)) if unc_mean_sd is not None else None],
            "artifacts": [_none_if_empty_mapping(artifact_paths_and_hashes)],
            "pred__preview": [pred_rows_df.head(5).to_dict(orient="records")],
            "schema__version": [LEDGER_SCHEMA_VERSION],
            "opal__version": [OPAL_VERSION],
        }
    )
