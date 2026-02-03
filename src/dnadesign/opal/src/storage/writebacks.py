"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/writebacks.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
    y_obj_scalar: np.ndarray,
    y_dim: int,
    y_hat_model_sd: Optional[np.ndarray],
    y_obj_scalar_sd: Optional[np.ndarray],
    obj_diagnostics: dict[str, Any],
    sel_emit: SelectionEmit,
) -> pd.DataFrame:
    n = len(ids)
    if y_hat_model.shape[0] != n or y_obj_scalar.shape[0] != n:
        raise ValueError("Length mismatch in run_pred events")

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
                    out[f"obj__{k}"] = (
                        vv.astype(float).tolist() if vv.size == n_rows else [float(np.nanmean(vv))] * n_rows
                    )
        return out

    rows: Dict[str, list] = {
        "event": ["run_pred"] * n,
        "run_id": [run_id] * n,
        "as_of_round": [int(as_of_round)] * n,
        "id": [str(x) for x in ids],
        "sequence": sequences,
        "pred__y_dim": [int(y_dim)] * n,
        "pred__y_hat_model": [list(map(float, row)) for row in y_hat_model],
        "pred__y_obj_scalar": list(map(float, y_obj_scalar)),
        "sel__rank_competition": sel_emit.ranks_competition.astype(int).tolist(),
        "sel__is_selected": sel_emit.selected_bool.astype(bool).tolist(),
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
    selection_name: str,
    selection_params: Dict[str, Any],
    selection_score_field: str,
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
            "objective__summary_stats": [_none_if_empty_mapping(objective_summary_stats)],
            # Mirrors for easy access
            "objective__denom_value": [float(denom_value) if denom_value is not None else None],
            "objective__denom_percentile": [int((objective_params or {}).get("scaling", {}).get("percentile", 95))],
            "selection__name": [selection_name],
            "selection__params": [_none_if_empty_mapping(selection_params)],
            "selection__score_field": [selection_score_field],
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
