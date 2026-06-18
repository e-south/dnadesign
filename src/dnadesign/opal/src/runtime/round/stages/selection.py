"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/selection.py

Selection-channel resolution and selection plugin execution for an OPAL round.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ....core.round_context import RoundCtx
from ....core.selection_contracts import (
    extract_selection_plugin_params,
    resolve_selection_objective_mode,
    resolve_selection_tie_handling,
)
from ....core.utils import OpalError, now_iso
from ....registries.selection import get_selection, normalize_selection_result, validate_selection_result
from ....storage.artifacts import append_round_log_event, write_objective_meta
from ..contracts import RoundInputs
from .objectives import ObjectiveEvaluation
from .telemetry import log


@dataclass(frozen=True)
class SelectionEvaluation:
    y_obj_scalar: np.ndarray
    diag: Dict[str, Any]
    obj_summary_stats: Optional[Dict[str, Any]]
    obj_name: str
    obj_params: Dict[str, Any]
    obj_mode: str
    score_ref: str
    uncertainty_ref: Optional[str]
    sel_name: str
    sel_params: Dict[str, Any]
    tie_handling: str
    mode: str
    ranks_competition: np.ndarray
    selected_bool: np.ndarray
    selected_effective: int
    top_k: int
    obj_sha: str
    scores: np.ndarray
    uq_scalar: Optional[np.ndarray]


def resolve_channel_ref(ref: str, channels: Dict[str, np.ndarray], *, label: str) -> np.ndarray:
    key = str(ref).strip()
    if not key:
        raise OpalError(f"{label} channel reference cannot be empty.")
    if key not in channels:
        raise OpalError(f"{label} channel '{key}' not found. Available: {sorted(channels.keys())}")
    return np.asarray(channels[key], dtype=float).reshape(-1)


def select_candidates(
    *,
    inputs: RoundInputs,
    rctx: RoundCtx,
    id_order_pool: List[str],
    objectives: ObjectiveEvaluation,
) -> SelectionEvaluation:
    cfg = inputs.cfg
    req = inputs.req
    rdir = inputs.rdir
    round_index = int(req.as_of_round)
    run_id = str(rctx.get("core/run_id", default=""))

    sel_name = cfg.selection.selection.name
    sel_params = dict(cfg.selection.selection.params)
    if req.k_override is not None:
        top_k = int(req.k_override)
    else:
        if "top_k" not in sel_params:
            raise OpalError("selection.params.top_k is required (or override with --k).")
        top_k = int(sel_params.get("top_k"))
    if top_k <= 0:
        raise OpalError("selection.params.top_k must be > 0 (or override with --k).")
    tie_handling = resolve_selection_tie_handling(sel_params, error_cls=OpalError)
    sel_params["tie_handling"] = tie_handling
    mode = resolve_selection_objective_mode(sel_params, error_cls=OpalError)
    sel_params["objective_mode"] = mode
    score_ref = str(sel_params.get("score_ref", "")).strip()
    if not score_ref:
        raise OpalError("selection.params.score_ref is required and must reference an objective score channel.")
    y_obj_scalar = resolve_channel_ref(score_ref, objectives.score_channels, label="score_ref")
    if score_ref not in objectives.channel_modes:
        raise OpalError(f"score_ref channel '{score_ref}' is missing objective mode metadata.")
    selected_mode = str(objectives.channel_modes[score_ref]).strip().lower()
    if selected_mode != mode:
        raise OpalError(
            "Objective mode mismatch: selected score channel "
            f"{score_ref!r} has mode {selected_mode!r} but selection objective_mode is {mode!r}."
        )

    uncertainty_ref = sel_params.get("uncertainty_ref")
    if uncertainty_ref is not None:
        uncertainty_ref = str(uncertainty_ref).strip() or None
    if sel_name == "expected_improvement" and not uncertainty_ref:
        raise OpalError("selection.params.uncertainty_ref is required for expected_improvement.")
    sq = (
        resolve_channel_ref(uncertainty_ref, objectives.uncertainty_channels, label="uncertainty_ref")
        if uncertainty_ref
        else None
    )

    sel_fn = get_selection(sel_name, sel_params)
    sctx = rctx.for_plugin(category="selection", name=sel_name, plugin=sel_fn)
    selection_call_params = extract_selection_plugin_params(sel_params)
    try:
        raw_sel = sel_fn(
            ids=np.array(id_order_pool),
            scores=y_obj_scalar,
            top_k=top_k,
            tie_handling=tie_handling,
            objective=mode,
            ctx=sctx,
            scalar_uncertainty=sq,
            **selection_call_params,
        )
    except OpalError:
        raise
    except Exception as e:
        raise OpalError(f"Selection plugin '{sel_name}' failed: {e}") from e
    validated_sel = validate_selection_result(raw_sel, plugin_name=sel_name, expected_len=len(id_order_pool))
    sel_norm = normalize_selection_result(
        {"order_idx": validated_sel.order_idx},
        ids=np.array(id_order_pool),
        scores=validated_sel.score,
        top_k=top_k,
        tie_handling=tie_handling,
        objective=mode,
    )

    required_keys = {"rank_competition", "selected_bool"}
    missing = sorted(k for k in required_keys if k not in sel_norm)
    if missing:
        raise OpalError(
            f"normalize_selection_result is missing required key(s): {missing}. Present keys: {sorted(sel_norm.keys())}"
        )

    ranks_competition = np.asarray(sel_norm["rank_competition"]).astype(int)
    selected_bool = np.asarray(sel_norm["selected_bool"]).astype(bool)
    n = len(id_order_pool)
    if ranks_competition.shape[0] != n or selected_bool.shape[0] != n:
        raise OpalError(
            "Selection normalization shape mismatch: "
            f"expected length={n}, got rank_competition={ranks_competition.shape}, "
            f"selected_bool={selected_bool.shape}"
        )

    selected_effective = int(selected_bool.sum())
    log(
        req.verbose,
        f"[selection:{sel_name}] objective={mode} tie={tie_handling} "
        f"requested_top_k={top_k} → selected={selected_effective}",
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": round_index,
            "run_id": run_id,
            "stage": "objective_done",
            "score_ref": score_ref,
            "uncertainty_ref": uncertainty_ref,
            "objective_count": len(objectives.objective_defs),
            "score_min": float(np.nanmin(y_obj_scalar)) if y_obj_scalar.size else None,
            "score_median": (float(np.nanmedian(y_obj_scalar)) if y_obj_scalar.size else None),
            "score_max": float(np.nanmax(y_obj_scalar)) if y_obj_scalar.size else None,
        },
    )
    append_round_log_event(
        rdir / "logs" / "round.log.jsonl",
        {
            "ts": now_iso(),
            "round": round_index,
            "run_id": run_id,
            "stage": "selection_done",
            "strategy": sel_name,
            "tie_handling": tie_handling,
            "objective_mode": mode,
            "score_ref": score_ref,
            "uncertainty_ref": uncertainty_ref,
            "requested_top_k": int(top_k),
            "effective_after_ties": int(selected_effective),
        },
    )
    rctx.set_core("core/selection/top_k_requested", int(top_k))
    rctx.set_core("core/selection/top_k_effective", int(selected_effective))
    rctx.set_core("core/selection/objective_mode", str(mode))
    rctx.set_core("core/selection/tie_handling", str(tie_handling))
    rctx.set_core("core/selection/score_ref", str(score_ref))
    rctx.set_core("core/selection/uncertainty_ref", uncertainty_ref)

    selected_obj_name, _ = score_ref.split("/", 1)
    selected_obj_params = next((d["params"] for d in objectives.objective_defs if d["name"] == selected_obj_name), {})
    selected_diag = objectives.diagnostics_by_objective.get(selected_obj_name, {})
    obj_summary_stats = selected_diag.get("summary_stats") if isinstance(selected_diag, dict) else None
    obj_meta = {
        "objectives": objectives.objective_defs,
        "selection": {
            "score_ref": score_ref,
            "uncertainty_ref": uncertainty_ref,
            "objective_mode": mode,
            "tie_handling": tie_handling,
        },
    }
    obj_sha = write_objective_meta(rdir / "metadata" / "objective_meta.json", obj_meta)

    return SelectionEvaluation(
        y_obj_scalar=y_obj_scalar,
        diag=selected_diag,
        obj_summary_stats=obj_summary_stats if isinstance(obj_summary_stats, dict) else None,
        obj_name=selected_obj_name,
        obj_params=selected_obj_params,
        obj_mode=selected_mode,
        score_ref=score_ref,
        uncertainty_ref=uncertainty_ref,
        sel_name=sel_name,
        sel_params=sel_params,
        tie_handling=tie_handling,
        mode=mode,
        ranks_competition=ranks_competition,
        selected_bool=selected_bool,
        selected_effective=selected_effective,
        top_k=top_k,
        obj_sha=obj_sha,
        scores=validated_sel.score,
        uq_scalar=sq,
    )
