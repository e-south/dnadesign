"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/selection.py

Selection-channel resolution and plugin execution for one OPAL round.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

import numpy as np

from ....core.round_context import RoundCtx
from ....core.selection_contracts import (
    extract_selection_plugin_params,
    require_exact_selection_count,
    resolve_selection_objective_mode,
    resolve_selection_tie_handling,
    resolve_selection_top_k,
)
from ....core.utils import OpalError, now_iso
from ....registries.selection import get_selection, normalize_selection_result, validate_selection_result
from ....storage.artifacts import append_round_log_event, write_objective_meta
from ..contracts import RoundInputs
from .objectives import ObjectiveEvaluation
from .selection_types import SelectionEvaluation
from .telemetry import log


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
) -> Dict[str, SelectionEvaluation]:
    cfg = inputs.cfg
    req = inputs.req
    rdir = inputs.rdir
    round_index = int(req.as_of_round)
    run_id = str(rctx.get("core/run_id", default=""))

    results: Dict[str, SelectionEvaluation] = {}
    selection_defs: list[dict[str, Any]] = []
    for view in cfg.selection_views:
        view_id = view.id
        sel_name = view.selection.name
        sel_params = dict(view.selection.params)
        top_k = resolve_selection_top_k(sel_params, view_id=view_id, override=req.k_override, error_cls=OpalError)
        tie_handling = resolve_selection_tie_handling(sel_params, error_cls=OpalError)
        sel_params["tie_handling"] = tie_handling
        mode = resolve_selection_objective_mode(sel_params, error_cls=OpalError)
        sel_params["objective_mode"] = mode

        score_channel = str(sel_params.get("score_ref", "")).strip()
        if not score_channel:
            raise OpalError(f"selection_views[{view_id}].selection.params.score_ref is required.")
        score_ref = f"{view_id}/{score_channel}"
        y_obj_scalar = resolve_channel_ref(score_ref, objectives.score_channels, label="score_ref")
        selected_mode = str(objectives.channel_modes[score_ref]).strip().lower()
        if selected_mode != mode:
            raise OpalError(
                f"Selection view {view_id!r} mode mismatch: channel {score_ref!r} is "
                f"{selected_mode!r}, selection declares {mode!r}."
            )

        uncertainty_channel = sel_params.get("uncertainty_ref")
        if uncertainty_channel is not None:
            uncertainty_channel = str(uncertainty_channel).strip() or None
        if sel_name == "expected_improvement" and not uncertainty_channel:
            raise OpalError(f"selection_views[{view_id}] requires uncertainty_ref for expected_improvement.")
        uncertainty_ref = f"{view_id}/{uncertainty_channel}" if uncertainty_channel else None
        sq = (
            resolve_channel_ref(uncertainty_ref, objectives.uncertainty_channels, label="uncertainty_ref")
            if uncertainty_ref
            else None
        )

        sel_fn = get_selection(sel_name, sel_params)
        sctx = rctx.for_plugin(category="selection", name=view_id, plugin=sel_fn)
        try:
            raw_sel = sel_fn(
                ids=np.array(id_order_pool),
                scores=y_obj_scalar,
                top_k=top_k,
                tie_handling=tie_handling,
                objective=mode,
                ctx=sctx,
                scalar_uncertainty=sq,
                **extract_selection_plugin_params(sel_params),
            )
        except OpalError:
            raise
        except Exception as e:
            raise OpalError(f"Selection view {view_id!r} plugin '{sel_name}' failed: {e}") from e
        validated_sel = validate_selection_result(raw_sel, plugin_name=sel_name, expected_len=len(id_order_pool))
        sel_norm = normalize_selection_result(
            {"order_idx": validated_sel.order_idx},
            ids=np.array(id_order_pool),
            scores=validated_sel.score,
            top_k=top_k,
            tie_handling=tie_handling,
            objective=mode,
        )
        order_idx = np.asarray(sel_norm["order_idx"]).astype(int)
        ranks_ordinal = np.empty(len(id_order_pool), dtype=int)
        ranks_ordinal[order_idx] = np.arange(1, len(id_order_pool) + 1, dtype=int)
        ranks_competition = np.asarray(sel_norm["rank_competition"]).astype(int)
        selected_bool = np.asarray(sel_norm["selected_bool"]).astype(bool)
        n = len(id_order_pool)
        if ranks_competition.shape[0] != n or selected_bool.shape[0] != n:
            raise OpalError(f"Selection view {view_id!r} returned arrays that do not match {n} candidates.")
        selected_effective = int(selected_bool.sum())
        require_exact_selection_count(
            sel_params,
            view_id=view_id,
            top_k=top_k,
            selected_count=selected_effective,
            tie_handling=tie_handling,
            error_cls=OpalError,
        )
        selected_diag = objectives.diagnostics_by_objective.get(view_id, {})
        obj_summary_stats = selected_diag.get("summary_stats") if isinstance(selected_diag, dict) else None
        result = SelectionEvaluation(
            selection_view_id=view_id,
            y_obj_scalar=y_obj_scalar,
            diag=selected_diag,
            obj_summary_stats=obj_summary_stats if isinstance(obj_summary_stats, dict) else None,
            obj_name=view.objective.name,
            obj_params=dict(view.objective.params),
            obj_mode=selected_mode,
            score_ref=score_ref,
            uncertainty_ref=uncertainty_ref,
            sel_name=sel_name,
            sel_params=sel_params,
            tie_handling=tie_handling,
            mode=mode,
            order_idx=order_idx,
            ranks_ordinal=ranks_ordinal,
            ranks_competition=ranks_competition,
            preferred_bool=selected_bool.copy(),
            selected_bool=selected_bool,
            allocation_slots=np.zeros(n, dtype=int),
            selected_effective=selected_effective,
            top_k=top_k,
            obj_sha="",
            scores=validated_sel.score,
            uq_scalar=sq,
        )
        results[view_id] = result
        selection_defs.append(
            {
                "selection_view_id": view_id,
                "selection_name": sel_name,
                "score_ref": score_ref,
                "uncertainty_ref": uncertainty_ref,
                "objective_mode": mode,
                "tie_handling": tie_handling,
                "top_k": top_k,
            }
        )
        log(
            req.verbose,
            f"[selection:{view_id}:{sel_name}] objective={mode} tie={tie_handling} "
            f"requested_top_k={top_k} selected={selected_effective}",
        )
        append_round_log_event(
            rdir / "logs" / "round.log.jsonl",
            {
                "ts": now_iso(),
                "round": round_index,
                "run_id": run_id,
                "stage": "selection_done",
                **selection_defs[-1],
                "effective_after_ties": selected_effective,
            },
        )
        prefix = f"core/selection_views/{view_id}"
        rctx.set_core(f"{prefix}/top_k_requested", top_k)
        rctx.set_core(f"{prefix}/top_k_effective", selected_effective)
        rctx.set_core(f"{prefix}/score_ref", score_ref)

    obj_meta = {"objectives": objectives.objective_defs, "selection_views": selection_defs}
    obj_sha = write_objective_meta(rdir / "metadata" / "objective_meta.json", obj_meta)
    return {view_id: replace(result, obj_sha=obj_sha) for view_id, result in results.items()}
