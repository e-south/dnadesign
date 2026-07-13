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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
from .selection_types import SelectionBatchEvaluation, SelectionEvaluation
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
        if req.k_override is not None:
            top_k = int(req.k_override)
        else:
            if "top_k" not in sel_params:
                raise OpalError(f"selection_views[{view_id}].selection.params.top_k is required.")
            top_k = int(sel_params["top_k"])
        if top_k <= 0:
            raise OpalError(f"selection_views[{view_id}].selection.params.top_k must be > 0.")
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
        ranks_competition = np.asarray(sel_norm["rank_competition"]).astype(int)
        selected_bool = np.asarray(sel_norm["selected_bool"]).astype(bool)
        n = len(id_order_pool)
        if ranks_competition.shape[0] != n or selected_bool.shape[0] != n:
            raise OpalError(f"Selection view {view_id!r} returned arrays that do not match {n} candidates.")

        selected_effective = int(selected_bool.sum())
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
            ranks_competition=ranks_competition,
            selected_bool=selected_bool,
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


def build_selection_batch(
    *,
    candidate_df: pd.DataFrame,
    id_order_pool: List[str],
    selections: Dict[str, SelectionEvaluation],
    deduplicate_by: Optional[str],
    expected_unique_count: Optional[int],
) -> SelectionBatchEvaluation:
    key_column = str(deduplicate_by or "id").strip()
    required = {"id", key_column}
    missing = sorted(required - set(candidate_df.columns))
    if missing:
        raise OpalError(f"selection_batch candidate data is missing column(s): {missing}")
    candidates = candidate_df.loc[:, sorted(required)].copy()
    candidates["id"] = candidates["id"].astype(str)
    if candidates["id"].duplicated().any():
        raise OpalError("selection_batch candidate ids must be unique.")
    if candidates[key_column].isna().any():
        raise OpalError(f"selection_batch deduplicate column {key_column!r} contains null values.")
    by_id = candidates.set_index("id", drop=False)

    batch: dict[str, dict[str, Any]] = {}
    for view_id, selection in selections.items():
        if len(selection.selected_bool) != len(id_order_pool):
            raise OpalError(f"Selection view {view_id!r} does not align with the candidate pool.")
        for idx in np.flatnonzero(selection.selected_bool):
            candidate_id = str(id_order_pool[int(idx)])
            if candidate_id not in by_id.index:
                raise OpalError(f"Selection view {view_id!r} references unknown candidate id {candidate_id!r}.")
            key_value = by_id.at[candidate_id, key_column]
            key = str(key_value)
            entry = batch.setdefault(
                key,
                {
                    "id": candidate_id,
                    "selection_batch_key": key,
                    "deduplicate_by": key_column,
                    "selection_view_ids": [],
                    "selection_memberships": [],
                },
            )
            if entry["id"] != candidate_id:
                raise OpalError(
                    f"selection_batch {key_column} value {key!r} maps to multiple candidate ids: "
                    f"{entry['id']!r}, {candidate_id!r}."
                )
            entry["selection_view_ids"].append(view_id)
            entry["selection_memberships"].append(
                {
                    "selection_view_id": view_id,
                    "rank": int(selection.ranks_competition[int(idx)]),
                    "score": float(selection.y_obj_scalar[int(idx)]),
                    "selection_score": float(selection.scores[int(idx)]),
                    "score_ref": selection.score_ref,
                }
            )

    rows = pd.DataFrame(list(batch.values()))
    if not rows.empty:
        rows = rows.sort_values(["selection_batch_key", "id"], kind="stable").reset_index(drop=True)
    unique_count = int(len(rows))
    if expected_unique_count is not None and unique_count != int(expected_unique_count):
        raise OpalError(
            f"selection_batch expected {int(expected_unique_count)} unique candidates, observed {unique_count}. "
            "OPAL does not fill or discard selection slots implicitly."
        )
    return SelectionBatchEvaluation(
        rows=rows,
        deduplicate_by=key_column,
        unique_count=unique_count,
        expected_unique_count=(None if expected_unique_count is None else int(expected_unique_count)),
    )
