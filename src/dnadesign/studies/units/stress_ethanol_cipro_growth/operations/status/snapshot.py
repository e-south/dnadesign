"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/snapshot.py

Study-owned snapshot enrichment and summary assembly for the.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .infer_runtime import (
    StressEthanolCiproGrowthInferRuntimeDependencies,
    StressEthanolCiproGrowthInferRuntimeResolvedContext,
    resolve_stress_ethanol_cipro_growth_infer_runtime_context,
)
from .record_normalizer import StressEthanolCiproGrowthResolvedContext


def _inspect_no_latentdna_readiness(**_: object) -> dict[str, object] | None:
    return {
        "configured": False,
        "state": "not_configured",
        "doc": "src/dnadesign/latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md",
        "binding_ref": None,
        "workspace_ref": None,
        "snapshot_ref": None,
        "workspace_id": None,
        "source_datasets": {},
        "decision_deliverables": [],
        "ok_deliverables": [],
        "pending_deliverables": [],
        "export_ids": [],
        "exports_ok": True,
        "browser_default_geometry_ids": [],
        "browser_preferred_hues": [],
        "supported_model_families": [],
        "default_model_family": None,
        "required_wildtype_references": [],
        "last_updated_at": None,
    }


def _inspect_no_additional_downstream_surfaces(**_: object) -> dict[str, dict[str, object]]:
    return {
        "cluster": {
            "configured": False,
            "state": "planned",
            "doc": "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md",
            "surface_ref": None,
            "results_root": None,
        },
        "opal": {
            "configured": False,
            "state": "not_configured",
            "doc": "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md",
            "surface_ref": None,
            "config": None,
        },
    }


def _inspect_no_exploratory_analysis(**_: object) -> dict[str, dict[str, object]]:
    return {}


def _inspect_no_sequence_view_contracts(**_: object) -> dict[str, object] | None:
    return None


def _inspect_no_infer_feature_completion(**_: object) -> dict[str, object] | None:
    return None


@dataclass(frozen=True)
class StressEthanolCiproGrowthStatusDependencies:
    infer_runtime: StressEthanolCiproGrowthInferRuntimeDependencies
    phase_matches_infer_model_family: Callable[..., bool]
    inspect_semantic_completeness: Callable[..., dict[str, object] | None]
    inspect_sequence_view_contracts: Callable[..., dict[str, object] | None] = _inspect_no_sequence_view_contracts
    inspect_infer_feature_completion: Callable[..., dict[str, object] | None] = _inspect_no_infer_feature_completion
    inspect_latentdna_readiness: Callable[..., dict[str, object] | None] = _inspect_no_latentdna_readiness
    inspect_additional_downstream_surfaces: Callable[..., dict[str, dict[str, object]]] = (
        _inspect_no_additional_downstream_surfaces
    )
    inspect_exploratory_analysis: Callable[..., dict[str, dict[str, object]]] = _inspect_no_exploratory_analysis


@dataclass(frozen=True)
class StressEthanolCiproGrowthStatusResolvedContext:
    infer_runtime: StressEthanolCiproGrowthInferRuntimeResolvedContext


def resolve_stress_ethanol_cipro_growth_status_context(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    status_kind: str,
    dependencies: StressEthanolCiproGrowthStatusDependencies,
) -> StressEthanolCiproGrowthStatusResolvedContext:
    infer_runtime = resolve_stress_ethanol_cipro_growth_infer_runtime_context(
        study_context=study_context,
        status_kind=status_kind,
        dependencies=dependencies.infer_runtime,
    )
    return StressEthanolCiproGrowthStatusResolvedContext(infer_runtime=infer_runtime)


def build_stress_ethanol_cipro_growth_status(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    status_context: StressEthanolCiproGrowthStatusResolvedContext,
    dependencies: StressEthanolCiproGrowthStatusDependencies,
    summary_scope: str,
) -> tuple[str, str, dict[str, object]]:
    handoff_readiness_state = _build_handoff_readiness_state(study_context=study_context)
    source_growth_state = _build_source_growth_state(
        study_context=study_context,
        handoff_readiness_state=handoff_readiness_state,
    )
    planned_outputs_state = _build_planned_outputs_state(study_context=study_context)
    semantic_completeness_state = dependencies.inspect_semantic_completeness(study_context=study_context)
    sequence_view_contract_state = dependencies.inspect_sequence_view_contracts(study_context=study_context)
    infer_feature_completion_state = dependencies.inspect_infer_feature_completion(study_context=study_context)
    latentdna_state = dependencies.inspect_latentdna_readiness(study_context=study_context)
    additional_downstream_surfaces = dependencies.inspect_additional_downstream_surfaces(study_context=study_context)
    opal_run_receipt = _attention_opal_run_receipt(additional_downstream_surfaces)
    exploratory_analysis = dependencies.inspect_exploratory_analysis(
        study_context=study_context,
        latentdna_state=latentdna_state,
        downstream_surfaces=additional_downstream_surfaces,
    )
    evidence = dict(study_context.evidence)
    evidence.update(
        {
            "summary_scope": summary_scope,
            "preferred_infer_model_family": status_context.infer_runtime.preferred_model_family,
            "supported_infer_model_families": list(status_context.infer_runtime.supported_model_families),
            "source_growth_state": source_growth_state,
            "handoff_readiness_state": handoff_readiness_state,
            "planned_outputs_state": planned_outputs_state,
            "semantic_completeness_state": semantic_completeness_state,
            "sequence_view_contract_state": sequence_view_contract_state,
            "infer_feature_completion_state": infer_feature_completion_state,
            "infer_runtime_models": [
                summary.as_dict() for summary in status_context.infer_runtime.runtime_model_summaries
            ],
            "infer_notify_profiles": {
                label: str(path) for label, path in status_context.infer_runtime.infer_notify_profile_paths.items()
            },
            "infer_notify_profile_errors": dict(status_context.infer_runtime.infer_notify_profile_errors),
        }
    )
    evidence["latentdna"] = latentdna_state
    evidence.update(additional_downstream_surfaces)
    evidence["downstream_surfaces"] = {
        "latentdna": latentdna_state,
        **additional_downstream_surfaces,
    }
    evidence["analysis_surfaces"] = exploratory_analysis

    summary_parts = [f"{study_context.resolved_study_dir.name}: phase {study_context.current_phase or 'unknown'}"]
    if status_context.infer_runtime.preferred_model_family is not None:
        summary_parts.append(f"preferred infer {status_context.infer_runtime.preferred_model_family}")
    if source_growth_state is not None and bool(source_growth_state.get("drives_top_level_attention")):
        summary_parts.append(str(source_growth_state["summary"]))
    if handoff_readiness_state is not None:
        summary_parts.append(str(handoff_readiness_state["summary"]))
    if source_growth_state is not None and not bool(source_growth_state.get("drives_top_level_attention")):
        summary_parts.append(str(source_growth_state["summary"]))
    if semantic_completeness_state is not None:
        summary_parts.append(str(semantic_completeness_state["summary"]))
    if sequence_view_contract_state is not None:
        summary_parts.append(str(sequence_view_contract_state["summary"]))
    if infer_feature_completion_state is not None:
        summary_parts.append(str(infer_feature_completion_state["summary"]))
    if latentdna_state is not None and _latentdna_readiness_drives_attention(latentdna_state):
        summary_text = str(latentdna_state.get("summary") or "").strip()
        if summary_text:
            summary_parts.append(summary_text)
    if opal_run_receipt is not None:
        summary_text = str(opal_run_receipt.get("summary") or "").strip()
        if summary_text:
            summary_parts.append(summary_text)
    if planned_outputs_state is not None and bool(planned_outputs_state.get("include_in_summary")):
        summary_parts.append(str(planned_outputs_state["summary"]))
    if study_context.next_ready_phase is not None:
        summary_parts.append(f"next ready {study_context.next_ready_phase['id']}")
    elif study_context.next_in_progress_phase is not None:
        summary_parts.append(f"next in_progress {study_context.next_in_progress_phase['id']}")
    elif study_context.next_planned_phase is not None:
        summary_parts.append(f"next planned {study_context.next_planned_phase['id']}")

    attention_reasons: list[str] = []
    if study_context.current_phase is not None and not study_context.current_phase_is_known:
        attention_reasons.append("current_phase does not match any declared phase id")
    if study_context.present_but_planned:
        attention_reasons.append("datasets.yaml is stale for newly materialized outputs")
    if source_growth_state is not None and bool(source_growth_state.get("drives_top_level_attention")):
        attention_reasons.append("DenseGen source gate is still active")
    if handoff_readiness_state is not None and bool(handoff_readiness_state.get("drives_top_level_attention")):
        attention_reasons.append("shared handoff outputs are pending or stale")
    if semantic_completeness_state is not None and bool(semantic_completeness_state.get("drives_top_level_attention")):
        attention_reasons.append("shared handoff metadata is semantically incomplete")
    if sequence_view_contract_state is not None and bool(
        sequence_view_contract_state.get("drives_top_level_attention")
    ):
        attention_reasons.append("sequence-view product contracts are incomplete")
    if infer_feature_completion_state is not None and bool(
        infer_feature_completion_state.get("drives_top_level_attention")
    ):
        attention_reasons.append("Infer feature completion is incomplete")
    if latentdna_state is not None and _latentdna_readiness_drives_attention(latentdna_state):
        attention_reasons.append("LatentDNA readiness is not ok")
    if opal_run_receipt is not None:
        attention_reasons.append("OPAL round-0 run receipt integrity is not ok")
    if planned_outputs_state is not None and bool(planned_outputs_state.get("drives_top_level_attention")):
        attention_reasons.append("planned shared outputs remain pending")
    if status_context.infer_runtime.preferred_model_family is not None and any(
        dependencies.phase_matches_infer_model_family(
            phase_id=str(phase.get("id") or ""),
            model_family=status_context.infer_runtime.preferred_model_family,
        )
        for phase in study_context.blocked_phases
    ):
        attention_reasons.append(
            "preferred infer family "
            f"{status_context.infer_runtime.preferred_model_family} is blocked by declared GPU policy"
        )
    if study_context.blocked_phases:
        attention_reasons.append("GPU-only infer lanes remain blocked")

    summary = "; ".join(summary_parts)
    if attention_reasons:
        evidence["attention_reasons"] = attention_reasons
        return ("attention", summary, evidence)
    return ("ok", summary, evidence)


def _latentdna_readiness_drives_attention(latentdna_state: dict[str, object]) -> bool:
    if not bool(latentdna_state.get("configured")):
        return False
    state = str(latentdna_state.get("state") or "").strip()
    return state not in {"", "ok"}


def _attention_opal_run_receipt(
    downstream_surfaces: Mapping[str, object],
) -> Mapping[str, object] | None:
    opal_surface = downstream_surfaces.get("opal")
    if not isinstance(opal_surface, Mapping):
        return None
    run_receipt = opal_surface.get("run_receipt")
    if not isinstance(run_receipt, Mapping):
        return None
    if not bool(run_receipt.get("configured")):
        return None
    if bool(run_receipt.get("drives_top_level_attention")):
        return run_receipt
    return None


def _build_source_growth_state(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    handoff_readiness_state: dict[str, object] | None,
) -> dict[str, object] | None:
    if study_context.densegen_dataset_id is None or study_context.densegen_rows is None:
        return None
    dataset_id = study_context.densegen_dataset_id
    current_rows = int(study_context.densegen_rows)
    target_rows = study_context.densegen_row_target
    if target_rows is None:
        return {
            "state": "ok",
            "dataset": dataset_id,
            "current_rows": current_rows,
            "target_rows": None,
            "gap_rows": None,
            "summary": f"source rows visible {dataset_id} {current_rows} rows",
        }
    target_rows = int(target_rows)
    gap_rows = max(target_rows - current_rows, 0)
    source_phase = _resolve_source_phase(study_context=study_context, dataset_id=dataset_id)
    source_phase_id = None
    source_phase_status = None
    if source_phase is not None:
        source_phase_id = str(source_phase.get("id") or "").strip() or None
        source_phase_status = str(source_phase.get("status") or "").strip() or None
    handoff_rows = _ready_handoff_rows(study_context=study_context, handoff_readiness_state=handoff_readiness_state)
    max_handoff_rows = max(handoff_rows.values()) if handoff_rows else None
    source_gate_superseded = (
        gap_rows > 0
        and max_handoff_rows is not None
        and max_handoff_rows >= target_rows
        and _source_gate_is_historical(
            study_context=study_context,
            source_phase_id=source_phase_id,
            source_phase_status=source_phase_status,
        )
    )
    if gap_rows:
        if source_gate_superseded:
            return {
                "state": "ok",
                "dataset": dataset_id,
                "current_rows": current_rows,
                "target_rows": target_rows,
                "gap_rows": gap_rows,
                "target_met": False,
                "gates_current_phase": False,
                "source_phase_id": source_phase_id,
                "source_phase_status": source_phase_status,
                "superseded_by_handoffs": True,
                "max_handoff_rows": max_handoff_rows,
                "drives_top_level_attention": False,
                "summary": (
                    f"source gate superseded by downstream handoffs {dataset_id} "
                    f"{current_rows}/{target_rows} rows (gap={gap_rows})"
                ),
            }
        return {
            "state": "attention",
            "dataset": dataset_id,
            "current_rows": current_rows,
            "target_rows": target_rows,
            "gap_rows": gap_rows,
            "target_met": False,
            "gates_current_phase": True,
            "source_phase_id": source_phase_id,
            "source_phase_status": source_phase_status,
            "superseded_by_handoffs": False,
            "max_handoff_rows": max_handoff_rows,
            "drives_top_level_attention": True,
            "summary": f"source gate active {dataset_id} {current_rows}/{target_rows} rows (gap={gap_rows})",
        }
    return {
        "state": "ok",
        "dataset": dataset_id,
        "current_rows": current_rows,
        "target_rows": target_rows,
        "gap_rows": 0,
        "target_met": True,
        "gates_current_phase": False,
        "source_phase_id": source_phase_id,
        "source_phase_status": source_phase_status,
        "superseded_by_handoffs": False,
        "max_handoff_rows": max_handoff_rows,
        "drives_top_level_attention": False,
        "summary": f"source ready {dataset_id} {current_rows}/{target_rows} rows",
    }


def _build_handoff_readiness_state(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
) -> dict[str, object] | None:
    declared_handoffs = [
        ("anchor", study_context.merged_anchor_dataset_id, study_context.merged_anchor_rows),
        ("construct", study_context.construct_context_dataset_id, study_context.construct_context_rows),
    ]
    declared_handoffs = [
        (label, dataset_id, rows) for label, dataset_id, rows in declared_handoffs if dataset_id is not None
    ]
    if not declared_handoffs:
        return None

    pending_datasets: list[str] = []
    ready_counts: list[str] = []
    for label, dataset_id, rows in declared_handoffs:
        dataset_state = next(
            (state for state in study_context.dataset_states if str(state.get("dataset") or "").strip() == dataset_id),
            None,
        )
        exists = bool(dataset_state and dataset_state.get("exists"))
        if not exists:
            pending_datasets.append(dataset_id)
            continue
        if rows is None:
            ready_counts.append(f"{label}=unknown")
        else:
            ready_counts.append(f"{label}={rows}")

    if pending_datasets:
        return {
            "state": "attention",
            "pending_datasets": pending_datasets,
            "stale_datasets": [],
            "drives_top_level_attention": True,
            "summary": "handoff outputs pending " + ", ".join(pending_datasets),
        }
    if study_context.stale_dataset_ids:
        return {
            "state": "attention",
            "pending_datasets": [],
            "stale_datasets": list(study_context.stale_dataset_ids),
            "drives_top_level_attention": True,
            "summary": "handoff lag " + ", ".join(study_context.stale_dataset_ids),
        }
    return {
        "state": "ok",
        "pending_datasets": [],
        "stale_datasets": [],
        "drives_top_level_attention": False,
        "summary": "handoffs ready " + " ".join(ready_counts),
    }


def _build_planned_outputs_state(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
) -> dict[str, object] | None:
    handoff_ids = {
        dataset_id
        for dataset_id in (
            study_context.merged_anchor_dataset_id,
            study_context.construct_context_dataset_id,
        )
        if dataset_id is not None
    }
    completed_output_ids = {
        str(phase.get("output_dataset") or "").strip()
        for phase in study_context.phase_states
        if str(phase.get("status") or "").strip() == "complete"
    }
    completed_output_ids.discard("")
    pending_outputs = [
        str(state["dataset"])
        for state in study_context.dataset_states
        if state["declared_status"] != "present"
        and not state["exists"]
        and str(state["dataset"]) not in handoff_ids
        and str(state["dataset"]) not in completed_output_ids
    ]
    if pending_outputs:
        return {
            "state": "ok",
            "pending_datasets": pending_outputs,
            "drives_top_level_attention": False,
            "include_in_summary": False,
            "summary": "future outputs still planned " + ", ".join(pending_outputs),
        }
    return {
        "state": "ok",
        "pending_datasets": [],
        "drives_top_level_attention": False,
        "include_in_summary": False,
        "summary": "planned outputs clear",
    }


def _resolve_source_phase(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    dataset_id: str,
) -> dict[str, object] | None:
    for phase in study_context.phase_states:
        if str(phase.get("primary_dataset") or "").strip() == dataset_id:
            return phase
    for phase in study_context.phase_states:
        phase_id = str(phase.get("id") or "").strip()
        if phase_id == "densegen_growth":
            return phase
    return None


def _ready_handoff_rows(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    handoff_readiness_state: dict[str, object] | None,
) -> dict[str, int]:
    if handoff_readiness_state is None or str(handoff_readiness_state.get("state") or "").strip() != "ok":
        return {}
    rows: dict[str, int] = {}
    if study_context.merged_anchor_dataset_id is not None and study_context.merged_anchor_rows is not None:
        rows[study_context.merged_anchor_dataset_id] = int(study_context.merged_anchor_rows)
    if study_context.construct_context_dataset_id is not None and study_context.construct_context_rows is not None:
        rows[study_context.construct_context_dataset_id] = int(study_context.construct_context_rows)
    return rows


def _source_gate_is_historical(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    source_phase_id: str | None,
    source_phase_status: str | None,
) -> bool:
    if source_phase_status in {"parallel_optional", "complete"}:
        return True
    if source_phase_id is None or study_context.current_phase is None or study_context.ops_contract is None:
        return False
    phase_order = tuple(study_context.ops_contract.phase_order)
    try:
        source_index = phase_order.index(source_phase_id)
        current_index = phase_order.index(study_context.current_phase)
    except ValueError:
        return False
    return current_index > source_index
