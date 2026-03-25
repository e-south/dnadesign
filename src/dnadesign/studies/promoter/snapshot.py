"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/promoter/snapshot.py

Study-owned snapshot enrichment and summary assembly for the
promoter family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .context import PromoterStudyResolvedContext
from .infer_runtime import (
    PromoterStudyInferRuntimeDependencies,
    PromoterStudyInferRuntimeResolvedContext,
    resolve_promoter_study_infer_runtime_context,
)


@dataclass(frozen=True)
class PromoterStudyStatusDependencies:
    infer_runtime: PromoterStudyInferRuntimeDependencies
    phase_matches_infer_model_family: Callable[..., bool]


@dataclass(frozen=True)
class PromoterStudyStatusResolvedContext:
    infer_runtime: PromoterStudyInferRuntimeResolvedContext


def resolve_promoter_study_status_context(
    *,
    study_context: PromoterStudyResolvedContext,
    status_kind: str,
    dependencies: PromoterStudyStatusDependencies,
) -> PromoterStudyStatusResolvedContext:
    infer_runtime = resolve_promoter_study_infer_runtime_context(
        study_context=study_context,
        status_kind=status_kind,
        dependencies=dependencies.infer_runtime,
    )
    return PromoterStudyStatusResolvedContext(infer_runtime=infer_runtime)


def build_promoter_study_status(
    *,
    study_context: PromoterStudyResolvedContext,
    status_context: PromoterStudyStatusResolvedContext,
    dependencies: PromoterStudyStatusDependencies,
    summary_scope: str,
) -> tuple[str, str, dict[str, object]]:
    evidence = dict(study_context.evidence)
    evidence.update(
        {
            "summary_scope": summary_scope,
            "preferred_infer_model_family": status_context.infer_runtime.preferred_model_family,
            "supported_infer_model_families": list(status_context.infer_runtime.supported_model_families),
            "infer_runtime_models": [
                summary.as_dict() for summary in status_context.infer_runtime.runtime_model_summaries
            ],
            "infer_notify_profiles": {
                label: str(path) for label, path in status_context.infer_runtime.infer_notify_profile_paths.items()
            },
            "infer_notify_profile_errors": dict(status_context.infer_runtime.infer_notify_profile_errors),
        }
    )

    summary_parts = [f"{study_context.resolved_study_dir.name}: phase {study_context.current_phase or 'unknown'}"]
    if status_context.infer_runtime.preferred_model_family is not None:
        summary_parts.append(f"preferred infer {status_context.infer_runtime.preferred_model_family}")
    if (
        study_context.densegen_dataset_id is not None
        and study_context.densegen_rows is not None
        and study_context.densegen_row_target is not None
    ):
        summary_parts.append(
            (
                f"{study_context.densegen_dataset_id} "
                f"{study_context.densegen_rows}/{study_context.densegen_row_target} rows"
            )
        )
    elif study_context.densegen_dataset_id is not None and study_context.densegen_rows is not None:
        summary_parts.append(f"{study_context.densegen_dataset_id} {study_context.densegen_rows} rows")

    pending_shared_datasets = [
        str(state["dataset"])
        for state in study_context.dataset_states
        if state["declared_status"] != "present" and not state["exists"]
    ]
    if pending_shared_datasets:
        summary_parts.append("pending " + ", ".join(pending_shared_datasets))
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
    if study_context.densegen_row_gap not in (None, 0):
        attention_reasons.append("DenseGen anchor target not met")
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
    if any(
        str(phase.get("status") or "") in {"ready", "planned", "in_progress", "blocked_gpu"}
        for phase in study_context.phase_states
    ):
        attention_reasons.append("study is not complete")

    summary = "; ".join(summary_parts)
    if attention_reasons:
        evidence["attention_reasons"] = attention_reasons
        return ("attention", summary, evidence)
    return ("ok", summary, evidence)
