"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/preflight.py

Study-owned preflight context resolution and orchestration for the
promoter family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.preflight import (
    CommandExecution,
    ContractPreflightCheckDependencies,
    PreflightCheck,
    build_contract_preflight_checks,
    contract_environment_flag_state,
    evaluate_preflight_checks,
)
from dnadesign.studies.core.models import StudyOpsContract
from dnadesign.studies.core.preflight_plan import StudyPreflightPlan, build_study_preflight_plan

from .infer_runtime import (
    PromoterInferPhaseTarget,
    PromoterStudyInferRuntimeDependencies,
    PromoterStudyInferRuntimeResolvedContext,
    resolve_promoter_study_infer_runtime_context,
)
from .record_normalizer import PromoterStudyResolvedContext


@dataclass(frozen=True)
class PromoterPreflightContextDependencies:
    infer_runtime: PromoterStudyInferRuntimeDependencies
    environ: Mapping[str, object | None]


@dataclass(frozen=True)
class PromoterPreflightCoordinatorDependencies:
    run_preflight_command: Callable[..., CommandExecution]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    choose_command_summary: Callable[..., str]
    inspect_local_gpu_inventory: Callable[[], dict[str, object]]
    environ: Mapping[str, object | None]


@dataclass(frozen=True)
class PromoterPreflightResolvedContext:
    contract: StudyOpsContract
    study_id: str
    study_repo_root: Path
    resolved_study_dir: Path
    study_pipeline: dict[str, object]
    execution_surface_index: dict[str, Path]
    dataset_index: dict[str, dict[str, object]]
    phase_states: tuple[Mapping[str, object], ...]
    current_phase: str | None
    next_ready_phase: Mapping[str, object] | None
    infer_runtime: PromoterStudyInferRuntimeResolvedContext
    infer_phase_targets: dict[str, PromoterInferPhaseTarget]
    scope_plan: StudyPreflightPlan


def resolve_promoter_preflight_context(
    *,
    study_context: PromoterStudyResolvedContext,
    scope: str | None,
    status_kind: str,
    contract: StudyOpsContract,
    dependencies: PromoterPreflightContextDependencies,
) -> PromoterPreflightResolvedContext:
    study_repo_root = study_context.study_repo_root
    if study_repo_root is None:
        raise ValueError("promoter-study preflight resolution requires a resolved study_repo_root")
    study_pipeline = dict(study_context.study_pipeline)
    execution_surface_index = dict(study_context.execution_surface_index)
    dataset_index = {dataset_id: dict(payload) for dataset_id, payload in study_context.dataset_index.items()}
    infer_runtime = resolve_promoter_study_infer_runtime_context(
        study_context=study_context,
        status_kind=status_kind,
        dependencies=dependencies.infer_runtime,
    )
    infer_phase_targets = dict(infer_runtime.phase_targets_by_id)
    current_phase = study_context.current_phase
    next_ready_phase = dict(study_context.next_ready_phase) if study_context.next_ready_phase is not None else None
    scope_plan = build_study_preflight_plan(
        current_phase=current_phase,
        next_ready_phase=next_ready_phase,
        scope=scope,
        contract=contract.preflight,
        runtime_phase_ids=tuple(infer_phase_targets),
    )
    study_id = study_context.study_id or study_context.resolved_study_dir.name
    phase_states = tuple(dict(phase) for phase in study_context.phase_states)
    return PromoterPreflightResolvedContext(
        contract=contract,
        study_id=study_id,
        study_repo_root=study_repo_root,
        resolved_study_dir=study_context.resolved_study_dir,
        study_pipeline=study_pipeline,
        execution_surface_index=execution_surface_index,
        dataset_index=dataset_index,
        phase_states=phase_states,
        current_phase=current_phase,
        next_ready_phase=next_ready_phase,
        infer_runtime=infer_runtime,
        infer_phase_targets=infer_phase_targets,
        scope_plan=scope_plan,
    )


def build_promoter_preflight_progress(
    *,
    context: PromoterPreflightResolvedContext,
    evidence: Mapping[str, object],
    dependencies: PromoterPreflightCoordinatorDependencies,
) -> tuple[str, str, dict[str, object]]:
    checks: list[PreflightCheck] = []
    counts: Counter[str] = Counter()
    resolved_evidence = dict(evidence)
    enabled_groups = set(context.scope_plan.included_groups)
    include_infer_checks = "infer" in enabled_groups
    local_gpu_inventory = (
        dependencies.inspect_local_gpu_inventory()
        if include_infer_checks
        else {"count": 0, "devices": [], "probe_error": None}
    )
    resolved_evidence.update(
        {
            "preferred_infer_model_family": context.infer_runtime.preferred_model_family,
            "supported_model_families": list(context.infer_runtime.supported_model_families),
            "infer_local_gpu_inventory": local_gpu_inventory,
            "infer_notify_profiles": {
                label: str(path) for label, path in context.infer_runtime.infer_notify_profile_paths.items()
            },
            "infer_notify_profile_errors": dict(context.infer_runtime.infer_notify_profile_errors),
        }
    )

    def add_check(check: PreflightCheck) -> None:
        checks.append(check)
        counts[check.state] += 1

    for check in build_contract_preflight_checks(
        repo_root=context.study_repo_root,
        study_root=context.resolved_study_dir,
        contract=context.contract,
        dataset_index=context.dataset_index,
        execution_surface_index=context.execution_surface_index,
        enabled_groups=enabled_groups,
        environ=dependencies.environ,
        gpu_inventory=local_gpu_inventory,
        dependencies=ContractPreflightCheckDependencies(
            run_preflight_command=dependencies.run_preflight_command,
            safe_json_loads=dependencies.safe_json_loads,
            choose_command_summary=dependencies.choose_command_summary,
            inspect_local_gpu_inventory=dependencies.inspect_local_gpu_inventory,
        ),
    ):
        add_check(check)

    resolved_evidence.update(
        {
            "notify_environment": contract_environment_flag_state(
                contract=context.contract,
                environ=dependencies.environ,
                check_group="notify_environment",
            ),
            "checks": [check.as_dict() for check in checks],
            "scope": context.scope_plan.scope,
            "counts": {state: int(counts.get(state, 0)) for state in ("ok", "attention", "missing")},
        }
    )

    evaluation = evaluate_preflight_checks(
        checks,
        phase_states=context.phase_states,
        scope_plan=context.scope_plan,
    )
    resolved_evidence.update(
        {
            "target_phase": context.scope_plan.target_phase_id,
            "scoped_counts": dict(evaluation.scoped_counts),
            "blocked_by": list(evaluation.blocked_by_ids),
            "deferred_check_ids": list(evaluation.deferred_check_ids),
            "nonblocking_attention_ids": list(evaluation.nonblocking_attention_ids),
        }
    )

    summary_parts = [f"{context.study_id}: preflight phase {context.current_phase or 'unknown'}"]
    if context.scope_plan.scope == "next" and context.scope_plan.target_phase_id is not None:
        phase_label = "next phase" if context.next_ready_phase is not None else "focus phase"
        summary_parts.append(f"{phase_label} {context.scope_plan.target_phase_id}")
    effective_counts = evaluation.effective_counts
    if effective_counts.get("ok"):
        summary_parts.append(f"{effective_counts['ok']} ok")
    if effective_counts.get("attention"):
        summary_parts.append(f"{effective_counts['attention']} attention")
    if effective_counts.get("missing"):
        summary_parts.append(f"{effective_counts['missing']} missing")
    if evaluation.blocker_checks:
        blocker_label = "blocked by" if context.scope_plan.scope == "next" else "first blockers"
        summary_parts.append(f"{blocker_label}: " + ", ".join(check.id for check in evaluation.blocker_checks[:3]))
    elif context.scope_plan.scope == "next" and evaluation.nonblocking_attention_checks:
        summary_parts.append(
            "ready with advisories: " + ", ".join(check.id for check in evaluation.nonblocking_attention_checks[:3])
        )
    elif context.scope_plan.scope == "next" and context.scope_plan.target_phase_id is not None:
        summary_parts.append("ready")
    if context.scope_plan.scope == "next" and evaluation.deferred_blockers:
        summary_parts.append(
            "deferred downstream blockers: " + ", ".join(check.id for check in evaluation.deferred_blockers[:3])
        )

    if effective_counts.get("missing"):
        return ("missing", "; ".join(summary_parts), resolved_evidence)
    if effective_counts.get("attention") or effective_counts.get("missing"):
        return ("attention", "; ".join(summary_parts), resolved_evidence)
    return ("ok", "; ".join(summary_parts), resolved_evidence)


__all__ = [
    "PromoterPreflightContextDependencies",
    "PromoterPreflightCoordinatorDependencies",
    "PromoterPreflightResolvedContext",
    "build_promoter_preflight_progress",
    "resolve_promoter_preflight_context",
]
