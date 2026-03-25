"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/stress_promoter_ethanol_cipro/preflight.py

Study-owned preflight context resolution and orchestration for the
stress_promoter_ethanol_cipro family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.contracts import InferRuntimePhaseTarget
from dnadesign.studies.core.models import StudyOpsContract

from .context import PromoterStudyResolvedContext
from .infer_runtime import (
    PromoterStudyInferRuntimeDependencies,
    PromoterStudyInferRuntimeResolvedContext,
    resolve_promoter_study_infer_runtime_context,
)
from .preflight_infer import (
    PromoterPreflightInferDependencies,
    build_promoter_preflight_infer_checks,
)
from .preflight_orchestration import (
    PromoterPreflightNotifyEnvironmentDependencies,
    PromoterPreflightRunbookPlanDependencies,
    PromoterPreflightRunbookPlanTarget,
    build_promoter_preflight_notify_environment_checks,
    build_promoter_preflight_runbook_plan_checks,
)
from .preflight_scope import (
    PromoterPreflightScopePlan,
    build_promoter_preflight_scope_plan,
    evaluate_promoter_preflight_checks,
)
from .preflight_upstream import (
    PromoterPreflightUpstreamDependencies,
    build_promoter_preflight_upstream_checks,
)


@dataclass(frozen=True)
class PromoterPreflightContextDependencies:
    infer_runtime: PromoterStudyInferRuntimeDependencies
    resolve_notify_environment_state: Callable[..., dict[str, bool]]
    environ: Mapping[str, object | None]


@dataclass(frozen=True)
class PromoterPreflightCoordinatorDependencies:
    load_orchestration_runbook_payload: Callable[[Path], dict[str, object]]
    resolve_input_path: Callable[[Path, Path | None], Path]
    run_progress_command: Callable[..., object]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    preflight_state_check: Callable[..., dict[str, object]]
    preflight_command_check: Callable[..., dict[str, object]]
    choose_command_summary: Callable[..., str]
    inspect_local_gpu_inventory: Callable[[], dict[str, object]]
    infer_usr_dataset_requirements: Callable[[Path], list[dict[str, object]]]
    build_infer_notify_setup_command: Callable[[Path], str]
    validate_infer_config_contract: Callable[[Path], object] | None = None
    validate_infer_dry_run_contract: Callable[[Path], object] | None = None
    resolve_infer_usr_output_contract: Callable[[Path], object] | None = None


@dataclass(frozen=True)
class PromoterPreflightResolvedContext:
    study_id: str
    study_repo_root: Path
    study_pipeline: dict[str, object]
    execution_surface_index: dict[str, Path]
    dataset_index: dict[str, dict[str, object]]
    phase_states: tuple[Mapping[str, object], ...]
    current_phase: str | None
    next_ready_phase: Mapping[str, object] | None
    infer_runtime: PromoterStudyInferRuntimeResolvedContext
    infer_phase_targets: dict[str, InferRuntimePhaseTarget]
    infer_batch_targets: tuple[PromoterPreflightRunbookPlanTarget, ...]
    densegen_phase_id: str
    construct_phase_id: str
    infer_preparation_phase_id: str
    scope_plan: PromoterPreflightScopePlan
    notify_env_state: dict[str, bool]


def resolve_promoter_preflight_context(
    *,
    study_context: PromoterStudyResolvedContext,
    scope: str | None,
    progress_kind: str,
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
        progress_kind=progress_kind,
        dependencies=dependencies.infer_runtime,
    )
    infer_phase_targets = dict(infer_runtime.phase_targets_by_id)
    densegen_phase_id = str(contract.preflight_phase_targets.get("densegen") or "").strip()
    construct_phase_id = str(contract.preflight_phase_targets.get("construct") or "").strip()
    infer_preparation_phase_id = str(contract.preflight_phase_targets.get("infer_preparation") or "").strip()
    if not densegen_phase_id or not construct_phase_id or not infer_preparation_phase_id:
        raise ValueError(
            "ops.study.yaml must define preflight.phase_targets for densegen, construct, and infer_preparation"
        )
    current_phase = study_context.current_phase
    next_ready_phase = dict(study_context.next_ready_phase) if study_context.next_ready_phase is not None else None
    scope_plan = build_promoter_preflight_scope_plan(
        current_phase=current_phase,
        next_ready_phase=next_ready_phase,
        scope=scope,
        default_scope=contract.preflight_default_scope,
        phase_group_overrides=contract.next_scope_phase_groups,
        infer_lane_groups=contract.infer_lane_groups,
        infer_phase_targets=infer_phase_targets,
    )
    notify_env_state = dependencies.resolve_notify_environment_state(environ=dependencies.environ)
    infer_batch_targets = _resolve_infer_batch_targets(
        execution_surface_index=execution_surface_index,
        infer_phase_targets=infer_runtime.phase_targets,
        notify_env_state=notify_env_state,
    )
    study_id = study_context.study_id or study_context.resolved_study_dir.name
    phase_states = tuple(dict(phase) for phase in study_context.phase_states)
    return PromoterPreflightResolvedContext(
        study_id=study_id,
        study_repo_root=study_repo_root,
        study_pipeline=study_pipeline,
        execution_surface_index=execution_surface_index,
        dataset_index=dataset_index,
        phase_states=phase_states,
        current_phase=current_phase,
        next_ready_phase=next_ready_phase,
        infer_runtime=infer_runtime,
        infer_phase_targets=infer_phase_targets,
        infer_batch_targets=infer_batch_targets,
        densegen_phase_id=densegen_phase_id,
        construct_phase_id=construct_phase_id,
        infer_preparation_phase_id=infer_preparation_phase_id,
        scope_plan=scope_plan,
        notify_env_state=notify_env_state,
    )


def build_promoter_preflight_progress(
    *,
    context: PromoterPreflightResolvedContext,
    evidence: Mapping[str, object],
    dependencies: PromoterPreflightCoordinatorDependencies,
) -> tuple[str, str, dict[str, object]]:
    checks: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    resolved_evidence = dict(evidence)

    def add_check(check: dict[str, object]) -> None:
        checks.append(check)
        counts[str(check.get("state") or "attention")] += 1

    for check in build_promoter_preflight_notify_environment_checks(
        notify_env_state=context.notify_env_state,
        infer_preparation_phase_id=context.infer_preparation_phase_id,
        include_notify_checks=context.scope_plan.include_notify_checks,
        dependencies=PromoterPreflightNotifyEnvironmentDependencies(
            preflight_state_check=dependencies.preflight_state_check,
        ),
    ):
        add_check(check)

    upstream_checks_result = build_promoter_preflight_upstream_checks(
        study_repo_root=context.study_repo_root,
        study_pipeline=context.study_pipeline,
        execution_surface_index=context.execution_surface_index,
        dataset_index=context.dataset_index,
        phase_states=context.phase_states,
        densegen_phase_id=context.densegen_phase_id,
        construct_phase_id=context.construct_phase_id,
        include_densegen_checks=context.scope_plan.include_densegen_checks,
        include_construct_checks=context.scope_plan.include_construct_checks,
        dependencies=PromoterPreflightUpstreamDependencies(
            load_orchestration_runbook_payload=dependencies.load_orchestration_runbook_payload,
            resolve_input_path=dependencies.resolve_input_path,
            run_progress_command=dependencies.run_progress_command,
            safe_json_loads=dependencies.safe_json_loads,
            preflight_state_check=dependencies.preflight_state_check,
            preflight_command_check=dependencies.preflight_command_check,
            choose_command_summary=dependencies.choose_command_summary,
        ),
    )
    for check in upstream_checks_result.checks:
        add_check(check)

    infer_checks_result = build_promoter_preflight_infer_checks(
        study_repo_root=context.study_repo_root,
        infer_runtime=context.infer_runtime,
        infer_preparation_phase_id=context.infer_preparation_phase_id,
        include_infer_checks=context.scope_plan.include_infer_checks,
        include_notify_checks=context.scope_plan.include_notify_checks,
        dependencies=PromoterPreflightInferDependencies(
            inspect_local_gpu_inventory=dependencies.inspect_local_gpu_inventory,
            infer_usr_dataset_requirements=dependencies.infer_usr_dataset_requirements,
            build_infer_notify_setup_command=dependencies.build_infer_notify_setup_command,
            run_progress_command=dependencies.run_progress_command,
            preflight_state_check=dependencies.preflight_state_check,
            preflight_command_check=dependencies.preflight_command_check,
            choose_command_summary=dependencies.choose_command_summary,
            validate_infer_config_contract=dependencies.validate_infer_config_contract,
            validate_infer_dry_run_contract=dependencies.validate_infer_dry_run_contract,
            resolve_infer_usr_output_contract=dependencies.resolve_infer_usr_output_contract,
        ),
    )
    resolved_evidence.update(infer_checks_result.evidence_updates)
    for check in infer_checks_result.checks:
        add_check(check)

    if context.scope_plan.include_infer_batch_plan_checks:
        for check in build_promoter_preflight_runbook_plan_checks(
            study_repo_root=context.study_repo_root,
            targets=context.infer_batch_targets,
            dependencies=PromoterPreflightRunbookPlanDependencies(
                run_progress_command=dependencies.run_progress_command,
                safe_json_loads=dependencies.safe_json_loads,
                preflight_command_check=dependencies.preflight_command_check,
                choose_command_summary=dependencies.choose_command_summary,
            ),
        ):
            add_check(check)

    resolved_evidence.update(
        {
            "notify_environment": context.notify_env_state,
            "checks": checks,
            "scope": context.scope_plan.scope,
            "counts": {state: int(counts.get(state, 0)) for state in ("ok", "attention", "missing")},
        }
    )

    evaluation = evaluate_promoter_preflight_checks(
        checks,
        phase_states=context.phase_states,
        scope_plan=context.scope_plan,
        infer_phase_targets=context.infer_phase_targets,
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
        summary_parts.append(
            f"{blocker_label}: " + ", ".join(str(check["id"]) for check in evaluation.blocker_checks[:3])
        )
    elif context.scope_plan.scope == "next" and context.scope_plan.target_phase_id is not None:
        summary_parts.append("ready")
    if context.scope_plan.scope == "next" and evaluation.deferred_blockers:
        summary_parts.append(
            "deferred downstream blockers: " + ", ".join(str(check["id"]) for check in evaluation.deferred_blockers[:3])
        )

    if effective_counts.get("missing"):
        return ("missing", "; ".join(summary_parts), resolved_evidence)
    if effective_counts.get("attention") or effective_counts.get("missing"):
        return ("attention", "; ".join(summary_parts), resolved_evidence)
    return ("ok", "; ".join(summary_parts), resolved_evidence)


def _resolve_infer_batch_targets(
    *,
    execution_surface_index: Mapping[str, Path],
    infer_phase_targets: Sequence[InferRuntimePhaseTarget],
    notify_env_state: Mapping[str, bool],
) -> tuple[PromoterPreflightRunbookPlanTarget, ...]:
    targets: list[PromoterPreflightRunbookPlanTarget] = []
    for phase_target in infer_phase_targets:
        runbook_path = execution_surface_index.get(phase_target.runbook_surface_label)
        if runbook_path is None:
            continue
        targets.append(
            PromoterPreflightRunbookPlanTarget(
                check_id=f"ops.runbook_plan.{phase_target.runbook_surface_label}",
                phase="ops",
                phase_id=phase_target.phase_id,
                runbook_path=runbook_path,
                fallback_summary="ops runbook plan completed",
                details={"notify_env": dict(notify_env_state)},
            )
        )
    return tuple(targets)


__all__ = [
    "PromoterPreflightContextDependencies",
    "PromoterPreflightCoordinatorDependencies",
    "PromoterPreflightResolvedContext",
    "build_promoter_preflight_progress",
    "resolve_promoter_preflight_context",
]
