"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/preflight.py

Study-owned preflight context resolution and orchestration for the
stress_ethanol_cipro_growth status surface.

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
    build_state_check,
    contract_environment_flag_state,
    evaluate_preflight_checks,
    execute_runbook_plan,
)
from dnadesign.ops.status import combine_states
from dnadesign.studies.core.models import StudyOpsContract
from dnadesign.studies.core.preflight_plan import StudyPreflightPlan, build_study_preflight_plan

from .infer_runtime import (
    StressEthanolCiproGrowthInferPhaseTarget,
    StressEthanolCiproGrowthInferRuntimeDependencies,
    StressEthanolCiproGrowthInferRuntimeResolvedContext,
    resolve_stress_ethanol_cipro_growth_infer_runtime_context,
)
from .record_normalizer import StressEthanolCiproGrowthResolvedContext


@dataclass(frozen=True)
class StressEthanolCiproGrowthPreflightContextDependencies:
    infer_runtime: StressEthanolCiproGrowthInferRuntimeDependencies
    environ: Mapping[str, object | None]


@dataclass(frozen=True)
class StressEthanolCiproGrowthPreflightCoordinatorDependencies:
    run_preflight_command: Callable[..., CommandExecution]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    choose_command_summary: Callable[..., str]
    inspect_local_gpu_inventory: Callable[[], dict[str, object]]
    environ: Mapping[str, object | None]
    execute_runbook_plan: Callable[..., CommandExecution] | None = None


@dataclass(frozen=True)
class StressEthanolCiproGrowthPreflightResolvedContext:
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
    dataset_refresh_states: tuple[Mapping[str, object], ...]
    infer_runtime: StressEthanolCiproGrowthInferRuntimeResolvedContext
    infer_phase_targets: dict[str, StressEthanolCiproGrowthInferPhaseTarget]
    scope_plan: StudyPreflightPlan


def resolve_stress_ethanol_cipro_growth_preflight_context(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    scope: str | None,
    status_kind: str,
    contract: StudyOpsContract,
    dependencies: StressEthanolCiproGrowthPreflightContextDependencies,
) -> StressEthanolCiproGrowthPreflightResolvedContext:
    study_repo_root = study_context.study_repo_root
    if study_repo_root is None:
        raise ValueError("stress_ethanol_cipro_growth preflight resolution requires a resolved study_repo_root")
    study_pipeline = dict(study_context.study_pipeline)
    execution_surface_index = dict(study_context.execution_surface_index)
    dataset_index = {dataset_id: dict(payload) for dataset_id, payload in study_context.dataset_index.items()}
    infer_runtime = resolve_stress_ethanol_cipro_growth_infer_runtime_context(
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
    return StressEthanolCiproGrowthPreflightResolvedContext(
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
        dataset_refresh_states=tuple(dict(state) for state in study_context.dataset_refresh_states),
        infer_runtime=infer_runtime,
        infer_phase_targets=infer_phase_targets,
        scope_plan=scope_plan,
    )


def build_stress_ethanol_cipro_growth_preflight_progress(
    *,
    context: StressEthanolCiproGrowthPreflightResolvedContext,
    evidence: Mapping[str, object],
    dependencies: StressEthanolCiproGrowthPreflightCoordinatorDependencies,
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
            execute_runbook_plan=dependencies.execute_runbook_plan or execute_runbook_plan,
            safe_json_loads=dependencies.safe_json_loads,
            choose_command_summary=dependencies.choose_command_summary,
            inspect_local_gpu_inventory=dependencies.inspect_local_gpu_inventory,
        ),
    ):
        add_check(check)
    for check in _build_dataset_refresh_checks(context=context):
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

    effective_state = combine_states(
        check.state for check in (evaluation.scoped_checks if context.scope_plan.scope == "next" else checks)
    )
    return (effective_state, "; ".join(summary_parts), resolved_evidence)


__all__ = [
    "StressEthanolCiproGrowthPreflightContextDependencies",
    "StressEthanolCiproGrowthPreflightCoordinatorDependencies",
    "StressEthanolCiproGrowthPreflightResolvedContext",
    "build_stress_ethanol_cipro_growth_preflight_progress",
    "resolve_stress_ethanol_cipro_growth_preflight_context",
]


def _build_dataset_refresh_checks(
    *,
    context: StressEthanolCiproGrowthPreflightResolvedContext,
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    phase_id = _infer_freshness_phase_id(context=context)
    if phase_id is None:
        return ()
    for refresh_state in context.dataset_refresh_states:
        state = str(refresh_state.get("state") or "").strip()
        if state not in {"ok", "attention", "missing"}:
            continue
        checks.append(
            build_state_check(
                check_id=f"infer.input.{str(refresh_state.get('id') or '').strip()}",
                kind="dataset_snapshot",
                required=True,
                check_group="infer",
                phase="infer",
                phase_id=phase_id,
                state=state,
                summary=str(refresh_state.get("summary") or "").strip(),
                artifact_id=str(refresh_state.get("downstream_dataset") or "").strip() or None,
                details=dict(refresh_state),
            )
        )
    return tuple(checks)


def _infer_freshness_phase_id(*, context: StressEthanolCiproGrowthPreflightResolvedContext) -> str | None:
    target_phase_id = str(context.scope_plan.target_phase_id or "").strip()
    if target_phase_id.startswith("infer"):
        return target_phase_id
    if str(context.current_phase or "").strip().startswith("infer"):
        return str(context.current_phase)
    return "infer_batch_preparation"
