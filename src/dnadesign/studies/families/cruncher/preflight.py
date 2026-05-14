"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/cruncher/preflight.py

Read-only preflight coordination for Cruncher study-family records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dnadesign.ops.preflight import (
    choose_command_summary,
    evaluate_preflight_checks,
    run_preflight_command,
    safe_json_loads,
)
from dnadesign.ops.preflight.contract_checks import (
    ContractPreflightCheckDependencies,
    build_contract_preflight_checks,
)
from dnadesign.studies.core.models import StudyOpsContract
from dnadesign.studies.core.preflight_plan import StudyPreflightPlan, build_study_preflight_plan

from .record_normalizer import CruncherStudyResolvedContext


@dataclass(frozen=True)
class CruncherPreflightResolvedContext:
    study_context: CruncherStudyResolvedContext
    contract: StudyOpsContract
    scope_plan: StudyPreflightPlan


@dataclass(frozen=True)
class CruncherPreflightDependencies:
    run_preflight_command: object = run_preflight_command
    safe_json_loads: object = safe_json_loads
    choose_command_summary: object = choose_command_summary
    inspect_local_gpu_inventory: object = lambda: {"count": 0, "devices": [], "probe_error": None}
    execute_runbook_plan: object | None = None
    environ: Mapping[str, object | None] = None  # type: ignore[assignment]


def resolve_cruncher_preflight_context(
    *,
    study_context: CruncherStudyResolvedContext,
    scope: str | None,
    contract: StudyOpsContract,
) -> CruncherPreflightResolvedContext:
    scope_plan = build_study_preflight_plan(
        current_phase=study_context.current_phase,
        next_ready_phase=study_context.next_ready_phase,
        scope=scope,
        contract=contract.preflight,
        runtime_phase_ids=(),
    )
    return CruncherPreflightResolvedContext(
        study_context=study_context,
        contract=contract,
        scope_plan=scope_plan,
    )


def build_cruncher_preflight_progress(
    *,
    context: CruncherPreflightResolvedContext,
    dependencies: CruncherPreflightDependencies,
) -> tuple[str, str, dict[str, object]]:
    checks = build_contract_preflight_checks(
        repo_root=context.study_context.study_repo_root,
        study_root=context.study_context.resolved_study_dir,
        contract=context.contract,
        dataset_index={},
        execution_surface_index=context.study_context.execution_surface_index,
        enabled_groups=context.scope_plan.included_groups,
        environ=dependencies.environ or {},
        dependencies=ContractPreflightCheckDependencies(
            run_preflight_command=dependencies.run_preflight_command,  # type: ignore[arg-type]
            safe_json_loads=dependencies.safe_json_loads,  # type: ignore[arg-type]
            choose_command_summary=dependencies.choose_command_summary,  # type: ignore[arg-type]
            inspect_local_gpu_inventory=dependencies.inspect_local_gpu_inventory,  # type: ignore[arg-type]
            execute_runbook_plan=dependencies.execute_runbook_plan,  # type: ignore[arg-type]
        ),
    )
    evaluation = evaluate_preflight_checks(
        checks,
        phase_states=context.contract.phase_states,
        scope_plan=context.scope_plan,
    )
    state = "attention" if evaluation.blocker_checks or evaluation.nonblocking_attention_checks else "ok"
    item_label = str(context.contract.lifecycle_item_label or "phase").strip() or "phase"
    item_key = item_label.strip().lower().replace("-", "_").replace(" ", "_") or "phase"
    summary = (
        f"{context.study_context.study_id}: preflight {context.scope_plan.scope} "
        f"for {context.scope_plan.target_phase_id or f'all {item_label}s'}; "
        f"blockers {len(evaluation.blocker_checks)}"
    )
    evidence = dict(context.study_context.evidence)
    evidence.update(
        {
            "scope": context.scope_plan.scope,
            "lifecycle_mode": context.contract.lifecycle_mode,
            "lifecycle_item_label": item_label,
            f"{item_key}_id": context.scope_plan.target_phase_id,
            "included_groups": list(context.scope_plan.included_groups),
            "phase_scoped_groups": list(context.scope_plan.phase_scoped_groups),
            "checks": [check.as_dict() for check in evaluation.scoped_checks],
            "blocked_by_ids": list(evaluation.blocked_by_ids),
            "deferred_check_ids": list(evaluation.deferred_check_ids),
            "nonblocking_attention_ids": list(evaluation.nonblocking_attention_ids),
            "scoped_counts": dict(evaluation.scoped_counts),
            "effective_counts": dict(evaluation.effective_counts),
        }
    )
    if item_key == "phase":
        evidence["phase_id"] = context.scope_plan.target_phase_id
    return state, summary, evidence


__all__ = [
    "CruncherPreflightDependencies",
    "CruncherPreflightResolvedContext",
    "build_cruncher_preflight_progress",
    "resolve_cruncher_preflight_context",
]
