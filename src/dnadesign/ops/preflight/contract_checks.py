"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/contract_checks.py

Generic execution for checked-in study preflight check specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dnadesign.ops.status.path_ref import resolve_path_ref
from dnadesign.usr import Dataset, SequenceViewContractExpectation, validate_sequence_view_contract

from .check_protocols import CommandCheckTarget, RunbookPlanCheckTarget, SchedulerQueueCheckTarget
from .checks import (
    CommandCheckDependencies,
    RunbookPlanCheckDependencies,
    SchedulerQueueCheckDependencies,
    build_command_checks,
    build_runbook_plan_checks,
    build_scheduler_queue_checks,
)
from .models import CommandExecution, PreflightCheck, build_command_check, build_state_check
from .support import execute_runbook_plan as default_execute_runbook_plan

_ENVIRONMENT_MATCH_MODES = frozenset({"all", "any"})


class _PreflightContractLike(Protocol):
    check_specs: Mapping[str, Sequence[Mapping[str, object]]]


class _StudyOpsContractLike(Protocol):
    preflight: _PreflightContractLike
    artifacts: Mapping[str, Mapping[str, object]]
    execution_surfaces: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class _CompiledContractCheck:
    check_id: str
    kind: str
    check_group: str
    phase_id: str
    phase: str
    summary: str
    required: bool
    payload: dict[str, object]


@dataclass(frozen=True)
class ContractPreflightCheckDependencies:
    run_preflight_command: Callable[..., CommandExecution]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    choose_command_summary: Callable[..., str]
    inspect_local_gpu_inventory: Callable[[], dict[str, object]]
    execute_runbook_plan: Callable[..., CommandExecution] | None = None


def build_contract_preflight_checks(
    *,
    repo_root: Path,
    study_root: Path,
    contract: _StudyOpsContractLike,
    dataset_index: Mapping[str, Mapping[str, object]],
    execution_surface_index: Mapping[str, Path],
    enabled_groups: Collection[str],
    environ: Mapping[str, object | None],
    gpu_inventory: Mapping[str, object] | None = None,
    dependencies: ContractPreflightCheckDependencies,
) -> tuple[PreflightCheck, ...]:
    enabled_group_set = {str(group).strip() for group in enabled_groups if str(group).strip()}
    checks: list[PreflightCheck] = []
    gpu_inventory_cache = dict(gpu_inventory or {}) or None

    compiled_checks = _compile_contract_preflight_checks(
        check_specs=contract.preflight.check_specs,
        enabled_groups=tuple(enabled_group_set),
    )
    execute_runbook_plan = dependencies.execute_runbook_plan or default_execute_runbook_plan
    for compiled in compiled_checks:
        kind = compiled.kind
        check_group = compiled.check_group
        phase_id = compiled.phase_id
        phase = compiled.phase
        check_id = compiled.check_id
        summary = compiled.summary
        required = compiled.required
        spec = dict(compiled.payload)
        details = {
            "contract_phase_id": phase_id,
        }

        if kind == "path_exists":
            artifact_id = str(spec.get("artifact") or "").strip()
            checks.append(
                _build_path_exists_check(
                    check_id=check_id,
                    check_group=check_group,
                    phase=phase,
                    phase_id=phase_id,
                    summary=summary,
                    required=required,
                    artifact_id=artifact_id,
                    contract=contract,
                    repo_root=repo_root,
                    study_root=study_root,
                    dataset_index=dataset_index,
                    base_details=details,
                )
            )
            continue

        if kind == "dataset_snapshot":
            artifact_id = str(spec.get("artifact") or "").strip()
            checks.append(
                _build_dataset_snapshot_check(
                    check_id=check_id,
                    check_group=check_group,
                    phase=phase,
                    phase_id=phase_id,
                    summary=summary,
                    required=required,
                    artifact_id=artifact_id,
                    target_rows=int(spec.get("target_rows") or 0),
                    contract=contract,
                    repo_root=repo_root,
                    study_root=study_root,
                    dataset_index=dataset_index,
                    base_details=details,
                )
            )
            continue

        if kind == "sequence_view_contract":
            artifact_id = str(spec.get("artifact") or "").strip()
            checks.append(
                _build_sequence_view_contract_check(
                    check_id=check_id,
                    check_group=check_group,
                    phase=phase,
                    phase_id=phase_id,
                    summary=summary,
                    required=required,
                    artifact_id=artifact_id,
                    expected_payload=spec.get("expected"),
                    contract=contract,
                    repo_root=repo_root,
                    study_root=study_root,
                    dataset_index=dataset_index,
                    base_details=details,
                )
            )
            continue

        if kind == "infer_sequence_view_completion":
            surface_id = str(spec.get("surface") or "").strip()
            checks.append(
                _build_infer_sequence_view_completion_check(
                    check_id=check_id,
                    check_group=check_group,
                    phase=phase,
                    phase_id=phase_id,
                    summary=summary,
                    required=required,
                    surface_id=surface_id,
                    expected_payload=spec.get("expected"),
                    contract=contract,
                    repo_root=repo_root,
                    study_root=study_root,
                    dependencies=dependencies,
                    base_details=details,
                )
            )
            continue

        if kind == "workspace_layout":
            surface_id = str(spec.get("surface") or "").strip()
            checks.append(
                _build_workspace_layout_check(
                    check_id=check_id,
                    check_group=check_group,
                    phase=phase,
                    phase_id=phase_id,
                    summary=summary,
                    required=required,
                    surface_id=surface_id,
                    execution_surface_index=execution_surface_index,
                    base_details=details,
                )
            )
            continue

        if kind == "environment":
            match_mode = str(spec.get("match_mode") or "all").strip().lower()
            if match_mode not in _ENVIRONMENT_MATCH_MODES:
                raise ValueError(f"environment check {check_id!r} has unsupported match_mode {match_mode!r}")
            vars_list = tuple(str(name).strip() for name in spec.get("vars") or () if str(name).strip())
            checks.append(
                _build_environment_check(
                    check_id=check_id,
                    check_group=check_group,
                    phase=phase,
                    phase_id=phase_id,
                    summary=summary,
                    required=required,
                    vars_list=vars_list,
                    match_mode=match_mode,
                    environ=environ,
                    base_details=details,
                )
            )
            continue

        if kind == "gpu_availability":
            if gpu_inventory_cache is None:
                gpu_inventory_cache = dict(dependencies.inspect_local_gpu_inventory() or {})
            checks.append(
                _build_gpu_availability_check(
                    check_id=check_id,
                    check_group=check_group,
                    phase=phase,
                    phase_id=phase_id,
                    summary=summary,
                    required=required,
                    min_visible=int(spec.get("min_visible") or 0),
                    gpu_inventory=gpu_inventory_cache,
                    base_details=details,
                )
            )
            continue

        if kind == "runbook_plan":
            surface_id = str(spec.get("surface") or "").strip()
            runbook_path = execution_surface_index[surface_id]
            checks.extend(
                build_runbook_plan_checks(
                    repo_root=repo_root,
                    targets=(
                        RunbookPlanCheckTarget(
                            check_id=check_id,
                            check_group=check_group,
                            phase=phase,
                            phase_id=phase_id,
                            runbook_path=runbook_path,
                            fallback_summary=summary,
                            required=required,
                            surface_id=surface_id,
                            details=details,
                        ),
                    ),
                    dependencies=RunbookPlanCheckDependencies(
                        execute_runbook_plan=execute_runbook_plan,
                        safe_json_loads=dependencies.safe_json_loads,
                        choose_command_summary=dependencies.choose_command_summary,
                    ),
                )
            )
            continue

        if kind == "command":
            surface_id = str(spec.get("surface") or "").strip()
            surface_payload = dict(contract.execution_surfaces.get(surface_id) or {})
            argv = tuple(str(token) for token in surface_payload.get("argv") or ())
            checks.extend(
                build_command_checks(
                    targets=(
                        CommandCheckTarget(
                            check_id=check_id,
                            check_group=check_group,
                            phase=phase,
                            phase_id=phase_id,
                            argv=argv,
                            cwd=_resolve_command_cwd(
                                surface_payload=surface_payload,
                                repo_root=repo_root,
                                study_root=study_root,
                            ),
                            fallback_summary=summary,
                            required=required,
                            surface_id=surface_id,
                            details=details,
                        ),
                    ),
                    dependencies=CommandCheckDependencies(
                        run_preflight_command=dependencies.run_preflight_command,
                        choose_command_summary=dependencies.choose_command_summary,
                    ),
                )
            )
            continue

        if kind == "scheduler_queue":
            surface_id = str(spec.get("surface") or "").strip()
            surface_payload = dict(contract.execution_surfaces.get(surface_id) or {})
            backend = str(surface_payload.get("backend") or "").strip()
            checks.extend(
                build_scheduler_queue_checks(
                    repo_root=repo_root,
                    targets=(
                        SchedulerQueueCheckTarget(
                            check_id=check_id,
                            check_group=check_group,
                            phase=phase,
                            phase_id=phase_id,
                            backend=backend,
                            max_running_jobs=int(spec.get("max_running_jobs") or 0),
                            max_queued_jobs=(
                                int(spec["max_queued_jobs"]) if spec.get("max_queued_jobs") is not None else None
                            ),
                            required=required,
                            surface_id=surface_id,
                            details=details,
                        ),
                    ),
                    dependencies=SchedulerQueueCheckDependencies(
                        run_preflight_command=dependencies.run_preflight_command,
                    ),
                )
            )
            continue

        raise ValueError(f"unsupported contract preflight check kind: {kind!r}")

    return tuple(checks)


def contract_environment_flag_state(
    *,
    contract: _StudyOpsContractLike,
    environ: Mapping[str, object | None],
    check_group: str | None = None,
) -> dict[str, bool]:
    requested_group = str(check_group or "").strip() or None
    ordered: dict[str, bool] = {}
    for specs in contract.preflight.check_specs.values():
        for spec in specs:
            if str(spec.get("kind") or "").strip() != "environment":
                continue
            spec_group = str(spec.get("check_group") or "").strip() or None
            if requested_group is not None and spec_group != requested_group:
                continue
            for env_var in spec.get("vars") or ():
                name = str(env_var).strip()
                if name and name not in ordered:
                    ordered[name] = bool(str(environ.get(name) or "").strip())
    return ordered


def _build_environment_check(
    *,
    check_id: str,
    check_group: str,
    phase: str,
    phase_id: str,
    summary: str,
    required: bool,
    vars_list: Sequence[str],
    match_mode: str,
    environ: Mapping[str, object | None],
    base_details: Mapping[str, object],
) -> PreflightCheck:
    flag_state = {name: bool(str(environ.get(name) or "").strip()) for name in vars_list}
    if match_mode == "all":
        matched = all(flag_state.values())
    else:
        matched = any(flag_state.values())
    missing_flags = [name for name, present in flag_state.items() if not present]
    resolved_summary = (
        summary
        if matched
        else _environment_attention_summary(
            missing_flags=missing_flags,
            vars_list=vars_list,
            match_mode=match_mode,
        )
    )
    return build_state_check(
        check_id=check_id,
        kind="environment",
        required=required,
        check_group=check_group,
        phase=phase,
        phase_id=phase_id,
        state="ok" if matched else "attention",
        summary=resolved_summary,
        details={
            **flag_state,
            **dict(base_details),
            "vars": list(vars_list),
            "match_mode": match_mode,
        },
    )


def _environment_attention_summary(
    *,
    missing_flags: Sequence[str],
    vars_list: Sequence[str],
    match_mode: str,
) -> str:
    if match_mode == "any":
        configured_names = ", ".join(vars_list)
        return f"None of the accepted environment variables are configured: {configured_names}."
    if len(missing_flags) == 1:
        return f"Required environment variable is not configured: {missing_flags[0]}."
    return "Required environment variables are not configured: " + ", ".join(missing_flags) + "."


def _build_gpu_availability_check(
    *,
    check_id: str,
    check_group: str,
    phase: str,
    phase_id: str,
    summary: str,
    required: bool,
    min_visible: int,
    gpu_inventory: Mapping[str, object],
    base_details: Mapping[str, object],
) -> PreflightCheck:
    visible_count = int(gpu_inventory.get("count") or 0)
    matched = visible_count >= min_visible
    resolved_summary = summary if matched else f"{summary.rstrip('.')} Visible GPUs: {visible_count}/{min_visible}."
    return build_state_check(
        check_id=check_id,
        kind="gpu_availability",
        required=required,
        check_group=check_group,
        phase=phase,
        phase_id=phase_id,
        state="ok" if matched else "attention",
        summary=resolved_summary,
        details={
            **dict(base_details),
            "min_visible": min_visible,
            "visible_count": visible_count,
            "gpu_inventory": dict(gpu_inventory),
        },
    )


def _build_workspace_layout_check(
    *,
    check_id: str,
    check_group: str,
    phase: str,
    phase_id: str,
    summary: str,
    required: bool,
    surface_id: str,
    execution_surface_index: Mapping[str, Path],
    base_details: Mapping[str, object],
) -> PreflightCheck:
    workspace_path = execution_surface_index[surface_id]
    exists = workspace_path.exists()
    is_dir = workspace_path.is_dir() if exists else False
    state = "ok" if exists and is_dir else "missing" if not exists else "attention"
    if state == "ok":
        resolved_summary = summary
    elif not exists:
        resolved_summary = f"{summary.rstrip('.')} Missing workspace root: {workspace_path}."
    else:
        resolved_summary = f"{summary.rstrip('.')} Workspace root is not a directory: {workspace_path}."
    return build_state_check(
        check_id=check_id,
        kind="workspace_layout",
        required=required,
        check_group=check_group,
        phase=phase,
        phase_id=phase_id,
        state=state,
        summary=resolved_summary,
        surface_id=surface_id,
        details={
            **dict(base_details),
            "workspace": str(workspace_path),
            "exists": exists,
            "is_dir": is_dir,
        },
    )


def _build_path_exists_check(
    *,
    check_id: str,
    check_group: str,
    phase: str,
    phase_id: str,
    summary: str,
    required: bool,
    artifact_id: str,
    contract: _StudyOpsContractLike,
    repo_root: Path,
    study_root: Path,
    dataset_index: Mapping[str, Mapping[str, object]],
    base_details: Mapping[str, object],
) -> PreflightCheck:
    artifact_state = _resolve_artifact_state(
        artifact_id=artifact_id,
        contract=contract,
        repo_root=repo_root,
        study_root=study_root,
        dataset_index=dataset_index,
    )
    exists = bool(artifact_state["exists"])
    resolved_summary = summary if exists else f"{summary.rstrip('.')} Missing artifact: {artifact_state['path']}."
    return build_state_check(
        check_id=check_id,
        kind="path_exists",
        required=required,
        check_group=check_group,
        phase=phase,
        phase_id=phase_id,
        state="ok" if exists else "missing",
        summary=resolved_summary,
        artifact_id=artifact_id,
        details={
            **dict(base_details),
            **artifact_state,
        },
    )


def _build_dataset_snapshot_check(
    *,
    check_id: str,
    check_group: str,
    phase: str,
    phase_id: str,
    summary: str,
    required: bool,
    artifact_id: str,
    target_rows: int,
    contract: _StudyOpsContractLike,
    repo_root: Path,
    study_root: Path,
    dataset_index: Mapping[str, Mapping[str, object]],
    base_details: Mapping[str, object],
) -> PreflightCheck:
    artifact_state = _resolve_artifact_state(
        artifact_id=artifact_id,
        contract=contract,
        repo_root=repo_root,
        study_root=study_root,
        dataset_index=dataset_index,
    )
    exists = bool(artifact_state["exists"])
    rows = artifact_state.get("rows")
    if not exists:
        state = "missing"
        resolved_summary = f"{summary.rstrip('.')} Missing dataset artifact: {artifact_state['path']}."
    elif rows is None:
        state = "attention"
        resolved_summary = f"{summary.rstrip('.')} Row count is not available."
    else:
        row_count = int(rows)
        if row_count >= target_rows:
            state = "ok"
            resolved_summary = summary
        else:
            state = "attention"
            resolved_summary = f"{summary.rstrip('.')} Current rows {row_count} below target {target_rows}."
    return build_state_check(
        check_id=check_id,
        kind="dataset_snapshot",
        required=required,
        check_group=check_group,
        phase=phase,
        phase_id=phase_id,
        state=state,
        summary=resolved_summary,
        artifact_id=artifact_id,
        details={
            **dict(base_details),
            **artifact_state,
            "target_rows": target_rows,
            "row_gap": max(target_rows - int(rows), 0) if isinstance(rows, int) else None,
        },
    )


def _build_sequence_view_contract_check(
    *,
    check_id: str,
    check_group: str,
    phase: str,
    phase_id: str,
    summary: str,
    required: bool,
    artifact_id: str,
    expected_payload: object,
    contract: _StudyOpsContractLike,
    repo_root: Path,
    study_root: Path,
    dataset_index: Mapping[str, Mapping[str, object]],
    base_details: Mapping[str, object],
) -> PreflightCheck:
    artifact_state = _resolve_artifact_state(
        artifact_id=artifact_id,
        contract=contract,
        repo_root=repo_root,
        study_root=study_root,
        dataset_index=dataset_index,
    )
    if not bool(artifact_state["exists"]):
        return build_state_check(
            check_id=check_id,
            kind="sequence_view_contract",
            required=required,
            check_group=check_group,
            phase=phase,
            phase_id=phase_id,
            state="missing",
            summary=f"{summary.rstrip('.')} Missing dataset artifact: {artifact_state['path']}.",
            artifact_id=artifact_id,
            details={**dict(base_details), **artifact_state},
        )
    dataset_id = str(artifact_state.get("dataset_id") or "").strip()
    dataset_state = dict(dataset_index.get(dataset_id) or {})
    usr_root = str(dataset_state.get("usr_root") or "").strip()
    if not dataset_id or not usr_root:
        return build_state_check(
            check_id=check_id,
            kind="sequence_view_contract",
            required=required,
            check_group=check_group,
            phase=phase,
            phase_id=phase_id,
            state="attention",
            summary=f"{summary.rstrip('.')} Dataset root could not be resolved for artifact {artifact_id}.",
            artifact_id=artifact_id,
            details={**dict(base_details), **artifact_state},
        )
    try:
        expectation = _sequence_view_expectation_from_payload(expected_payload)
        report = validate_sequence_view_contract(
            Dataset(Path(usr_root), dataset_id),
            expectation=expectation,
            raise_on_error=False,
        )
    except Exception as exc:
        return build_state_check(
            check_id=check_id,
            kind="sequence_view_contract",
            required=required,
            check_group=check_group,
            phase=phase,
            phase_id=phase_id,
            state="attention",
            summary=f"{summary.rstrip('.')} Sequence-view contract probe failed: {exc}",
            artifact_id=artifact_id,
            details={**dict(base_details), **artifact_state, "probe_error": str(exc)},
        )
    report_details = {
        "dataset": report.dataset,
        "total_records": report.total_records,
        "total_views": report.total_views,
        "counts_by_product_kind": report.counts_by_product_kind,
        "counts_by_orientation": report.counts_by_orientation,
        "counts_by_context_kind": report.counts_by_context_kind,
        "counts_by_recommended_pooling": report.counts_by_recommended_pooling,
        "invalid_bounds": report.invalid_bounds,
        "errors": list(report.errors),
    }
    return build_state_check(
        check_id=check_id,
        kind="sequence_view_contract",
        required=required,
        check_group=check_group,
        phase=phase,
        phase_id=phase_id,
        state="ok" if report.ok else "attention",
        summary=summary if report.ok else f"{summary.rstrip('.')} {len(report.errors)} contract error(s).",
        artifact_id=artifact_id,
        details={
            **dict(base_details),
            **artifact_state,
            **report_details,
        },
    )


def _build_infer_sequence_view_completion_check(
    *,
    check_id: str,
    check_group: str,
    phase: str,
    phase_id: str,
    summary: str,
    required: bool,
    surface_id: str,
    expected_payload: object,
    contract: _StudyOpsContractLike,
    repo_root: Path,
    study_root: Path,
    dependencies: ContractPreflightCheckDependencies,
    base_details: Mapping[str, object],
) -> PreflightCheck:
    surface_payload = dict(contract.execution_surfaces.get(surface_id) or {})
    argv = tuple(str(token) for token in surface_payload.get("argv") or ())
    execution = dependencies.run_preflight_command(
        argv,
        cwd=_resolve_command_cwd(
            surface_payload=surface_payload,
            repo_root=repo_root,
            study_root=study_root,
        ),
    )
    base_command_details = {
        **dict(base_details),
        "surface": surface_id,
    }
    if execution.returncode != 0 or execution.timed_out:
        return build_command_check(
            check_id=check_id,
            kind="infer_sequence_view_completion",
            required=required,
            check_group=check_group,
            phase=phase,
            phase_id=phase_id,
            summary=summary,
            execution=execution,
            surface_id=surface_id,
            details=base_command_details,
        )

    try:
        raw_payload = json.loads(execution.stdout or "null")
        plans = _infer_completion_plan_list(raw_payload)
        aggregate = _aggregate_infer_completion_plans(plans)
        expectation = _infer_completion_expectation_from_payload(expected_payload)
    except Exception as exc:
        return build_command_check(
            check_id=check_id,
            kind="infer_sequence_view_completion",
            required=required,
            check_group=check_group,
            phase=phase,
            phase_id=phase_id,
            summary=f"{summary.rstrip('.')} Infer completion planner output could not be parsed: {exc}",
            execution=execution,
            surface_id=surface_id,
            details={**base_command_details, "probe_error": str(exc)},
            override_state="attention",
        )

    violations = _infer_completion_threshold_violations(aggregate=aggregate, expectation=expectation)
    state = "attention" if violations else "ok"
    resolved_summary = (
        f"{summary.rstrip('.')}. "
        f"reusable={aggregate['reusable_vectors']} stale={aggregate['stale_vectors']} "
        f"missing={aggregate['missing_vectors']} missing_products={aggregate['missing_products']}."
    )
    return build_command_check(
        check_id=check_id,
        kind="infer_sequence_view_completion",
        required=required,
        check_group=check_group,
        phase=phase,
        phase_id=phase_id,
        summary=resolved_summary,
        execution=execution,
        surface_id=surface_id,
        details={
            **base_command_details,
            **aggregate,
            "thresholds": expectation,
            "violations": violations,
            "plans": plans,
        },
        override_state=state,
    )


def _infer_completion_plan_list(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("plans"), list):
        payload = payload["plans"]
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list or an object with a plans list.")
    plans: list[dict[str, object]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"plan entry {index} must be a mapping.")
        plans.append(dict(item))
    if not plans:
        raise ValueError("planner returned zero plans.")
    return plans


def _aggregate_infer_completion_plans(plans: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scalar_fields = (
        "required_views",
        "required_vectors",
        "existing_vectors",
        "reusable_vectors",
        "stale_vectors",
        "missing_vectors",
        "missing_products",
        "persisted_vector_reusable",
        "legacy_digest_reusable",
        "legacy_unclassified_vectors",
        "existing_aliases",
    )
    totals = {field: 0 for field in scalar_fields}
    product_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    pooling_counts: Counter[str] = Counter()
    command_lists: dict[str, list[str]] = {
        "construct_completion": [],
        "infer_backfill": [],
        "alias_backfill": [],
    }
    datasets: list[str] = []
    bundle_ids: list[str] = []
    model_families: list[str] = []
    for plan in plans:
        for field in scalar_fields:
            totals[field] += _required_int(plan.get(field, 0))
        _update_counter(product_counts, plan.get("by_product_kind"))
        _update_counter(orientation_counts, plan.get("by_orientation"))
        _update_counter(pooling_counts, plan.get("by_pooling_operation"))
        commands = plan.get("commands")
        if isinstance(commands, Mapping):
            for key in command_lists:
                command_lists[key].extend(str(item) for item in commands.get(key) or () if str(item).strip())
        dataset = str(plan.get("dataset") or "").strip()
        if dataset:
            datasets.append(dataset)
        bundle_id = str(plan.get("bundle_id") or "").strip()
        if bundle_id:
            bundle_ids.append(bundle_id)
        model_family = str(plan.get("model_family") or "").strip()
        if model_family:
            model_families.append(model_family)
    return {
        "plans_count": len(plans),
        "datasets": _ordered_unique(datasets),
        "bundle_ids": _ordered_unique(bundle_ids),
        "model_families": _ordered_unique(model_families),
        **totals,
        "counts_by_product_kind": dict(sorted(product_counts.items())),
        "counts_by_orientation": dict(sorted(orientation_counts.items())),
        "counts_by_pooling_operation": dict(sorted(pooling_counts.items())),
        "commands": {key: _ordered_unique(values) for key, values in command_lists.items()},
    }


def _update_counter(counter: Counter[str], payload: object) -> None:
    if not isinstance(payload, Mapping):
        return
    for raw_key, raw_value in payload.items():
        key = str(raw_key or "").strip()
        if key:
            counter[key] += _required_int(raw_value)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _infer_completion_expectation_from_payload(payload: object) -> dict[str, int]:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("infer_sequence_view_completion expected payload must be a mapping.")
    return {
        "max_missing_vectors": _optional_int(payload.get("max_missing_vectors")) or 0,
        "max_stale_vectors": _optional_int(payload.get("max_stale_vectors")) or 0,
        "max_missing_products": _optional_int(payload.get("max_missing_products")) or 0,
    }


def _infer_completion_threshold_violations(
    *,
    aggregate: Mapping[str, object],
    expectation: Mapping[str, int],
) -> list[str]:
    checks = (
        ("missing_vectors", "max_missing_vectors"),
        ("stale_vectors", "max_stale_vectors"),
        ("missing_products", "max_missing_products"),
    )
    violations: list[str] = []
    for observed_key, threshold_key in checks:
        observed = _required_int(aggregate.get(observed_key, 0))
        threshold = _required_int(expectation.get(threshold_key, 0))
        if observed > threshold:
            violations.append(f"{observed_key}={observed} exceeds {threshold_key}={threshold}")
    return violations


def _sequence_view_expectation_from_payload(payload: object) -> SequenceViewContractExpectation:
    if payload is None:
        return SequenceViewContractExpectation()
    if not isinstance(payload, Mapping):
        raise ValueError("sequence_view_contract expected payload must be a mapping.")
    return SequenceViewContractExpectation(
        total_records=_optional_int(payload.get("total_records")),
        total_views=_optional_int(payload.get("total_views")),
        counts_by_product_kind=_string_int_mapping(payload.get("counts_by_product_kind")),
        counts_by_orientation=_string_int_mapping(payload.get("counts_by_orientation")),
        counts_by_context_kind=_string_int_mapping(payload.get("counts_by_context_kind")),
        counts_by_recommended_pooling=_string_int_mapping(payload.get("counts_by_recommended_pooling")),
        exact_lengths_by_product_kind=_string_int_mapping(payload.get("exact_lengths_by_product_kind")),
        require_bounds_for_pooling=tuple(
            str(item).strip()
            for item in payload.get("require_bounds_for_pooling", ("anchor_mean",))
            if str(item).strip()
        ),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("sequence_view_contract integer expectations must not be boolean.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text else None


def _string_int_mapping(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("sequence_view_contract count expectations must be mappings.")
    out: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key:
            raise ValueError("sequence_view_contract count expectation keys must be non-empty.")
        out[key] = _required_int(raw_value)
    return out


def _required_int(value: object) -> int:
    resolved = _optional_int(value)
    if resolved is None:
        raise ValueError("sequence_view_contract count expectation values must be integers.")
    return resolved


def _resolve_artifact_state(
    *,
    artifact_id: str,
    contract: _StudyOpsContractLike,
    repo_root: Path,
    study_root: Path,
    dataset_index: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    artifact_payload = dict(contract.artifacts.get(artifact_id) or {})
    artifact_type = str(artifact_payload.get("artifact_type") or "path").strip() or "path"
    dataset_id = str(artifact_payload.get("dataset_id") or "").strip() or None
    if dataset_id is not None and dataset_id in dataset_index:
        dataset_state = dict(dataset_index[dataset_id] or {})
        path_text = str(dataset_state.get("records_path") or artifact_payload.get("ref") or "").strip()
        return {
            "artifact": artifact_id,
            "artifact_type": artifact_type,
            "dataset_id": dataset_id,
            "path": path_text,
            "exists": bool(dataset_state.get("exists")),
            "rows": dataset_state.get("rows"),
        }
    resolved_path = _resolve_artifact_path(
        artifact_payload=artifact_payload,
        repo_root=repo_root,
        study_root=study_root,
    )
    return {
        "artifact": artifact_id,
        "artifact_type": artifact_type,
        "dataset_id": dataset_id,
        "path": str(resolved_path),
        "exists": resolved_path.exists(),
        "rows": None,
    }


def _resolve_artifact_path(
    *,
    artifact_payload: Mapping[str, object],
    repo_root: Path,
    study_root: Path,
) -> Path:
    ref = str(artifact_payload.get("ref") or "").strip()
    if not ref:
        raise ValueError(f"artifact payload must define ref: {artifact_payload!r}")
    return resolve_path_ref(
        ref,
        repo_root=repo_root,
        manifest_dir=study_root,
        default_base="repo",
        label="artifact ref",
    )


def _resolve_command_cwd(
    *,
    surface_payload: Mapping[str, object],
    repo_root: Path,
    study_root: Path,
) -> Path:
    cwd_ref = str(surface_payload.get("cwd_ref") or "").strip()
    if not cwd_ref:
        return repo_root
    return resolve_path_ref(
        cwd_ref,
        repo_root=repo_root,
        manifest_dir=study_root,
        default_base="repo",
        label="command cwd_ref",
    )


def _phase_label(*, spec: Mapping[str, object], check_group: str, kind: str) -> str:
    explicit_phase = str(spec.get("phase") or "").strip()
    if explicit_phase:
        return explicit_phase
    if kind == "runbook_plan" or check_group.endswith("_plan"):
        return "ops"
    if kind == "scheduler_queue":
        return "scheduler"
    if check_group.startswith("notify"):
        return "notify"
    return check_group or kind


def _compile_contract_preflight_checks(
    *,
    check_specs: Mapping[str, Sequence[Mapping[str, object]]],
    enabled_groups: Sequence[str],
) -> tuple[_CompiledContractCheck, ...]:
    enabled_group_set = {str(group).strip() for group in enabled_groups if str(group).strip()}
    compiled_checks: list[_CompiledContractCheck] = []
    for declared_phase_id, specs in check_specs.items():
        for raw_spec in specs:
            spec = dict(raw_spec)
            check_group = str(spec.get("check_group") or "").strip()
            if not check_group or check_group not in enabled_group_set:
                continue
            kind = str(spec.get("kind") or "").strip()
            compiled_checks.append(
                _CompiledContractCheck(
                    check_id=str(spec.get("check_id") or "").strip(),
                    kind=kind,
                    check_group=check_group,
                    phase_id=str(spec.get("phase_id") or declared_phase_id).strip() or declared_phase_id,
                    phase=_phase_label(spec=spec, check_group=check_group, kind=kind),
                    summary=str(spec.get("summary") or "").strip(),
                    required=bool(spec.get("required", True)),
                    payload={
                        key: value
                        for key, value in spec.items()
                        if key not in {"check_id", "kind", "check_group", "phase_id", "phase", "summary", "required"}
                    },
                )
            )
    return tuple(compiled_checks)


__all__ = [
    "ContractPreflightCheckDependencies",
    "build_contract_preflight_checks",
    "contract_environment_flag_state",
]
