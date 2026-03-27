"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/contract_checks.py

Generic execution for checked-in study preflight check specs.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.status.path_ref import resolve_path_ref
from dnadesign.studies.core.models import StudyOpsContract
from dnadesign.studies.core.preflight_plan import compile_study_preflight_execution_plan

from .check_protocols import CommandCheckTarget, RunbookPlanCheckTarget, SchedulerQueueCheckTarget
from .checks import (
    CommandCheckDependencies,
    RunbookPlanCheckDependencies,
    SchedulerQueueCheckDependencies,
    build_command_checks,
    build_runbook_plan_checks,
    build_scheduler_queue_checks,
)
from .models import CommandExecution, PreflightCheck, build_state_check

_ENVIRONMENT_MATCH_MODES = frozenset({"all", "any"})


@dataclass(frozen=True)
class ContractPreflightCheckDependencies:
    run_preflight_command: Callable[..., CommandExecution]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    choose_command_summary: Callable[..., str]
    inspect_local_gpu_inventory: Callable[[], dict[str, object]]


def build_contract_preflight_checks(
    *,
    repo_root: Path,
    study_root: Path,
    contract: StudyOpsContract,
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

    compiled_plan = compile_study_preflight_execution_plan(
        contract=contract.preflight,
        enabled_groups=tuple(enabled_group_set),
    )
    for compiled in compiled_plan.checks:
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
                        run_preflight_command=dependencies.run_preflight_command,
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
    contract: StudyOpsContract,
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
    contract: StudyOpsContract,
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
    contract: StudyOpsContract,
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


def _resolve_artifact_state(
    *,
    artifact_id: str,
    contract: StudyOpsContract,
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


__all__ = [
    "ContractPreflightCheckDependencies",
    "build_contract_preflight_checks",
    "contract_environment_flag_state",
]
