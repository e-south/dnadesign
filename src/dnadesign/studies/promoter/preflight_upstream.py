"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/promoter/preflight_upstream.py

Study-owned DenseGen and Construct preflight builders for the
promoter family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.preflight import (
    CommandCheckDependencies,
    CommandCheckTarget,
    CommandExecution,
    PreflightCheck,
    RunbookPlanCheckDependencies,
    RunbookPlanCheckTarget,
    build_command_checks,
    build_runbook_plan_checks,
    build_state_check,
)


@dataclass(frozen=True)
class PromoterPreflightUpstreamDependencies:
    load_orchestration_runbook_payload: Callable[[Path], dict[str, object]]
    resolve_input_path: Callable[[Path, Path | None], Path]
    run_preflight_command: Callable[..., CommandExecution]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    choose_command_summary: Callable[..., str]


@dataclass(frozen=True)
class PromoterPreflightUpstreamChecksResult:
    checks: tuple[PreflightCheck, ...]


def build_promoter_preflight_upstream_checks(
    *,
    study_repo_root: Path,
    study_pipeline: Mapping[str, object],
    execution_surface_index: Mapping[str, Path],
    dataset_index: Mapping[str, Mapping[str, object]],
    phase_states: Sequence[Mapping[str, object]],
    densegen_phase_id: str,
    construct_phase_id: str,
    enabled_groups: Collection[str],
    dependencies: PromoterPreflightUpstreamDependencies,
) -> PromoterPreflightUpstreamChecksResult:
    checks: list[PreflightCheck] = []
    if "densegen" in enabled_groups:
        checks.extend(
            _build_densegen_checks(
                study_repo_root=study_repo_root,
                execution_surface_index=execution_surface_index,
                densegen_phase_id=densegen_phase_id,
                dependencies=dependencies,
            )
        )
    if "construct" in enabled_groups:
        checks.extend(
            _build_construct_checks(
                study_repo_root=study_repo_root,
                study_pipeline=study_pipeline,
                execution_surface_index=execution_surface_index,
                dataset_index=dataset_index,
                phase_states=phase_states,
                construct_phase_id=construct_phase_id,
                dependencies=dependencies,
            )
        )
    return PromoterPreflightUpstreamChecksResult(checks=tuple(checks))


def _build_densegen_checks(
    *,
    study_repo_root: Path,
    execution_surface_index: Mapping[str, Path],
    densegen_phase_id: str,
    dependencies: PromoterPreflightUpstreamDependencies,
) -> tuple[PreflightCheck, ...]:
    densegen_batch_runbook = execution_surface_index.get("densegen_batch_with_notify")
    if densegen_batch_runbook is None:
        return ()

    checks: list[PreflightCheck] = []
    densegen_runbook = dependencies.load_orchestration_runbook_payload(densegen_batch_runbook)
    densegen_config_text = _string_or_none(((densegen_runbook.get("densegen") or {}).get("config")))
    densegen_resources = dict(densegen_runbook.get("resources") or {})
    checks.append(
        build_state_check(
            check_id="densegen.batch.resources",
            check_group="densegen",
            phase="densegen",
            phase_id=densegen_phase_id,
            state="ok",
            summary=(
                "densegen batch resources declared"
                if densegen_resources
                else "densegen batch resources missing from runbook"
            ),
            details={
                "runbook": str(densegen_batch_runbook),
                "resources": densegen_resources,
            },
        )
    )
    if densegen_config_text is not None:
        densegen_config_path = dependencies.resolve_input_path(
            Path(densegen_config_text),
            densegen_batch_runbook.parent,
        )
        checks.extend(
            build_command_checks(
                targets=(
                    CommandCheckTarget(
                        check_id="densegen.config.probe_solver",
                        check_group="densegen",
                        phase="densegen",
                        phase_id=densegen_phase_id,
                        argv=(
                            "uv",
                            "run",
                            "dense",
                            "validate-config",
                            "--probe-solver",
                            "-c",
                            str(densegen_config_path),
                        ),
                        cwd=study_repo_root,
                        fallback_summary="densegen config probe completed",
                        details={"config": str(densegen_config_path)},
                    ),
                ),
                dependencies=CommandCheckDependencies(
                    run_preflight_command=dependencies.run_preflight_command,
                    choose_command_summary=dependencies.choose_command_summary,
                ),
            )
        )

    checks.extend(
        build_runbook_plan_checks(
            repo_root=study_repo_root,
            targets=(
                RunbookPlanCheckTarget(
                    check_id="densegen.batch.plan",
                    check_group="densegen",
                    phase="densegen",
                    phase_id=densegen_phase_id,
                    runbook_path=densegen_batch_runbook,
                    fallback_summary="densegen batch plan completed",
                ),
            ),
            dependencies=RunbookPlanCheckDependencies(
                run_preflight_command=dependencies.run_preflight_command,
                safe_json_loads=dependencies.safe_json_loads,
                choose_command_summary=dependencies.choose_command_summary,
            ),
        )
    )
    return tuple(checks)


def _build_construct_checks(
    *,
    study_repo_root: Path,
    study_pipeline: Mapping[str, object],
    execution_surface_index: Mapping[str, Path],
    dataset_index: Mapping[str, Mapping[str, object]],
    phase_states: Sequence[Mapping[str, object]],
    construct_phase_id: str,
    dependencies: PromoterPreflightUpstreamDependencies,
) -> tuple[PreflightCheck, ...]:
    construct_workspace_path = execution_surface_index.get("construct_workspace")
    if construct_workspace_path is None:
        return ()

    checks: list[PreflightCheck] = []
    checks.extend(
        build_command_checks(
            targets=(
                CommandCheckTarget(
                    check_id="construct.workspace.doctor",
                    check_group="construct",
                    phase="construct",
                    phase_id=construct_phase_id,
                    argv=(
                        "uv",
                        "run",
                        "construct",
                        "workspace",
                        "doctor",
                        "--workspace",
                        str(construct_workspace_path),
                    ),
                    cwd=study_repo_root,
                    fallback_summary="construct workspace doctor completed",
                    details={"workspace": str(construct_workspace_path)},
                ),
            ),
            dependencies=CommandCheckDependencies(
                run_preflight_command=dependencies.run_preflight_command,
                choose_command_summary=dependencies.choose_command_summary,
            ),
        )
    )

    merged_anchor_dataset = _string_or_none(((study_pipeline.get("datasets") or {}).get("merged_anchor_dataset")))
    merged_anchor_state = dataset_index.get(merged_anchor_dataset or "") if merged_anchor_dataset else None
    construct_context_dataset = _string_or_none(
        ((study_pipeline.get("datasets") or {}).get("construct_context_dataset"))
    )
    construct_context_state = dataset_index.get(construct_context_dataset or "") if construct_context_dataset else None
    construct_phase_state = next(
        (phase for phase in phase_states if str(phase.get("id")) == "construct_context_expansion"),
        None,
    )
    construct_workspace_projects = list(((study_pipeline.get("construct") or {}).get("workspace_projects")) or [])
    for project_payload in construct_workspace_projects:
        if not isinstance(project_payload, dict):
            continue
        project_id = _string_or_none(project_payload.get("id"))
        if project_id is None:
            continue
        check_id = f"construct.runtime.{project_id}"
        if merged_anchor_state is None or not bool(merged_anchor_state.get("exists")):
            checks.append(
                build_state_check(
                    check_id=check_id,
                    check_group="construct",
                    phase="construct",
                    phase_id=construct_phase_id,
                    state="missing",
                    summary=(
                        f"requires merged anchor dataset {merged_anchor_dataset} before runtime validation"
                        if merged_anchor_dataset is not None
                        else "requires merged anchor dataset before runtime validation"
                    ),
                    details={
                        "dataset": merged_anchor_dataset,
                        "records_path": merged_anchor_state.get("records_path") if merged_anchor_state else None,
                        "workspace": str(construct_workspace_path),
                        "project": project_id,
                    },
                )
            )
            continue

        construct_details: dict[str, object] = {
            "workspace": str(construct_workspace_path),
            "project": project_id,
        }
        if (
            construct_context_dataset is not None
            and construct_context_state is not None
            and bool(construct_context_state.get("exists"))
            and str((construct_phase_state or {}).get("status") or "") == "complete"
        ):
            construct_details.update(
                {
                    "output_dataset": construct_context_dataset,
                    "records_path": construct_context_state.get("records_path"),
                    "rows": construct_context_state.get("rows"),
                    "skipped_runtime_revalidation": True,
                }
            )
            checks.append(
                build_state_check(
                    check_id=check_id,
                    check_group="construct",
                    phase="construct",
                    phase_id=construct_phase_id,
                    state="ok",
                    summary="construct output dataset is already materialized; skipping rerun runtime preflight",
                    details=construct_details,
                )
            )
            continue

        checks.extend(
            build_command_checks(
                targets=(
                    CommandCheckTarget(
                        check_id=check_id,
                        check_group="construct",
                        phase="construct",
                        phase_id=construct_phase_id,
                        argv=(
                            "uv",
                            "run",
                            "construct",
                            "workspace",
                            "validate-project",
                            "--workspace",
                            str(construct_workspace_path),
                            "--project",
                            project_id,
                            "--runtime",
                        ),
                        cwd=study_repo_root,
                        fallback_summary="construct runtime validation completed",
                        details=construct_details,
                    ),
                ),
                dependencies=CommandCheckDependencies(
                    run_preflight_command=dependencies.run_preflight_command,
                    choose_command_summary=dependencies.choose_command_summary,
                ),
            )
        )

    return tuple(checks)


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = [
    "PromoterPreflightUpstreamChecksResult",
    "PromoterPreflightUpstreamDependencies",
    "build_promoter_preflight_upstream_checks",
]
