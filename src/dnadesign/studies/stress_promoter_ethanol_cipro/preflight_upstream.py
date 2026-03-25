"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/stress_promoter_ethanol_cipro/preflight_upstream.py

Study-owned DenseGen and Construct preflight builders for the
stress_promoter_ethanol_cipro family.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .preflight_orchestration import (
    PromoterPreflightRunbookPlanDependencies,
    PromoterPreflightRunbookPlanTarget,
    build_promoter_preflight_runbook_plan_checks,
)


@dataclass(frozen=True)
class PromoterPreflightUpstreamDependencies:
    load_orchestration_runbook_payload: Callable[[Path], dict[str, object]]
    resolve_input_path: Callable[[Path, Path | None], Path]
    run_progress_command: Callable[..., object]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    preflight_state_check: Callable[..., dict[str, object]]
    preflight_command_check: Callable[..., dict[str, object]]
    choose_command_summary: Callable[..., str]


@dataclass(frozen=True)
class PromoterPreflightUpstreamChecksResult:
    checks: tuple[dict[str, object], ...]


def build_promoter_preflight_upstream_checks(
    *,
    study_repo_root: Path,
    study_pipeline: Mapping[str, object],
    execution_surface_index: Mapping[str, Path],
    dataset_index: Mapping[str, Mapping[str, object]],
    phase_states: Sequence[Mapping[str, object]],
    densegen_phase_id: str,
    construct_phase_id: str,
    include_densegen_checks: bool,
    include_construct_checks: bool,
    dependencies: PromoterPreflightUpstreamDependencies,
) -> PromoterPreflightUpstreamChecksResult:
    checks: list[dict[str, object]] = []
    if include_densegen_checks:
        checks.extend(
            _build_densegen_checks(
                study_repo_root=study_repo_root,
                execution_surface_index=execution_surface_index,
                densegen_phase_id=densegen_phase_id,
                dependencies=dependencies,
            )
        )
    if include_construct_checks:
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
) -> tuple[dict[str, object], ...]:
    densegen_batch_runbook = execution_surface_index.get("densegen_batch_with_notify")
    if densegen_batch_runbook is None:
        return ()

    checks: list[dict[str, object]] = []
    densegen_runbook = dependencies.load_orchestration_runbook_payload(densegen_batch_runbook)
    densegen_config_text = _string_or_none(((densegen_runbook.get("densegen") or {}).get("config")))
    densegen_resources = dict(densegen_runbook.get("resources") or {})
    checks.append(
        dependencies.preflight_state_check(
            check_id="densegen.batch.resources",
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
        densegen_probe = dependencies.run_progress_command(
            ("uv", "run", "dense", "validate-config", "--probe-solver", "-c", str(densegen_config_path)),
            cwd=study_repo_root,
        )
        checks.append(
            dependencies.preflight_command_check(
                check_id="densegen.config.probe_solver",
                phase="densegen",
                phase_id=densegen_phase_id,
                summary=dependencies.choose_command_summary(
                    densegen_probe,
                    fallback="densegen config probe completed",
                ),
                execution=densegen_probe,
                details={"config": str(densegen_config_path)},
            )
        )

    checks.extend(
        build_promoter_preflight_runbook_plan_checks(
            study_repo_root=study_repo_root,
            targets=(
                PromoterPreflightRunbookPlanTarget(
                    check_id="densegen.batch.plan",
                    phase="densegen",
                    phase_id=densegen_phase_id,
                    runbook_path=densegen_batch_runbook,
                    fallback_summary="densegen batch plan completed",
                ),
            ),
            dependencies=PromoterPreflightRunbookPlanDependencies(
                run_progress_command=dependencies.run_progress_command,
                safe_json_loads=dependencies.safe_json_loads,
                preflight_command_check=dependencies.preflight_command_check,
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
) -> tuple[dict[str, object], ...]:
    construct_workspace_path = execution_surface_index.get("construct_workspace")
    if construct_workspace_path is None:
        return ()

    checks: list[dict[str, object]] = []
    construct_doctor = dependencies.run_progress_command(
        ("uv", "run", "construct", "workspace", "doctor", "--workspace", str(construct_workspace_path)),
        cwd=study_repo_root,
    )
    checks.append(
        dependencies.preflight_command_check(
            check_id="construct.workspace.doctor",
            phase="construct",
            phase_id=construct_phase_id,
            summary=dependencies.choose_command_summary(
                construct_doctor,
                fallback="construct workspace doctor completed",
            ),
            execution=construct_doctor,
            details={"workspace": str(construct_workspace_path)},
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
                dependencies.preflight_state_check(
                    check_id=check_id,
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
                dependencies.preflight_state_check(
                    check_id=check_id,
                    phase="construct",
                    phase_id=construct_phase_id,
                    state="ok",
                    summary="construct output dataset is already materialized; skipping rerun runtime preflight",
                    details=construct_details,
                )
            )
            continue

        construct_runtime = dependencies.run_progress_command(
            (
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
        )
        checks.append(
            dependencies.preflight_command_check(
                check_id=check_id,
                phase="construct",
                phase_id=construct_phase_id,
                summary=dependencies.choose_command_summary(
                    construct_runtime,
                    fallback="construct runtime validation completed",
                ),
                execution=construct_runtime,
                details=construct_details,
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
