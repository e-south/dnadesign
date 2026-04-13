"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_promoter_preflight.py

Focused tests for the study-owned preflight context coordination layer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dnadesign.ops.preflight import CommandExecution
from dnadesign.ops.preflight.models import supported_preflight_check_kinds
from dnadesign.studies.core.models import (
    StudyOpsContract,
    StudyPhaseContract,
    StudyPreflightContract,
    StudyPreflightNextScopeContract,
)
from dnadesign.studies.families.promoter.infer_runtime import PromoterStudyInferRuntimeDependencies
from dnadesign.studies.families.promoter.preflight import (
    PromoterPreflightContextDependencies,
    PromoterPreflightCoordinatorDependencies,
    PromoterPreflightResolvedContext,
    build_promoter_preflight_progress,
    resolve_promoter_preflight_context,
)
from dnadesign.studies.families.promoter.record_normalizer import PromoterStudyResolvedContext


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list_or_empty(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _string_or_none(item)
        if text is not None:
            result.append(text)
    return result


def _execution(argv: tuple[str, ...], cwd: Path, *, returncode: int, stdout: str = "", stderr: str = "") -> object:
    return CommandExecution(
        argv=argv,
        cwd=str(cwd),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


def test_resolve_promoter_preflight_context_uses_contract_scope_groups_and_runtime_lane_order(tmp_path: Path) -> None:
    study_repo_root = tmp_path
    infer_configs = {
        "anchor_only_20b": tmp_path / "config.anchor_only.evo2_20b.yaml",
        "anchor_plus_template_20b": tmp_path / "config.anchor_plus_template.evo2_20b.yaml",
        "anchor_only_7b": tmp_path / "config.anchor_only.evo2_7b.yaml",
    }
    study_context = PromoterStudyResolvedContext(
        study_dir_exists=True,
        requested_study_dir=None,
        resolved_study_dir=tmp_path / "docs" / "studies" / "demo_study",
        study_repo_root=study_repo_root,
        study_id="demo_study",
        selection_source="explicit",
        registry_path=tmp_path / "docs" / "studies" / "index.yaml",
        active_study="demo_study",
        required_paths={},
        missing_required_files=(),
        pipeline_path=tmp_path / "docs" / "studies" / "demo_study" / "pipeline.yaml",
        pipeline_present=True,
        datasets_entries=(),
        study_pipeline={
            "infer": {
                "preferred_model_family": "evo2_20b",
                "supported_model_families": ["evo2_20b", "evo2_7b"],
                "configs": infer_configs,
            }
        },
        canonical_usr_root_path=None,
        dataset_states=(),
        dataset_index={},
        missing_declared_present=(),
        present_but_planned=(),
        execution_surface_states=(),
        execution_surface_index={
            "infer_batch_7b_with_notify.anchor_only": tmp_path / "runbooks" / "anchor_only_7b.yaml",
            "infer_batch_20b_with_notify.anchor_plus_template": tmp_path / "runbooks" / "anchor_plus_template_20b.yaml",
            "infer_batch_20b_with_notify.anchor_only": tmp_path / "runbooks" / "anchor_only_20b.yaml",
            "densegen_batch_with_notify": tmp_path / "runbooks" / "densegen.yaml",
        },
        missing_execution_surfaces=(),
        phase_states=(
            {
                "id": "infer_anchor_only_20b",
                "status": "planned",
                "next_surface": "runbooks/anchor_only_20b.yaml",
            },
            {
                "id": "infer_anchor_plus_template_20b",
                "status": "planned",
                "next_surface": "runbooks/anchor_plus_template_20b.yaml",
            },
            {
                "id": "infer_anchor_only_7b",
                "status": "planned",
                "next_surface": "runbooks/anchor_only_7b.yaml",
            },
        ),
        current_phase="infer_batch_preparation",
        current_phase_is_known=True,
        next_ready_phase=None,
        next_in_progress_phase=None,
        next_planned_phase=None,
        blocked_phases=(),
        densegen_dataset_id=None,
        densegen_rows=None,
        densegen_row_target=None,
        densegen_row_gap=None,
        merged_anchor_dataset_id=None,
        merged_anchor_rows=None,
        construct_context_dataset_id=None,
        construct_context_rows=None,
        dataset_refresh_states=(),
        stale_dataset_ids=(),
        evidence={},
    )

    def _resolve_named_path_mapping(value, *, repo_root, label, status_kind):
        del repo_root, label, status_kind
        return {name: Path(path) for name, path in dict(value or {}).items()}

    def _resolve_infer_runtime_lane_contracts(config_paths, *, preferred_model_family):
        del preferred_model_family
        return (
            SimpleNamespace(
                phase_id="infer_anchor_only_20b",
                config_label="anchor_only_20b",
                runtime_label="anchor_only_20b",
                config_path=config_paths["anchor_only_20b"],
            ),
            SimpleNamespace(
                phase_id="infer_anchor_plus_template_20b",
                config_label="anchor_plus_template_20b",
                runtime_label="anchor_plus_template_20b",
                config_path=config_paths["anchor_plus_template_20b"],
            ),
            SimpleNamespace(
                phase_id="infer_anchor_only_7b",
                config_label="anchor_only_7b",
                runtime_label="anchor_only_7b",
                config_path=config_paths["anchor_only_7b"],
            ),
        )

    resolved = resolve_promoter_preflight_context(
        study_context=study_context,
        scope="next",
        status_kind="promoter-study-preflight",
        contract=StudyOpsContract(
            study_id="demo_study",
            family="promoter",
            phase_order=(
                "densegen_growth",
                "construct_context_expansion",
                "infer_batch_preparation",
                "infer_anchor_only_20b",
                "infer_anchor_plus_template_20b",
                "infer_anchor_only_7b",
            ),
            snapshot_summary_scope="repo",
            preflight=StudyPreflightContract(
                default_scope="next",
                group_phase_bindings={
                    "densegen": "densegen_growth",
                    "construct": "construct_context_expansion",
                    "notify_environment": "infer_batch_preparation",
                },
                next_scope=StudyPreflightNextScopeContract(
                    target_phase_groups={
                        "densegen_growth": ("densegen",),
                        "construct_context_expansion": ("construct",),
                        "infer_batch_preparation": (
                            "infer",
                            "notify_environment",
                            "notify",
                            "infer_batch_plan",
                        ),
                    },
                    runtime_phase_groups=("infer", "notify", "infer_batch_plan"),
                    runtime_shared_groups=("notify_environment",),
                ),
            ),
            current_phase_id="infer_batch_preparation",
            phases=(
                StudyPhaseContract(id="densegen_growth", status="parallel_optional"),
                StudyPhaseContract(id="construct_context_expansion", status="complete"),
                StudyPhaseContract(id="infer_batch_preparation", status="in_progress"),
                StudyPhaseContract(
                    id="infer_anchor_only_20b",
                    status="planned",
                    next_surface="runbooks/anchor_only_20b.yaml",
                ),
                StudyPhaseContract(
                    id="infer_anchor_plus_template_20b",
                    status="planned",
                    next_surface="runbooks/anchor_plus_template_20b.yaml",
                ),
                StudyPhaseContract(
                    id="infer_anchor_only_7b",
                    status="planned",
                    next_surface="runbooks/anchor_only_7b.yaml",
                ),
            ),
            raw_payload={},
        ),
        dependencies=PromoterPreflightContextDependencies(
            infer_runtime=PromoterStudyInferRuntimeDependencies(
                resolve_named_path_mapping=_resolve_named_path_mapping,
                resolve_infer_runtime_lane_contracts=_resolve_infer_runtime_lane_contracts,
                derive_infer_notify_profile_paths=lambda config_paths: (
                    {label: path.parent / "notify" / f"{label}.json" for label, path in config_paths.items()},
                    {},
                ),
                load_infer_model_summary=lambda config_path: {
                    "model_id": f"model-{config_path.stem}",
                    "device": "cuda:0",
                },
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            environ={},
        ),
    )

    assert resolved.scope_plan.scope == "next"
    assert resolved.scope_plan.target_phase_id == "infer_batch_preparation"
    assert resolved.scope_plan.included_groups == (
        "infer",
        "notify_environment",
        "notify",
        "infer_batch_plan",
    )
    assert tuple(resolved.infer_phase_targets) == (
        "infer_anchor_only_20b",
        "infer_anchor_plus_template_20b",
        "infer_anchor_only_7b",
    )


def test_build_promoter_preflight_progress_uses_only_contract_declared_generic_checks(tmp_path: Path) -> None:
    runbook_path = tmp_path / "runbooks" / "densegen.yaml"
    commands: list[tuple[str, ...]] = []
    contract = StudyOpsContract(
        study_id="demo_study",
        family="promoter",
        phase_order=("densegen_growth", "infer_batch_preparation"),
        snapshot_summary_scope="repo",
        execution_surfaces={
            "densegen_batch": {
                "surface_type": "runbook",
                "runbook_ref": "repo:runbooks/densegen.yaml",
            },
            "scheduler_default": {
                "surface_type": "scheduler",
                "backend": "sge",
            },
        },
        preflight=StudyPreflightContract(
            default_scope="next",
            group_phase_bindings={
                "densegen": "densegen_growth",
                "infer_batch_plan": "infer_batch_preparation",
            },
            next_scope=StudyPreflightNextScopeContract(
                target_phase_groups={"densegen_growth": ("densegen",)},
                runtime_phase_groups=("infer_batch_plan",),
                runtime_shared_groups=(),
            ),
            check_specs={
                "densegen_growth": (
                    {
                        "kind": "runbook_plan",
                        "check_id": "densegen.batch.plan",
                        "check_group": "densegen",
                        "summary": "DenseGen batch runbook renders cleanly.",
                        "required": True,
                        "surface": "densegen_batch",
                    },
                    {
                        "kind": "scheduler_queue",
                        "check_id": "densegen.batch.queue",
                        "check_group": "densegen",
                        "summary": "Scheduler queue is below the DenseGen submit threshold.",
                        "required": False,
                        "surface": "scheduler_default",
                        "max_running_jobs": 3,
                    },
                )
            },
        ),
        current_phase_id="densegen_growth",
        phases=(
            StudyPhaseContract(id="densegen_growth", status="in_progress"),
            StudyPhaseContract(id="infer_batch_preparation", status="planned"),
        ),
        raw_payload={},
    )

    def _run_progress_command(argv, *, cwd, timeout_seconds=180):
        del timeout_seconds
        commands.append(tuple(argv))
        if tuple(argv[:7]) == (
            "uv",
            "run",
            "python",
            "-m",
            "dnadesign.ops.orchestrator.gates",
            "session-counts",
            "--allow-missing-qstat",
        ):
            return _execution(
                tuple(argv),
                cwd,
                returncode=0,
                stdout="queue_probe=ok running_jobs=1 queued_jobs=0 eqw_jobs=0",
            )
        raise AssertionError(f"unexpected command: {' '.join(argv)}")

    def _execute_runbook_plan(*, runbook_path: Path, repo_root: Path) -> CommandExecution:
        commands.append(
            (
                "uv",
                "run",
                "ops",
                "runbook",
                "plan",
                "--runbook",
                str(runbook_path),
                "--repo-root",
                str(repo_root),
            )
        )
        return _execution(
            (
                "uv",
                "run",
                "ops",
                "runbook",
                "plan",
                "--runbook",
                str(runbook_path),
                "--repo-root",
                str(repo_root),
            ),
            repo_root,
            returncode=0,
            stdout='{"selected_mode":"resume"}',
        )

    state, _summary, evidence = build_promoter_preflight_progress(
        context=PromoterPreflightResolvedContext(
            contract=contract,
            study_id="demo_study",
            study_repo_root=tmp_path,
            resolved_study_dir=tmp_path / "docs" / "studies" / "demo_study",
            study_pipeline={},
            execution_surface_index={"densegen_batch": runbook_path},
            dataset_index={},
            phase_states=tuple(phase.as_dict() for phase in contract.phases),
            current_phase="densegen_growth",
            next_ready_phase=None,
            dataset_refresh_states=(),
            infer_runtime=SimpleNamespace(
                preferred_model_family=None,
                supported_model_families=(),
                infer_notify_profile_paths={},
                infer_notify_profile_errors={},
            ),
            infer_phase_targets={},
            scope_plan=SimpleNamespace(
                scope="next",
                target_phase_id="densegen_growth",
                included_groups=("densegen",),
                phase_scoped_groups=(),
            ),
        ),
        evidence={},
        dependencies=PromoterPreflightCoordinatorDependencies(
            run_preflight_command=_run_progress_command,
            execute_runbook_plan=_execute_runbook_plan,
            safe_json_loads=lambda text: {"selected_mode": "resume"} if text else None,
            choose_command_summary=lambda *_args, fallback, **_kwargs: fallback,
            inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
            environ={},
        ),
    )

    check_ids = [check["id"] for check in evidence["checks"]]
    check_kinds = {check["kind"] for check in evidence["checks"]}

    assert state == "ok"
    assert check_ids == ["densegen.batch.plan", "densegen.batch.queue"]
    assert check_kinds <= set(supported_preflight_check_kinds())
    assert commands == [
        (
            "uv",
            "run",
            "ops",
            "runbook",
            "plan",
            "--runbook",
            str(runbook_path),
            "--repo-root",
            str(tmp_path),
        ),
        (
            "uv",
            "run",
            "python",
            "-m",
            "dnadesign.ops.orchestrator.gates",
            "session-counts",
            "--allow-missing-qstat",
        ),
    ]


def test_build_promoter_preflight_progress_resolves_command_cwd_from_resolved_study_dir(tmp_path: Path) -> None:
    commands: list[tuple[tuple[str, ...], str]] = []
    resolved_study_dir = tmp_path / "study-records" / "demo_study"
    (resolved_study_dir / "workspace").mkdir(parents=True, exist_ok=True)
    contract = StudyOpsContract(
        study_id="demo_study",
        family="promoter",
        phase_order=("infer_batch_preparation",),
        snapshot_summary_scope="repo",
        execution_surfaces={
            "study_scoped_probe": {
                "surface_type": "command",
                "argv": ["uv", "run", "python", "-c", "print('ok')"],
                "cwd_ref": "manifest:workspace",
            },
        },
        preflight=StudyPreflightContract(
            default_scope="next",
            group_phase_bindings={"infer": "infer_batch_preparation"},
            next_scope=StudyPreflightNextScopeContract(
                target_phase_groups={"infer_batch_preparation": ("infer",)},
                runtime_phase_groups=(),
                runtime_shared_groups=(),
            ),
            check_specs={
                "infer_batch_preparation": (
                    {
                        "kind": "command",
                        "check_id": "study.scoped.probe",
                        "check_group": "infer",
                        "summary": "Study-scoped probe completed.",
                        "required": True,
                        "surface": "study_scoped_probe",
                    },
                )
            },
        ),
        current_phase_id="infer_batch_preparation",
        phases=(StudyPhaseContract(id="infer_batch_preparation", status="in_progress"),),
        raw_payload={},
    )

    def _run_progress_command(argv, *, cwd, timeout_seconds=180):
        del timeout_seconds
        commands.append((tuple(argv), str(cwd)))
        return _execution(tuple(argv), cwd, returncode=0, stdout="probe ok")

    state, _summary, evidence = build_promoter_preflight_progress(
        context=PromoterPreflightResolvedContext(
            contract=contract,
            study_id="demo_study",
            study_repo_root=tmp_path,
            resolved_study_dir=resolved_study_dir,
            study_pipeline={},
            execution_surface_index={},
            dataset_index={},
            phase_states=tuple(phase.as_dict() for phase in contract.phases),
            current_phase="infer_batch_preparation",
            next_ready_phase=None,
            dataset_refresh_states=(),
            infer_runtime=SimpleNamespace(
                preferred_model_family=None,
                supported_model_families=(),
                infer_notify_profile_paths={},
                infer_notify_profile_errors={},
            ),
            infer_phase_targets={},
            scope_plan=SimpleNamespace(
                scope="next",
                target_phase_id="infer_batch_preparation",
                included_groups=("infer",),
                phase_scoped_groups=(),
            ),
        ),
        evidence={},
        dependencies=PromoterPreflightCoordinatorDependencies(
            run_preflight_command=_run_progress_command,
            safe_json_loads=lambda text: {"ok": True} if text else None,
            choose_command_summary=lambda *_args, fallback, **_kwargs: fallback,
            inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
            environ={},
        ),
    )

    assert state == "ok"
    assert commands == [
        (
            ("uv", "run", "python", "-c", "print('ok')"),
            str(resolved_study_dir / "workspace"),
        )
    ]
    checks = {check["id"]: check for check in evidence["checks"]}
    assert checks["study.scoped.probe"]["cwd"] == str(resolved_study_dir / "workspace")
