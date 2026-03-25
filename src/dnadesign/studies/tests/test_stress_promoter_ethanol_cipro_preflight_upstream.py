"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_stress_promoter_ethanol_cipro_preflight_upstream.py

Focused tests for the study-owned DenseGen and Construct preflight builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dnadesign.studies.stress_promoter_ethanol_cipro.preflight_upstream import (
    PromoterPreflightUpstreamDependencies,
    build_promoter_preflight_upstream_checks,
)


def _state_check(**kwargs) -> dict[str, object]:
    return {
        "id": kwargs["check_id"],
        "check_group": kwargs.get("check_group"),
        "phase": kwargs["phase"],
        "phase_id": kwargs["phase_id"],
        "state": kwargs["state"],
        "summary": kwargs["summary"],
        "details": kwargs.get("details", {}),
    }


def _command_check(**kwargs) -> dict[str, object]:
    execution = kwargs["execution"]
    return {
        "id": kwargs["check_id"],
        "check_group": kwargs.get("check_group"),
        "phase": kwargs["phase"],
        "phase_id": kwargs["phase_id"],
        "state": "attention" if getattr(execution, "returncode", 1) != 0 else "ok",
        "summary": kwargs["summary"],
        "details": kwargs.get("details", {}),
        "returncode": getattr(execution, "returncode", None),
    }


def _execution(argv: tuple[str, ...], cwd: Path, *, returncode: int, stdout: str = "", stderr: str = "") -> object:
    return SimpleNamespace(
        argv=argv,
        cwd=str(cwd),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


def test_build_promoter_preflight_upstream_checks_reports_densegen_probe_and_batch_plan(tmp_path: Path) -> None:
    study_repo_root = tmp_path
    runbook_path = tmp_path / "workspace" / "densegen_batch_with_notify.yaml"
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    densegen_config_path = runbook_path.parent / "densegen-config.yaml"
    commands: list[tuple[str, ...]] = []

    def _run_progress_command(argv, *, cwd, timeout_seconds=180):
        del timeout_seconds
        commands.append(tuple(argv))
        if "dense" in argv:
            return _execution(tuple(argv), cwd, returncode=0, stdout="solver ok")
        if "ops" in argv and "runbook" in argv:
            return _execution(
                tuple(argv),
                cwd,
                returncode=0,
                stdout=json.dumps(
                    {
                        "selected_mode": "resume",
                        "workflow_id": "densegen_batch_with_notify",
                        "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                    }
                ),
            )
        raise AssertionError(f"unexpected command: {' '.join(argv)}")

    result = build_promoter_preflight_upstream_checks(
        study_repo_root=study_repo_root,
        study_pipeline={},
        execution_surface_index={"densegen_batch_with_notify": runbook_path},
        dataset_index={},
        phase_states=[],
        densegen_phase_id="densegen_growth",
        construct_phase_id="construct_context_expansion",
        enabled_groups={"densegen"},
        dependencies=PromoterPreflightUpstreamDependencies(
            load_orchestration_runbook_payload=lambda _: {
                "densegen": {"config": "densegen-config.yaml"},
                "resources": {"queue": "gpu"},
            },
            resolve_input_path=lambda path, base_dir: (base_dir / path).resolve() if base_dir else path.resolve(),
            run_progress_command=_run_progress_command,
            safe_json_loads=lambda text: json.loads(text or "") if text else None,
            preflight_state_check=_state_check,
            preflight_command_check=_command_check,
            choose_command_summary=lambda *_args, fallback, **_kwargs: fallback,
        ),
    )

    checks = {check["id"]: check for check in result.checks}

    assert checks["densegen.batch.resources"]["state"] == "ok"
    assert checks["densegen.batch.resources"]["check_group"] == "densegen"
    assert checks["densegen.config.probe_solver"]["state"] == "ok"
    assert checks["densegen.config.probe_solver"]["details"]["config"] == str(densegen_config_path.resolve())
    assert checks["densegen.batch.plan"]["state"] == "ok"
    assert checks["densegen.batch.plan"]["details"]["selected_mode"] == "resume"
    assert checks["densegen.batch.plan"]["details"]["notify_secret_ref"] == "file:///tmp/webhook"
    assert commands == [
        ("uv", "run", "dense", "validate-config", "--probe-solver", "-c", str(densegen_config_path.resolve())),
        (
            "uv",
            "run",
            "ops",
            "runbook",
            "plan",
            "--runbook",
            str(runbook_path),
            "--repo-root",
            str(study_repo_root),
        ),
    ]


def test_build_promoter_preflight_upstream_checks_skips_construct_runtime_when_output_materialized(
    tmp_path: Path,
) -> None:
    study_repo_root = tmp_path
    workspace_path = tmp_path / "workspace" / "construct"
    workspace_path.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[str, ...]] = []

    def _run_progress_command(argv, *, cwd, timeout_seconds=180):
        del timeout_seconds
        commands.append(tuple(argv))
        if tuple(argv[:5]) == ("uv", "run", "construct", "workspace", "doctor"):
            return _execution(tuple(argv), cwd, returncode=0, stdout="workspace_doctor: ok")
        raise AssertionError(f"unexpected command: {' '.join(argv)}")

    result = build_promoter_preflight_upstream_checks(
        study_repo_root=study_repo_root,
        study_pipeline={
            "datasets": {
                "merged_anchor_dataset": "promoter/demo_anchor_set",
                "construct_context_dataset": "promoter/demo_construct_contexts",
            },
            "construct": {"workspace_projects": [{"id": "slot_a_window"}]},
        },
        execution_surface_index={"construct_workspace": workspace_path},
        dataset_index={
            "promoter/demo_anchor_set": {
                "exists": True,
                "records_path": str(tmp_path / "usr" / "promoter" / "demo_anchor_set" / "records.parquet"),
            },
            "promoter/demo_construct_contexts": {
                "exists": True,
                "records_path": str(tmp_path / "usr" / "promoter" / "demo_construct_contexts" / "records.parquet"),
                "rows": 2,
            },
        },
        phase_states=[{"id": "construct_context_expansion", "status": "complete"}],
        densegen_phase_id="densegen_growth",
        construct_phase_id="construct_context_expansion",
        enabled_groups={"construct"},
        dependencies=PromoterPreflightUpstreamDependencies(
            load_orchestration_runbook_payload=lambda _: {},
            resolve_input_path=lambda path, base_dir: (base_dir / path).resolve() if base_dir else path.resolve(),
            run_progress_command=_run_progress_command,
            safe_json_loads=lambda text: json.loads(text or "") if text else None,
            preflight_state_check=_state_check,
            preflight_command_check=_command_check,
            choose_command_summary=lambda *_args, fallback, **_kwargs: fallback,
        ),
    )

    checks = {check["id"]: check for check in result.checks}

    assert checks["construct.workspace.doctor"]["state"] == "ok"
    assert checks["construct.workspace.doctor"]["check_group"] == "construct"
    assert checks["construct.runtime.slot_a_window"]["state"] == "ok"
    assert checks["construct.runtime.slot_a_window"]["details"]["skipped_runtime_revalidation"] is True
    assert commands == [
        ("uv", "run", "construct", "workspace", "doctor", "--workspace", str(workspace_path)),
    ]
