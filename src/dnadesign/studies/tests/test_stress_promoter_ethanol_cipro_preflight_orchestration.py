"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_stress_promoter_ethanol_cipro_preflight_orchestration.py

Focused tests for the study-owned orchestration preflight builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dnadesign.studies.stress_promoter_ethanol_cipro.preflight_orchestration import (
    PromoterPreflightNotifyEnvironmentDependencies,
    PromoterPreflightRunbookPlanDependencies,
    PromoterPreflightRunbookPlanTarget,
    build_promoter_preflight_notify_environment_checks,
    build_promoter_preflight_runbook_plan_checks,
    resolve_notify_environment_state,
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


def test_build_promoter_preflight_notify_environment_checks_uses_explicit_state_contract() -> None:
    notify_env_state = resolve_notify_environment_state(
        environ={
            "NOTIFY_WEBHOOK": "",
            "NOTIFY_WEBHOOK_FILE": "/tmp/webhook",
            "SSL_CERT_FILE": "",
        }
    )

    checks = build_promoter_preflight_notify_environment_checks(
        notify_env_state=notify_env_state,
        notify_environment_phase_id="infer_batch_preparation",
        enabled_groups={"notify_environment"},
        dependencies=PromoterPreflightNotifyEnvironmentDependencies(
            preflight_state_check=_state_check,
        ),
    )

    by_id = {check["id"]: check for check in checks}
    assert by_id["notify.environment.webhook"]["state"] == "ok"
    assert by_id["notify.environment.tls"]["state"] == "attention"
    assert by_id["notify.environment.webhook"]["details"] == {
        "NOTIFY_WEBHOOK": False,
        "NOTIFY_WEBHOOK_FILE": True,
        "SSL_CERT_FILE": False,
    }


def test_build_promoter_preflight_runbook_plan_checks_merges_payload_and_details(tmp_path: Path) -> None:
    study_repo_root = tmp_path
    runbook_path = tmp_path / "workspace" / "infer_batch_with_notify.yaml"
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[str, ...]] = []

    def _run_progress_command(argv, *, cwd, timeout_seconds=180):
        del timeout_seconds
        commands.append(tuple(argv))
        return _execution(
            tuple(argv),
            cwd,
            returncode=0,
            stdout=json.dumps(
                {
                    "selected_mode": "resume",
                    "workflow_id": "infer_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                }
            ),
        )

    checks = build_promoter_preflight_runbook_plan_checks(
        study_repo_root=study_repo_root,
        targets=(
            PromoterPreflightRunbookPlanTarget(
                check_id="ops.runbook_plan.infer_batch_7b_with_notify.anchor_only",
                check_group="infer_batch_plan",
                phase="ops",
                phase_id="infer_anchor_only_7b",
                runbook_path=runbook_path,
                fallback_summary="ops runbook plan completed",
                details={"notify_env": {"NOTIFY_WEBHOOK": False}},
            ),
        ),
        dependencies=PromoterPreflightRunbookPlanDependencies(
            run_progress_command=_run_progress_command,
            safe_json_loads=lambda text: json.loads(text or "") if text else None,
            preflight_command_check=_command_check,
            choose_command_summary=lambda *_args, fallback, **_kwargs: fallback,
        ),
    )

    assert len(checks) == 1
    check = checks[0]
    assert check["state"] == "ok"
    assert check["check_group"] == "infer_batch_plan"
    assert check["details"]["runbook"] == str(runbook_path)
    assert check["details"]["notify_env"] == {"NOTIFY_WEBHOOK": False}
    assert check["details"]["selected_mode"] == "resume"
    assert check["details"]["workflow_id"] == "infer_batch_with_notify"
    assert check["details"]["notify_secret_ref"] == "file:///tmp/webhook"
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
            str(study_repo_root),
        )
    ]
