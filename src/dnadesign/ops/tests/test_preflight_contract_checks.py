"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_preflight_contract_checks.py

Focused tests for generic OPS execution of contract-declared study preflight.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from dnadesign.ops.preflight import (
    CommandExecution,
    ContractPreflightCheckDependencies,
    build_command_check,
    build_contract_preflight_checks,
    choose_command_summary,
    contract_environment_flag_state,
    run_preflight_command,
)
from dnadesign.ops.preflight.models import supported_preflight_check_kinds
from dnadesign.ops.study import (
    StudyOpsContract,
    StudyPreflightContract,
)


def _execution(argv: tuple[str, ...], cwd: Path, *, returncode: int, stdout: str = "", stderr: str = "") -> object:
    return CommandExecution(
        argv=argv,
        cwd=str(cwd),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


def _contract() -> StudyOpsContract:
    return StudyOpsContract(
        study_id="demo_study",
        status_kind="demo-study-status",
        preflight_kind="demo-study-preflight",
        title="Demo study",
        snapshot_summary_scope="repo",
        execution_surfaces={
            "infer_batch_20b_anchor_only": {
                "surface_type": "runbook",
                "runbook_ref": "repo:workspace/runbooks/infer_anchor_only_20b.yaml",
            },
            "infer_batch_7b_anchor_only": {
                "surface_type": "runbook",
                "runbook_ref": "repo:workspace/runbooks/infer_anchor_only_7b.yaml",
            },
            "infer_validate_anchor_only_20b": {
                "surface_type": "command",
                "argv": [
                    "uv",
                    "run",
                    "infer",
                    "validate",
                    "config",
                    "--config",
                    "workspace/infer/config.anchor_only.evo2_20b.yaml",
                ],
            },
            "scheduler_default": {
                "surface_type": "scheduler",
                "backend": "sge",
            },
        },
        artifacts={
            "construct_context_dataset": {
                "artifact_type": "dataset",
                "dataset_id": "promoter/demo_construct_contexts",
                "ref": "repo:usr_root/promoter/demo_construct_contexts",
            }
        },
        preflight=StudyPreflightContract(
            default_scope="next",
            scope_groups={
                "next": ("infer", "notify_environment", "infer_batch_plan"),
                "full": ("infer", "notify_environment", "infer_batch_plan"),
            },
            check_specs={
                "infer_batch_preparation": (
                    {
                        "kind": "environment",
                        "check_id": "notify.environment.webhook",
                        "check_group": "notify_environment",
                        "summary": "Batch notify secret is configured in the environment.",
                        "required": False,
                        "vars": ["NOTIFY_WEBHOOK", "NOTIFY_WEBHOOK_FILE"],
                        "match_mode": "any",
                    },
                    {
                        "kind": "path_exists",
                        "check_id": "infer.construct.contexts",
                        "check_group": "infer",
                        "summary": "Construct contexts are present for infer.",
                        "required": True,
                        "artifact": "construct_context_dataset",
                    },
                    {
                        "kind": "command",
                        "check_id": "infer.validate.anchor_only_20b",
                        "check_group": "infer",
                        "summary": "Infer config validation completed.",
                        "required": True,
                        "surface": "infer_validate_anchor_only_20b",
                    },
                    {
                        "kind": "scheduler_queue",
                        "check_id": "infer.batch.queue",
                        "check_group": "infer_batch_plan",
                        "summary": "Scheduler queue is below the declared submit thresholds.",
                        "required": False,
                        "surface": "scheduler_default",
                        "max_running_jobs": 3,
                        "max_queued_jobs": 2,
                    },
                    {
                        "kind": "runbook_plan",
                        "check_id": "infer.batch.20b.anchor_only.plan",
                        "check_group": "infer_batch_plan",
                        "summary": "Anchor-only 20B infer runbook renders cleanly.",
                        "required": False,
                        "surface": "infer_batch_20b_anchor_only",
                    },
                    {
                        "kind": "runbook_plan",
                        "check_id": "infer.batch.7b.anchor_only.plan",
                        "check_group": "infer_batch_plan",
                        "summary": "Anchor-only 7B infer runbook renders cleanly.",
                        "required": False,
                        "surface": "infer_batch_7b_anchor_only",
                    },
                )
            },
        ),
        raw_payload={},
    )


def test_run_preflight_command_captures_native_gate_failure_text(tmp_path: Path) -> None:
    execution = run_preflight_command(
        (
            "uv",
            "run",
            "python",
            "-m",
            "dnadesign.ops.orchestrator.gates",
            "qa-submit-preflight",
            "--template",
            "/tmp/does-not-exist.qsub",
        ),
        cwd=tmp_path,
    )

    assert execution.returncode == 2
    assert execution.stdout == ""
    assert "template_missing=/tmp/does-not-exist.qsub" in execution.stderr


def test_run_preflight_command_binds_explicit_environment(tmp_path: Path) -> None:
    execution = run_preflight_command(
        (sys.executable, "-c", "import os; print(os.environ['DNADESIGN_USR_ROOT'])"),
        cwd=tmp_path,
        env={"DNADESIGN_USR_ROOT": str(tmp_path)},
    )

    assert execution.returncode == 0
    assert execution.stdout.strip() == str(tmp_path)


def test_run_preflight_command_binds_environment_for_native_gate(tmp_path: Path) -> None:
    qstat = tmp_path / "qstat"
    qstat.write_text(
        "#!/bin/sh\n"
        'test "$1" = -u || exit 91\n'
        'test "$2" = stress-operator || exit 92\n'
        "printf 'job-ID prior name user state submit/start at queue slots ja-task-ID\\n'\n",
        encoding="utf-8",
    )
    qstat.chmod(0o755)

    execution = run_preflight_command(
        (
            "uv",
            "run",
            "python",
            "-m",
            "dnadesign.ops.orchestrator.gates",
            "session-counts",
        ),
        cwd=tmp_path,
        env={"PATH": str(tmp_path), "USER": "stress-operator"},
    )

    assert execution.returncode == 0, execution.stderr
    assert execution.stdout == ("queue_probe=ok qstat_source=live running_jobs=0 queued_jobs=0 eqw_jobs=0\n")


def test_supported_preflight_kinds_are_generic_only() -> None:
    supported = supported_preflight_check_kinds()

    assert "scheduler_queue" in supported
    assert "command" in supported
    assert "infer_validate_config" not in supported
    assert "infer_local_runtime" not in supported
    assert "infer_dry_run" not in supported
    assert "notify_profile_doctor" not in supported
    assert "notify_resolve_events" not in supported


def test_build_command_check_omits_success_output_tails() -> None:
    check = build_command_check(
        check_id="infer.validate.anchor_only_20b",
        check_group="infer",
        category="infer",
        check_set_id="infer_batch_preparation",
        summary="Infer config validation completed.",
        execution=CommandExecution(
            argv=("uv", "run", "infer", "validate", "config"),
            cwd="/tmp/demo",
            returncode=0,
            stdout="rich table noise\n✔ Config validated.",
            stderr="",
            timed_out=False,
        ),
    )

    assert check.state == "ok"
    assert check.stdout_tail is None
    assert check.stderr_tail is None
    assert check.as_dict()["category"] == "infer"
    assert check.as_dict()["check_set_id"] == "infer_batch_preparation"
    assert "phase" not in check.as_dict()
    assert "phase_id" not in check.as_dict()


def test_build_command_check_keeps_failure_output_tails() -> None:
    check = build_command_check(
        check_id="notify.profile.anchor_only_20b",
        check_group="notify",
        category="notify",
        check_set_id="infer_batch_preparation",
        summary="Notify profile doctor completed.",
        execution=CommandExecution(
            argv=("uv", "run", "notify", "profile", "doctor"),
            cwd="/tmp/demo",
            returncode=1,
            stdout='{"ok": false, "error": "profile file not found"}',
            stderr="",
            timed_out=False,
        ),
    )

    assert check.state == "attention"
    assert check.stdout_tail == '{"ok": false, "error": "profile file not found"}'
    assert check.stderr_tail is None


def test_build_contract_preflight_checks_skips_non_enabled_groups(tmp_path: Path) -> None:
    contract = _contract()

    checks = build_contract_preflight_checks(
        repo_root=tmp_path,
        study_root=tmp_path / "docs" / "studies" / "demo_study",
        contract=contract,
        dataset_index={
            "promoter/demo_construct_contexts": {
                "dataset": "promoter/demo_construct_contexts",
                "exists": True,
                "records_path": str(tmp_path / "usr_root" / "promoter" / "demo_construct_contexts" / "records.parquet"),
                "rows": 2,
            }
        },
        execution_surface_index={
            "infer_batch_20b_anchor_only": tmp_path / "workspace" / "runbooks" / "infer_anchor_only_20b.yaml",
            "infer_batch_7b_anchor_only": tmp_path / "workspace" / "runbooks" / "infer_anchor_only_7b.yaml",
        },
        enabled_groups={"notify_environment"},
        environ={"NOTIFY_WEBHOOK": "", "NOTIFY_WEBHOOK_FILE": "", "SSL_CERT_FILE": ""},
        dependencies=ContractPreflightCheckDependencies(
            run_preflight_command=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("group-gated commands should not execute")
            ),
            execute_runbook_plan=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("group-gated runbook planning should not execute")
            ),
            safe_json_loads=lambda text: json.loads(text or "") if text else None,
            choose_command_summary=lambda *_args, fallback, **_kwargs: fallback,
            inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
        ),
    )

    assert [check.id for check in checks] == ["notify.environment.webhook"]
    assert checks[0].state == "attention"
    assert checks[0].check_group == "notify_environment"
    assert checks[0].details["match_mode"] == "any"
    assert checks[0].required is False
    assert (
        checks[0].summary
        == "None of the accepted environment variables are configured: NOTIFY_WEBHOOK, NOTIFY_WEBHOOK_FILE."
    )


def test_build_contract_preflight_checks_executes_declared_command_and_scheduler_surfaces(tmp_path: Path) -> None:
    contract = _contract()
    runbook_root = tmp_path / "workspace" / "runbooks"
    runbook_root.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[str, ...]] = []
    planned_runbooks: list[Path] = []

    def _run_progress_command(argv, *, cwd, timeout_seconds=180):
        del timeout_seconds
        commands.append(tuple(argv))
        if tuple(argv[:5]) == ("uv", "run", "infer", "validate", "config"):
            return _execution(tuple(argv), cwd, returncode=0, stdout="config ok")
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
                stdout="queue_probe=ok running_jobs=4 queued_jobs=1 eqw_jobs=0",
            )
        raise AssertionError(f"unexpected command: {' '.join(argv)}")

    def _execute_runbook_plan(*, runbook_path: Path, repo_root: Path) -> CommandExecution:
        planned_runbooks.append(runbook_path)
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
            returncode=2,
            stderr="Runbook contract error: notify webhook secret file is required for batch notify workflows",
        )

    checks = build_contract_preflight_checks(
        repo_root=tmp_path,
        study_root=tmp_path / "docs" / "studies" / "demo_study",
        contract=contract,
        dataset_index={
            "promoter/demo_construct_contexts": {
                "dataset": "promoter/demo_construct_contexts",
                "exists": True,
                "records_path": str(tmp_path / "usr_root" / "promoter" / "demo_construct_contexts" / "records.parquet"),
                "rows": 2,
            }
        },
        execution_surface_index={
            "infer_batch_20b_anchor_only": runbook_root / "infer_anchor_only_20b.yaml",
            "infer_batch_7b_anchor_only": runbook_root / "infer_anchor_only_7b.yaml",
        },
        enabled_groups={"infer", "infer_batch_plan"},
        environ={},
        dependencies=ContractPreflightCheckDependencies(
            run_preflight_command=_run_progress_command,
            execute_runbook_plan=_execute_runbook_plan,
            safe_json_loads=lambda text: json.loads(text or "") if text else None,
            choose_command_summary=choose_command_summary,
            inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
        ),
    )

    by_id = {check.id: check for check in checks}

    assert by_id["infer.validate.anchor_only_20b"].kind == "command"
    assert by_id["infer.validate.anchor_only_20b"].state == "ok"
    assert by_id["infer.validate.anchor_only_20b"].surface_id == "infer_validate_anchor_only_20b"
    assert by_id["infer.validate.anchor_only_20b"].summary == "Infer config validation completed."
    assert by_id["infer.batch.queue"].kind == "scheduler_queue"
    assert by_id["infer.batch.queue"].state == "attention"
    assert by_id["infer.batch.queue"].required is False
    assert by_id["infer.batch.queue"].details["running_jobs"] == 4
    assert by_id["infer.batch.20b.anchor_only.plan"].check_set_id == "infer_batch_preparation"
    assert by_id["infer.batch.7b.anchor_only.plan"].check_set_id == "infer_batch_preparation"
    assert by_id["infer.batch.20b.anchor_only.plan"].details["check_set_id"] == "infer_batch_preparation"
    assert "contract_phase_id" not in by_id["infer.batch.20b.anchor_only.plan"].details
    assert by_id["infer.batch.20b.anchor_only.plan"].state == "attention"
    assert (
        by_id["infer.batch.20b.anchor_only.plan"].summary
        == "Runbook contract error: notify webhook secret file is required for batch notify workflows"
    )
    assert commands == [
        (
            "uv",
            "run",
            "infer",
            "validate",
            "config",
            "--config",
            "workspace/infer/config.anchor_only.evo2_20b.yaml",
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
    assert planned_runbooks == [
        runbook_root / "infer_anchor_only_20b.yaml",
        runbook_root / "infer_anchor_only_7b.yaml",
    ]


def test_dataset_snapshot_check_can_require_exact_rows(tmp_path: Path) -> None:
    base = _contract()
    contract = replace(
        base,
        preflight=replace(
            base.preflight,
            check_specs={
                "infer_batch_preparation": (
                    {
                        "kind": "dataset_snapshot",
                        "check_id": "infer.construct.contexts.rows",
                        "check_group": "infer",
                        "summary": "Construct contexts have the expected exact row count.",
                        "required": True,
                        "artifact": "construct_context_dataset",
                        "target_rows": 2,
                        "row_count_mode": "exact",
                    },
                )
            },
        ),
    )

    checks = build_contract_preflight_checks(
        repo_root=tmp_path,
        study_root=tmp_path / "docs" / "studies" / "demo_study",
        contract=contract,
        dataset_index={
            "promoter/demo_construct_contexts": {
                "dataset": "promoter/demo_construct_contexts",
                "exists": True,
                "records_path": str(tmp_path / "usr_root" / "promoter" / "demo_construct_contexts" / "records.parquet"),
                "rows": 3,
            }
        },
        execution_surface_index={},
        enabled_groups={"infer"},
        environ={},
        dependencies=ContractPreflightCheckDependencies(
            run_preflight_command=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no commands expected")),
            safe_json_loads=lambda text: json.loads(text or "") if text else None,
            choose_command_summary=choose_command_summary,
            inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
        ),
    )

    assert len(checks) == 1
    assert checks[0].state == "attention"
    assert checks[0].summary == (
        "Construct contexts have the expected exact row count. Current rows 3 do not equal expected 2."
    )
    assert checks[0].details["row_count_mode"] == "exact"
    assert checks[0].details["row_delta"] == 1


def test_contract_environment_flag_state_reads_declared_environment_checks() -> None:
    contract = _contract()

    state = contract_environment_flag_state(
        contract=contract,
        environ={
            "NOTIFY_WEBHOOK": "",
            "NOTIFY_WEBHOOK_FILE": "/tmp/webhook",
            "SSL_CERT_FILE": "",
        },
        check_group="notify_environment",
    )

    assert state == {
        "NOTIFY_WEBHOOK": False,
        "NOTIFY_WEBHOOK_FILE": True,
    }
