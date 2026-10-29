"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_execution_audit_progress.py

Execution attempts reserve private audit storage and preserve uncertain outcomes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import subprocess
from dataclasses import replace

import pytest

from dnadesign.ops.orchestrator import execute
from dnadesign.ops.orchestrator.plan import CommandSpec
from dnadesign.ops.tests.test_command_failure_audit import _plan


def test_invalid_audit_destination_prevents_execution(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("fixture")
    calls = []
    with pytest.raises(OSError):
        execute.execute_batch_plan(
            plan=_plan(CommandSpec(argv=("example",))),
            audit_json_path=blocked / "audit.json",
            submit=False,
            command_runner=lambda command: (calls.append(command) or 0, "", ""),
        )
    assert not calls


def test_audit_is_private_and_marks_active_command_before_execution(tmp_path):
    path = tmp_path / "audit.json"

    def runner(command):
        payload = json.loads(path.read_text())["execution"]
        assert payload["status"] == "running"
        assert payload["active_command"]["phase"] == "preflight"
        assert payload["active_command"]["index"] == 0
        assert payload["commands"] == []
        assert path.stat().st_mode & 0o777 == 0o600
        return 0, "result", ""

    result = execute.execute_batch_plan(
        plan=_plan(CommandSpec(argv=("example",))),
        audit_json_path=path,
        submit=False,
        command_runner=runner,
    )
    assert result.ok
    saved = json.loads(path.read_text())["execution"]
    assert saved["status"] == "completed"
    assert saved["active_command"] is None


def test_interruption_preserves_progress_and_prevents_blind_retry(tmp_path):
    path = tmp_path / "audit.json"
    calls = []
    plan = replace(_plan(CommandSpec(argv=("first",))), notify_smoke_commands=[CommandSpec(argv=("second",))])

    def runner(command):
        calls.append(command.argv[0])
        if command.argv[0] == "second":
            raise KeyboardInterrupt()
        return 0, "completed-first", ""

    with pytest.raises(KeyboardInterrupt):
        execute.execute_batch_plan(plan=plan, audit_json_path=path, submit=False, command_runner=runner)
    saved = path.read_bytes()
    progress = json.loads(saved)["execution"]
    assert progress["commands"][0]["stdout"] == "completed-first"
    assert progress["active_command"]["phase"] == "notify_smoke"
    with pytest.raises(ValueError, match="Review the prior"):
        execute.execute_batch_plan(plan=plan, audit_json_path=path, submit=False, command_runner=runner)
    assert path.read_bytes() == saved
    assert calls == ["first", "second"]


@pytest.mark.parametrize(
    "failure, outcome",
    [
        (FileNotFoundError("missing"), "not_started"),
        (subprocess.TimeoutExpired("example", 1), "unknown"),
    ],
)
def test_launch_and_timeout_have_distinct_outcomes(tmp_path, monkeypatch, failure, outcome):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(execute.subprocess, "run", fail)
    path = tmp_path / "audit.json"
    result = execute.execute_batch_plan(plan=_plan(CommandSpec(argv=("example",))), audit_json_path=path, submit=False)
    assert not result.ok
    saved = json.loads(path.read_text())["execution"]
    assert saved["commands"][0]["outcome"] == outcome
    assert saved["retry_requires_review"] == (outcome == "unknown")


def test_completed_attempt_is_preserved_before_next_attempt(tmp_path):
    path = tmp_path / "audit.json"
    plan = _plan(CommandSpec(argv=("example",)))
    execute.execute_batch_plan(plan=plan, audit_json_path=path, submit=False, command_runner=lambda _: (0, "first", ""))
    original = path.read_bytes()
    path.chmod(0o644)
    execute.execute_batch_plan(
        plan=plan, audit_json_path=path, submit=False, command_runner=lambda _: (0, "second", "")
    )
    assert json.loads(path.read_text())["execution"]["commands"][0]["stdout"] == "second"
    histories = list((tmp_path / "audit.attempts").glob("*.json"))
    assert len(histories) == 1 and histories[0].read_bytes() == original
    assert all(file.stat().st_mode & 0o777 == 0o600 for file in [path, *histories])


def test_audit_symlink_is_rejected_without_touching_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("fixture")
    path = tmp_path / "audit.json"
    path.symlink_to(target)
    calls = []
    with pytest.raises((OSError, ValueError)):
        execute.execute_batch_plan(
            plan=_plan(CommandSpec(argv=("example",))),
            audit_json_path=path,
            submit=False,
            command_runner=lambda command: (calls.append(command) or 0, "", ""),
        )
    assert not calls and target.read_text() == "fixture"


def test_concurrent_attempt_cannot_replace_active_audit(tmp_path):
    path = tmp_path / "audit.json"
    plan = _plan(CommandSpec(argv=("example",)))
    concurrent_calls = []

    def runner(command):
        before = path.read_bytes()
        with pytest.raises(BlockingIOError):
            execute.execute_batch_plan(
                plan=plan,
                audit_json_path=path,
                submit=False,
                command_runner=lambda other: (concurrent_calls.append(other) or 0, "", ""),
            )
        assert path.read_bytes() == before
        return 0, "", ""

    execute.execute_batch_plan(plan=plan, audit_json_path=path, submit=False, command_runner=runner)
    assert not concurrent_calls


def test_failed_result_after_effects_requires_review_before_retry(tmp_path):
    path = tmp_path / "audit.json"
    plan = _plan(CommandSpec(argv=("example",)))
    calls = []

    def runner(command):
        calls.append(command)
        return 1, "partial output", "failed"

    result = execute.execute_batch_plan(plan=plan, audit_json_path=path, submit=False, command_runner=runner)
    assert not result.ok
    with pytest.raises(ValueError, match="Review the prior"):
        execute.execute_batch_plan(plan=plan, audit_json_path=path, submit=False, command_runner=runner)
    assert len(calls) == 1


def test_failed_progress_update_stops_before_next_command(tmp_path, monkeypatch):
    from dnadesign.ops.orchestrator import audit

    path = tmp_path / "audit.json"
    calls = []
    plan = replace(_plan(CommandSpec(argv=("first",))), notify_smoke_commands=[CommandSpec(argv=("second",))])
    original = audit._write_atomic

    def fail_after_result(path, payload):
        if payload["execution"]["commands"]:
            raise OSError("storage full")
        original(path, payload)

    monkeypatch.setattr(audit, "_write_atomic", fail_after_result)
    with pytest.raises(OSError, match="storage full"):
        execute.execute_batch_plan(
            plan=plan,
            audit_json_path=path,
            submit=False,
            command_runner=lambda command: (calls.append(command.argv[0]) or 0, "", ""),
        )
    assert calls == ["first"]
    assert json.loads(path.read_text())["execution"]["active_command"]["index"] == 0
