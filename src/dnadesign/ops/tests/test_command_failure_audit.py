"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_command_failure_audit.py

Process failures retain a readable execution audit.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dnadesign.ops.orchestrator import execute
from dnadesign.ops.orchestrator.plan import BatchPlan, CommandSpec
from dnadesign.ops.orchestrator.state import OpsJobIdentity


def _plan(command):
    return BatchPlan(
        runbook_id="failure-test",
        workflow_id="failure-test",
        project="example",
        workspace_root="example",
        job_identity=OpsJobIdentity(*(["example"] * 5)),
        selected_mode="fresh",
        selected_smoke=None,
        submit_behavior="submit",
        hold_jid=None,
        preflight_commands=[command],
        notify_smoke_commands=[],
        submit_commands=[],
        orchestration_notify=None,
        decision_reason="failure-test",
    )


@pytest.mark.parametrize("stdout,stderr", [(b"started\n", b""), (b"", b"warning\n"), (b"bad\xff", b"bad\xfe")])
def test_timeout_output_is_text_in_failure_audit(tmp_path, monkeypatch, stdout, stderr):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("example", 0.1, output=stdout, stderr=stderr)

    monkeypatch.setattr(execute.subprocess, "run", timeout)
    audit = tmp_path / "audit.json"
    result = execute.execute_batch_plan(
        plan=_plan(CommandSpec(argv=("example",))),
        audit_json_path=audit,
        submit=False,
        command_timeout_seconds=0.1,
    )
    saved = json.loads(audit.read_text())["execution"]
    assert not result.ok and saved["failed_phase"] == "preflight"
    assert saved["commands"][0]["returncode"] == 124
    assert saved["commands"][0]["stdout"] == stdout.decode("utf-8", errors="replace")
    expected_stderr = stderr.decode("utf-8", errors="replace").strip()
    assert saved["commands"][0]["stderr"] == (expected_stderr + "\ncommand timed out after 0.1 seconds").strip()


@pytest.mark.parametrize(
    "failure,code", [(FileNotFoundError("missing executable"), 127), (PermissionError("not executable"), 126)]
)
def test_launch_failure_is_recorded(tmp_path, monkeypatch, failure, code):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(execute.subprocess, "run", fail)
    audit = tmp_path / "audit.json"
    result = execute.execute_batch_plan(plan=_plan(CommandSpec(argv=("example",))), audit_json_path=audit, submit=False)
    saved = json.loads(audit.read_text())["execution"]
    assert not result.ok
    assert saved["commands"][0]["returncode"] == code
    assert str(failure) in saved["commands"][0]["stderr"]


def test_real_timeout_with_output_preserves_diagnostic(tmp_path):
    command = CommandSpec(argv=(sys.executable, "-c", "import time; print('started', flush=True); time.sleep(2)"))
    result = execute.execute_batch_plan(
        plan=_plan(command), audit_json_path=tmp_path / "audit.json", submit=False, command_timeout_seconds=0.3
    )
    assert result.commands[0].returncode == 124
    assert result.commands[0].stdout == "started\n"


def test_audit_write_error_is_not_reported_as_success(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("audit destination denied")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(PermissionError, match="audit destination denied"):
        execute.execute_batch_plan(
            plan=_plan(CommandSpec(argv=("example",))),
            audit_json_path=tmp_path / "audit.json",
            submit=False,
            command_runner=lambda command: (1, "", "original failure"),
        )
