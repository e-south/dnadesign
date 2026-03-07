"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/execute.py

Fail-fast execution of orchestration plans with audit JSON emission.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .orchestration_notify import build_orchestration_notify_argv
from .plan import BatchPlan, CommandSpec

CommandRunner = Callable[[CommandSpec], tuple[int, str, str]]
_JOB_ID_PATTERN = re.compile(r"^\s*([0-9]+)(?:\D|$)")


@dataclass(frozen=True)
class CommandExecution:
    phase: str
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class BatchExecutionResult:
    ok: bool
    failed_phase: str | None
    commands: list[CommandExecution]
    audit_json_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "failed_phase": self.failed_phase,
            "audit_json_path": str(self.audit_json_path),
            "commands": [asdict(entry) for entry in self.commands],
        }


def _default_command_runner(command: CommandSpec, *, timeout_seconds: float | None) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.update(command.env)
    try:
        if command.argv is not None:
            result = subprocess.run(
                list(command.argv),
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
        else:
            assert command.shell is not None
            result = subprocess.run(
                ["bash", "-lc", command.shell],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "").strip()
        timeout_message = f"command timed out after {timeout_seconds} seconds"
        stderr = f"{stderr}\n{timeout_message}".strip() if stderr else timeout_message
        return 124, stdout, stderr
    return int(result.returncode), result.stdout, result.stderr


def _run_command(
    *,
    phase: str,
    command: CommandSpec,
    command_timeout_seconds: float | None,
    command_runner: CommandRunner | None,
    commands: list[CommandExecution],
) -> int:
    if command_runner is None:
        returncode, stdout, stderr = _default_command_runner(command, timeout_seconds=command_timeout_seconds)
    else:
        returncode, stdout, stderr = command_runner(command)
    commands.append(
        CommandExecution(
            phase=phase,
            command=command.render_shell(),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
    )
    return returncode


def _submit_job_ids(commands: list[CommandExecution]) -> tuple[str, ...]:
    job_ids: list[str] = []
    for command in commands:
        if command.phase != "submit" or command.returncode != 0:
            continue
        for line in command.stdout.splitlines():
            match = _JOB_ID_PATTERN.match(line.strip())
            if not match:
                continue
            job_id = match.group(1)
            if job_id not in job_ids:
                job_ids.append(job_id)
    return tuple(job_ids)


def _orchestration_notify_command(*, plan: BatchPlan, status: str, message: str) -> CommandSpec:
    notify = plan.orchestration_notify
    if notify is None:
        raise ValueError("orchestration notify command requested without notify contract")
    return CommandSpec(
        argv=build_orchestration_notify_argv(
            notify=notify,
            status=status,
            message=message,
        )
    )


def execute_batch_plan(
    *,
    plan: BatchPlan,
    audit_json_path: Path,
    submit: bool,
    command_timeout_seconds: float | None = None,
    command_runner: CommandRunner | None = None,
) -> BatchExecutionResult:
    if command_timeout_seconds is not None and command_timeout_seconds <= 0:
        raise ValueError("command_timeout_seconds must be positive when provided")
    commands: list[CommandExecution] = []
    failed_phase: str | None = None
    ok = True

    phase_map: list[tuple[str, list[CommandSpec]]] = [
        ("preflight", plan.preflight_commands),
        ("notify_smoke", plan.notify_smoke_commands),
    ]
    if submit:
        phase_map.append(("submit", plan.submit_commands))

    for phase, phase_commands in phase_map:
        if phase == "submit" and plan.submit_behavior == "blocked":
            failed_phase = "submit"
            ok = False
            break
        if phase == "submit" and plan.orchestration_notify is not None:
            started_message = f"ops submit requested workflow={plan.workflow_id} project={plan.project}"
            started_command = _orchestration_notify_command(
                plan=plan,
                status="started",
                message=started_message,
            )
            returncode = _run_command(
                phase="orchestration_notify",
                command=started_command,
                command_timeout_seconds=command_timeout_seconds,
                command_runner=command_runner,
                commands=commands,
            )
            if returncode != 0:
                failed_phase = "orchestration_notify"
                ok = False
                break
        for command in phase_commands:
            returncode = _run_command(
                phase=phase,
                command=command,
                command_timeout_seconds=command_timeout_seconds,
                command_runner=command_runner,
                commands=commands,
            )
            if returncode != 0:
                failed_phase = phase
                ok = False
                break
        if not ok:
            break

    if submit and plan.orchestration_notify is not None:
        if ok:
            submit_job_ids = _submit_job_ids(commands)
            job_ids_text = ",".join(submit_job_ids) if submit_job_ids else "none"
            final_status = "success"
            final_message = (
                f"ops submit accepted workflow={plan.workflow_id} project={plan.project} job_ids={job_ids_text}"
            )
        else:
            final_status = "failure"
            final_phase = failed_phase or "unknown"
            final_message = (
                f"ops orchestration failed workflow={plan.workflow_id} project={plan.project} phase={final_phase}"
            )
        final_notify_command = _orchestration_notify_command(
            plan=plan,
            status=final_status,
            message=final_message,
        )
        final_notify_returncode = _run_command(
            phase="orchestration_notify",
            command=final_notify_command,
            command_timeout_seconds=command_timeout_seconds,
            command_runner=command_runner,
            commands=commands,
        )
        if final_notify_returncode != 0 and ok:
            failed_phase = "orchestration_notify"
            ok = False

    audit_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan.as_dict(),
        "execution": {
            "ok": ok,
            "failed_phase": failed_phase,
            "commands": [asdict(entry) for entry in commands],
        },
    }
    audit_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return BatchExecutionResult(
        ok=ok,
        failed_phase=failed_phase,
        commands=commands,
        audit_json_path=audit_json_path,
    )
