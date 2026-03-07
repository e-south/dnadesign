"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/plan.py

Deterministic preflight/smoke/submit command-plan rendering for batch workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Literal, Sequence

from ..runbooks.path_policy import WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR
from ..runbooks.schema import OrchestrationRunbookV1
from .orchestration_notify import (
    OrchestrationNotifySpec,
    build_notify_setup_secret_contract,
    build_orchestration_notify_argv,
    build_orchestration_notify_spec,
    resolve_notify_runtime_contract,
)
from .plan_tools import ToolCommandSpec, resolve_plan_tool_adapter
from .state import ModeDecision, resolve_mode_decision

SmokeMode = Literal["dry", "live"]


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...] | None = None
    shell: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_argv = bool(self.argv)
        has_shell = bool((self.shell or "").strip())
        if has_argv == has_shell:
            raise ValueError("command spec requires exactly one of argv or shell")

    def render_shell(self) -> str:
        env_prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in sorted(self.env.items()))
        if self.argv is not None:
            command_text = " ".join(shlex.quote(part) for part in self.argv)
        else:
            assert self.shell is not None
            command_text = f"bash -lc {shlex.quote(self.shell)}"
        return f"{env_prefix} {command_text}".strip()

    def as_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv) if self.argv is not None else None,
            "shell": self.shell,
            "env": dict(sorted(self.env.items())),
            "rendered": self.render_shell(),
        }


def _argv_command(*parts: object, env: dict[str, str] | None = None) -> CommandSpec:
    return CommandSpec(argv=tuple(str(part) for part in parts), env=env or {})


def _shell_command(command_text: str, *, env: dict[str, str] | None = None) -> CommandSpec:
    return CommandSpec(shell=command_text, env=env or {})


def _ops_gate_command(*parts: object) -> CommandSpec:
    return _argv_command("uv", "run", "python", "-m", "dnadesign.ops.orchestrator.gates", *parts)


def _render_tool_command(command: ToolCommandSpec) -> CommandSpec:
    if command.kind == "ops_gate":
        return _ops_gate_command(*command.parts)
    return _argv_command(*command.parts, env=command.env)


@dataclass(frozen=True)
class BatchPlan:
    workflow_id: str
    project: str
    selected_mode: str
    selected_smoke: SmokeMode | None
    submit_behavior: str
    hold_jid: str | None
    preflight_commands: list[CommandSpec]
    notify_smoke_commands: list[CommandSpec]
    submit_commands: list[CommandSpec]
    orchestration_notify: OrchestrationNotifySpec | None
    decision_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "project": self.project,
            "selected_mode": self.selected_mode,
            "selected_smoke": self.selected_smoke,
            "submit_behavior": self.submit_behavior,
            "hold_jid": self.hold_jid,
            "preflight_commands": [command.as_dict() for command in self.preflight_commands],
            "notify_smoke_commands": [command.as_dict() for command in self.notify_smoke_commands],
            "submit_commands": [command.as_dict() for command in self.submit_commands],
            "orchestration_notify": (
                self.orchestration_notify.as_dict() if self.orchestration_notify is not None else None
            ),
            "decision_reason": self.decision_reason,
        }


def _stdout_file_path(runbook: OrchestrationRunbookV1) -> str:
    return f"{runbook.logging.stdout_dir}/$JOB_NAME.$JOB_ID.out"


def _runtime_to_seconds(h_rt: str) -> int:
    hours_str, minutes_str, seconds_str = h_rt.split(":", maxsplit=2)
    return (int(hours_str) * 3600) + (int(minutes_str) * 60) + int(seconds_str)


def _preflight_commands(
    runbook: OrchestrationRunbookV1,
    *,
    mode_decision: ModeDecision,
    planned_submits: int,
    requires_order: bool,
) -> list[CommandSpec]:
    stdout_dir = str(runbook.logging.stdout_dir)
    runtime_log_dir = str((runbook.workspace_root / WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR).resolve())
    retention_manifest_path = str((runbook.logging.stdout_dir / "retention-manifest.json").resolve())
    runtime_retention_manifest_path = str(
        ((runbook.workspace_root / WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR) / "retention-manifest.json").resolve()
    )
    stdout_file = _stdout_file_path(runbook)
    shape_command_parts: list[str] = [
        "submit-shape-advisor",
        "--planned-submits",
        str(planned_submits),
        "--warn-over-running",
        "3",
    ]
    if requires_order:
        shape_command_parts.append("--requires-order")

    shared = [
        _ops_gate_command(
            "ensure-dir-writable",
            "--path",
            stdout_dir,
        ),
        _ops_gate_command(
            "ensure-dir-writable",
            "--path",
            runtime_log_dir,
        ),
        _ops_gate_command(
            "prune-ops-logs",
            "--stdout-dir",
            stdout_dir,
            "--log-kind",
            "sge",
            "--runbook-id",
            runbook.id,
            "--keep-last",
            str(runbook.logging.retention.keep_last),
            "--max-age-days",
            str(runbook.logging.retention.max_age_days),
            "--manifest-path",
            retention_manifest_path,
            "--json",
        ),
        _ops_gate_command(
            "prune-ops-logs",
            "--stdout-dir",
            runtime_log_dir,
            "--log-kind",
            "runtime",
            "--runbook-id",
            runbook.id,
            "--keep-last",
            str(runbook.logging.retention.keep_last),
            "--max-age-days",
            str(runbook.logging.retention.max_age_days),
            "--manifest-path",
            runtime_retention_manifest_path,
            "--json",
        ),
        _ops_gate_command("session-counts"),
    ]
    if runbook.notify is not None:
        notify_template = str(runbook.notify.qsub_template)
        shared.extend(
            [
                _argv_command(
                    "qsub",
                    "-verify",
                    "-P",
                    runbook.project,
                    "-o",
                    stdout_file,
                    "-l",
                    f"h_rt={runbook.resources.h_rt}",
                    notify_template,
                ),
                _ops_gate_command("qa-submit-preflight", "--template", notify_template),
            ]
        )

    tool_adapter = resolve_plan_tool_adapter(runbook)
    tool_commands = [
        _render_tool_command(command)
        for command in tool_adapter.build_preflight_commands(runbook, mode_decision, stdout_file)
    ]
    return [
        *shared,
        *tool_commands,
        _ops_gate_command(*shape_command_parts),
        _ops_gate_command(
            "operator-brief",
            "--planned-submits",
            str(planned_submits),
            "--warn-over-running",
            "3",
        ),
    ]


def _notify_smoke_commands(
    runbook: OrchestrationRunbookV1,
    *,
    smoke_mode: SmokeMode | None,
) -> list[CommandSpec]:
    if runbook.notify is None:
        return []

    tool_adapter = resolve_plan_tool_adapter(runbook)
    config = str(tool_adapter.notify_config_path(runbook))
    resolve_tool = tool_adapter.tool
    notify_runtime = resolve_notify_runtime_contract(
        runbook.notify.webhook_env,
        profile_path=runbook.notify.profile,
    )
    secret_contract = build_notify_setup_secret_contract(notify_runtime.secret_ref)
    commands = [
        _argv_command(
            "uv",
            "run",
            "notify",
            "profile",
            "smoke",
            "--profile",
            str(runbook.notify.profile),
            "--tool",
            resolve_tool,
            "--config",
            config,
            "--cursor",
            str(runbook.notify.cursor),
            "--spool-dir",
            str(runbook.notify.spool_dir),
            "--policy",
            runbook.notify.policy,
            *secret_contract,
            "--tls-ca-bundle",
            notify_runtime.tls_ca_bundle,
            "--only-tools",
            runbook.notify.tool,
            "--dry-run",
            "--no-advance-cursor-on-dry-run",
        ),
    ]
    if smoke_mode == "live":
        commands.append(
            _argv_command(
                *build_orchestration_notify_argv(
                    notify=OrchestrationNotifySpec(
                        tool=runbook.notify.tool,
                        provider="slack",
                        webhook_env=runbook.notify.webhook_env,
                        secret_ref=notify_runtime.secret_ref,
                        run_id="notify-smoke-live",
                        tls_ca_bundle=notify_runtime.tls_ca_bundle,
                    ),
                    status="started",
                    message="notify live smoke canary",
                )
            )
        )
    return commands


def _submit_commands(runbook: OrchestrationRunbookV1, *, mode_decision: ModeDecision) -> list[CommandSpec]:
    if mode_decision.submit_behavior == "blocked":
        return []

    stdout_file = _stdout_file_path(runbook)
    submit_commands: list[CommandSpec] = []
    if runbook.notify is not None:
        webhook_env_name = runbook.notify.webhook_env
        notify_runtime = resolve_notify_runtime_contract(
            webhook_env_name,
            profile_path=runbook.notify.profile,
        )
        notify_idle_timeout_seconds = str(_runtime_to_seconds(runbook.resources.h_rt))
        notify_submit_parts = [
            f"NOTIFY_PROFILE={runbook.notify.profile}",
            f"WEBHOOK_ENV={webhook_env_name}",
            f"NOTIFY_IDLE_TIMEOUT_SECONDS={notify_idle_timeout_seconds}",
            "NOTIFY_ENFORCE_TERMINAL_ON_IDLE=1",
            f"NOTIFY_TLS_CA_BUNDLE={notify_runtime.tls_ca_bundle}",
            f"WEBHOOK_FILE={notify_runtime.webhook_file}",
        ]
        notify_submit_env = ",".join(notify_submit_parts)
        submit_commands.append(
            _argv_command(
                "qsub",
                "-terse",
                "-P",
                runbook.project,
                "-o",
                stdout_file,
                "-l",
                f"h_rt={runbook.resources.h_rt}",
                "-v",
                notify_submit_env,
                str(runbook.notify.qsub_template),
            )
        )

    hold_fragment: list[str] = []
    if mode_decision.submit_behavior == "hold_jid" and mode_decision.hold_jid:
        hold_fragment = ["-hold_jid", mode_decision.hold_jid]

    tool_adapter = resolve_plan_tool_adapter(runbook)
    submit_commands.extend(
        _render_tool_command(command)
        for command in tool_adapter.build_submit_commands(
            runbook,
            mode_decision,
            stdout_file,
            tuple(hold_fragment),
        )
    )
    return submit_commands


def build_batch_plan(
    *,
    runbook: OrchestrationRunbookV1,
    requested_mode: Literal["auto", "fresh", "resume"] | None,
    requested_smoke: SmokeMode | None,
    active_job_ids: Sequence[str],
    allow_fresh_reset: bool = False,
) -> BatchPlan:
    if runbook.notify is None and requested_smoke is not None:
        raise ValueError("notify smoke override is not valid when runbook.notify is absent")
    resolve_plan_tool_adapter(runbook).validate_resources(runbook)
    selected_smoke: SmokeMode | None = requested_smoke or (runbook.notify.smoke if runbook.notify is not None else None)
    mode_decision = resolve_mode_decision(
        runbook=runbook,
        requested_mode=requested_mode,
        active_job_ids=active_job_ids,
        allow_fresh_reset=allow_fresh_reset,
    )
    submit_commands = _submit_commands(runbook, mode_decision=mode_decision)
    fallback_submits = 2 if runbook.notify is not None else 1
    planned_submits = len(submit_commands) or fallback_submits
    requires_order = mode_decision.submit_behavior == "hold_jid"
    orchestration_notify: OrchestrationNotifySpec | None = None
    if runbook.notify is not None and runbook.notify.orchestration_events:
        orchestration_notify = build_orchestration_notify_spec(
            tool=runbook.notify.tool,
            provider="slack",
            webhook_env=runbook.notify.webhook_env,
            run_id=runbook.id,
            profile_path=runbook.notify.profile,
        )
    return BatchPlan(
        workflow_id=runbook.workflow_id,
        project=runbook.project,
        selected_mode=mode_decision.selected_mode,
        selected_smoke=selected_smoke,
        submit_behavior=mode_decision.submit_behavior,
        hold_jid=mode_decision.hold_jid,
        preflight_commands=_preflight_commands(
            runbook,
            mode_decision=mode_decision,
            planned_submits=planned_submits,
            requires_order=requires_order,
        ),
        notify_smoke_commands=_notify_smoke_commands(runbook, smoke_mode=selected_smoke),
        submit_commands=submit_commands,
        orchestration_notify=orchestration_notify,
        decision_reason=mode_decision.reason,
    )
