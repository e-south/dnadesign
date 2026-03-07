"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/plan.py

Deterministic preflight/smoke/submit command-plan rendering for batch workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from dnadesign._contracts.notify_webhook_profile import parse_notify_profile_webhook, resolve_file_secret_ref_path
from dnadesign._contracts.tls_ca_bundle import (
    DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES,
    resolve_tls_ca_bundle_path,
)

from ..runbooks.path_policy import WORKSPACE_RUNTIME_LOGS_RELATIVE_DIR
from ..runbooks.schema import OrchestrationRunbookV1
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


def _notify_webhook_file_path_from_profile(profile_path: Path) -> str | None:
    if not profile_path.is_file():
        return None
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"notify profile is not valid JSON: {profile_path}") from exc
    except OSError as exc:
        raise ValueError(f"notify profile is not readable: {profile_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"notify profile root must be an object: {profile_path}")
    source, ref = parse_notify_profile_webhook(payload)
    if source != "secret_ref":
        return None
    path = resolve_file_secret_ref_path(
        ref,
        source_label=f"notify profile webhook secret_ref (profile={profile_path})",
    )
    return str(path)


def _notify_webhook_file_path(webhook_env_name: str, *, profile_path: Path | None = None) -> str | None:
    webhook_file_env_name = f"{webhook_env_name}_FILE"
    webhook_file = os.environ.get(webhook_file_env_name, "").strip()
    if not webhook_file:
        if profile_path is None:
            return None
        return _notify_webhook_file_path_from_profile(profile_path)
    return str(Path(webhook_file).expanduser().resolve())


def _require_notify_webhook_file_path(webhook_env_name: str, *, profile_path: Path | None = None) -> str:
    webhook_file_env_name = f"{webhook_env_name}_FILE"
    webhook_file = _notify_webhook_file_path(webhook_env_name, profile_path=profile_path)
    if webhook_file is None:
        profile_hint = ""
        if profile_path is not None:
            profile_hint = (
                f", or configure {profile_path} with webhook.source=secret_ref and a file:// secret reference"
            )
        raise ValueError(
            "notify webhook secret file is required for batch notify workflows. "
            f"Set {webhook_file_env_name} to a readable file path{profile_hint}."
        )
    if not os.path.isfile(webhook_file):
        raise ValueError(
            f"notify webhook secret file does not exist or is not a file: {webhook_file} (from {webhook_file_env_name})"
        )
    if not os.access(webhook_file, os.R_OK):
        raise ValueError(f"notify webhook secret file is not readable: {webhook_file} (from {webhook_file_env_name})")
    return webhook_file


def _notify_setup_secret_contract(secret_ref: str) -> tuple[str, ...]:
    return (
        "--secret-source",
        "file",
        "--secret-ref",
        secret_ref,
        "--no-store-webhook",
    )


@dataclass(frozen=True)
class NotifyRuntimeContract:
    webhook_file: str
    secret_ref: str
    tls_ca_bundle: str


def _resolve_notify_runtime_contract(
    webhook_env_name: str, *, profile_path: Path | None = None
) -> NotifyRuntimeContract:
    webhook_file = _require_notify_webhook_file_path(webhook_env_name, profile_path=profile_path)
    tls_ca_bundle = _resolve_tls_ca_bundle()
    return NotifyRuntimeContract(
        webhook_file=webhook_file,
        secret_ref=Path(webhook_file).as_uri(),
        tls_ca_bundle=tls_ca_bundle,
    )


@dataclass(frozen=True)
class OrchestrationNotifySpec:
    tool: str
    provider: str
    webhook_env: str
    secret_ref: str
    run_id: str
    tls_ca_bundle: str

    def as_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "provider": self.provider,
            "webhook_env": self.webhook_env,
            "secret_ref": self.secret_ref,
            "run_id": self.run_id,
            "tls_ca_bundle": self.tls_ca_bundle,
        }


def _resolve_tls_ca_bundle() -> str:
    return str(
        resolve_tls_ca_bundle_path(
            explicit_path=None,
            env_var_name="SSL_CERT_FILE",
            allow_system_candidates=True,
            system_candidates=DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES,
            not_configured_error=(
                "notify TLS CA bundle is not configured. "
                "Set SSL_CERT_FILE to a readable CA bundle path before running notify workflows."
            ),
            source_label="notify TLS CA bundle path",
        )
    )


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
    notify_runtime = _resolve_notify_runtime_contract(
        runbook.notify.webhook_env,
        profile_path=runbook.notify.profile,
    )
    secret_contract = _notify_setup_secret_contract(notify_runtime.secret_ref)
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
                "uv",
                "run",
                "notify",
                "send",
                "--status",
                "started",
                "--tool",
                runbook.notify.tool,
                "--run-id",
                "notify-smoke-live",
                "--provider",
                "slack",
                "--secret-ref",
                notify_runtime.secret_ref,
                "--message",
                "notify live smoke canary",
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
        notify_runtime = _resolve_notify_runtime_contract(
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
        notify_runtime = _resolve_notify_runtime_contract(
            runbook.notify.webhook_env,
            profile_path=runbook.notify.profile,
        )
        orchestration_notify = OrchestrationNotifySpec(
            tool=runbook.notify.tool,
            provider="slack",
            webhook_env=runbook.notify.webhook_env,
            secret_ref=notify_runtime.secret_ref,
            run_id=runbook.id,
            tls_ca_bundle=notify_runtime.tls_ca_bundle,
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
