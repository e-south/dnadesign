"""
CLI formatting and error helpers for latentdna.
"""

from __future__ import annotations

import json
from typing import Any

import typer
import yaml

from ..contracts.errors import exit_code_for_error


def _render_quiet_command_result(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "ok"))
    artifact_kind = payload.get("artifact_kind")
    artifact_id = payload.get("artifact_id")
    label = (
        f"{artifact_kind}:{artifact_id}" if artifact_kind and artifact_id else str(payload.get("command", "command"))
    )
    summary = f"{status}: {label}"
    if payload.get("dry_run"):
        summary += " (dry-run)"
    outputs = payload.get("outputs") or []
    if outputs:
        summary += f" -> {outputs[0]}"
    return summary


def render_payload(payload: Any, *, format_name: str, quiet: bool = False) -> str:
    if quiet and format_name == "text" and isinstance(payload, dict):
        if payload.get("schema_version") == "latentdna.command_result.v1":
            return _render_quiet_command_result(payload)
    if format_name == "json":
        return json.dumps(payload, sort_keys=False)
    if format_name == "yaml":
        return yaml.safe_dump(payload, sort_keys=False)
    if isinstance(payload, dict):
        return "\n".join(f"{key}: {value}" for key, value in payload.items())
    return str(payload)


def emit(payload: Any, *, format_name: str, quiet: bool = False) -> None:
    rendered = render_payload(payload, format_name=format_name, quiet=quiet)
    if rendered:
        typer.echo(rendered)


def resolve_progress_mode(progress_mode: str) -> str:
    if progress_mode not in {"none", "human", "json"}:
        raise typer.BadParameter("progress must be one of: none, human, json")
    return progress_mode


def progress_sink_for_mode(progress_mode: str):
    mode = resolve_progress_mode(progress_mode)
    if mode == "none":
        return None
    if mode == "human":
        return lambda event: typer.echo(_render_progress_event(event), err=True)
    return lambda event: typer.echo(json.dumps(event, sort_keys=False))


def emit_with_progress(
    payload: Any,
    *,
    progress_mode: str,
    format_name: str,
    quiet: bool = False,
) -> None:
    mode = resolve_progress_mode(progress_mode)
    if mode == "json":
        typer.echo(json.dumps({"event_type": "command_result", "result": payload}, sort_keys=False))
        return
    emit(payload, format_name=format_name, quiet=quiet)


def resolve_format(*, json_output: bool, format_name: str) -> str:
    return "json" if json_output else format_name


def fail(exc: Exception) -> None:
    typer.echo(str(exc))
    raise typer.Exit(code=exit_code_for_error(exc))


def _render_progress_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type", "event"))
    run_id = str(event.get("run_id", "run"))
    current_step = event.get("current_step")
    if event_type == "step_started":
        return f"[{run_id}] start {current_step}"
    if event_type == "step_finished":
        return f"[{run_id}] finish {current_step} ({event.get('status', 'ok')})"
    if event_type == "step_progress":
        message = str(event.get("message") or "progress")
        return f"[{run_id}] {current_step}: {message}"
    if event_type == "heartbeat":
        return f"[{run_id}] heartbeat {current_step}"
    if event_type == "warning":
        return f"[{run_id}] warning: {event.get('message', '')}"
    if event_type == "run_started":
        return f"[{run_id}] run started"
    if event_type == "run_succeeded":
        return f"[{run_id}] run succeeded"
    if event_type == "run_failed":
        return f"[{run_id}] run failed: {event.get('message', '')}"
    return f"[{run_id}] {event_type}"
