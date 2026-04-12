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


def resolve_format(*, json_output: bool, format_name: str) -> str:
    return "json" if json_output else format_name


def fail(exc: Exception) -> None:
    typer.echo(str(exc))
    raise typer.Exit(code=exit_code_for_error(exc))
