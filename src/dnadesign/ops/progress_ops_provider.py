"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/progress_ops_provider.py

Provider-owned OPS status builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from .progress_support import required_path


def provide_ops_audit_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    del repo_root
    return ops_audit_progress(inputs.get("audit_json"))


def ops_audit_progress(audit_json: object) -> tuple[str, str, dict[str, object]]:
    resolved_audit = required_path(audit_json, flag_name="--audit-json", progress_kind="ops-audit-json")
    if not resolved_audit.exists():
        return (
            "missing",
            "audit artifact not found",
            {"audit_json": str(resolved_audit)},
        )
    payload = json.loads(resolved_audit.read_text(encoding="utf-8"))
    execution = dict(payload.get("execution") or {})
    plan = dict(payload.get("plan") or {})
    commands = list(execution.get("commands") or [])
    phase_counts = Counter(str(command.get("phase") or "unknown") for command in commands)
    ok = bool(execution.get("ok", False))
    failed_phase = execution.get("failed_phase")
    queue_probe = _extract_queue_probe_evidence(commands)
    if ok and queue_probe is not None and queue_probe["status"] == "degraded":
        summary = "latest orchestration audit passed with degraded queue probe"
    elif ok:
        summary = "latest orchestration audit passed"
    else:
        summary = f"latest orchestration audit failed at {failed_phase or 'unknown'}"
    return (
        "attention" if (not ok or (queue_probe is not None and queue_probe["status"] == "degraded")) else "ok",
        summary,
        {
            "audit_json": str(resolved_audit),
            "workflow_id": plan.get("workflow_id"),
            "project": plan.get("project"),
            "runbook_id": plan.get("runbook_id"),
            "workspace_root": plan.get("workspace_root"),
            "execution_ok": ok,
            "failed_phase": failed_phase,
            "command_count": len(commands),
            "phase_counts": dict(sorted(phase_counts.items())),
            "queue_probe": queue_probe,
        },
    )


def _extract_queue_probe_evidence(commands: list[object]) -> dict[str, object] | None:
    queue_probe_commands: list[dict[str, object]] = []
    status = "ok"
    for command in commands:
        if not isinstance(command, dict):
            continue
        fields = _parse_record_fields(command.get("stdout"))
        queue_probe = fields.get("queue_probe")
        if queue_probe is None:
            continue
        if queue_probe != "ok":
            status = "degraded"
        queue_probe_commands.append(
            {
                "phase": command.get("phase"),
                "command": command.get("command"),
                "queue_probe": queue_probe,
                "next_action": fields.get("next_action"),
                "submit_gate": fields.get("submit_gate"),
                "advisor": fields.get("advisor"),
                "stderr": str(command.get("stderr") or "").strip() or None,
            }
        )
    if not queue_probe_commands:
        return None
    return {
        "status": status,
        "commands": queue_probe_commands,
    }


def _parse_record_fields(raw_text: object) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in str(raw_text or "").splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", maxsplit=1)
            if key:
                fields[key] = value
    return fields


__all__ = [
    "ops_audit_progress",
    "provide_ops_audit_status",
]
