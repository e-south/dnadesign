"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/payload.py

Payload construction for notifications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .errors import NotifyValidationError

ALLOWED_STATUSES = {"success", "failure", "started", "running"}


def _require_text(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise NotifyValidationError(f"{field} must be a non-empty string")
    return text


def build_payload(
    *,
    status: str,
    tool: str,
    run_id: str,
    message: str | None = None,
    meta: dict[str, Any] | None = None,
    timestamp: str | None = None,
    host: str | None = None,
    cwd: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    status_norm = _require_text(status, field="status").lower()
    if status_norm not in ALLOWED_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_STATUSES))
        raise NotifyValidationError(f"status must be one of: {allowed}")
    tool_norm = _require_text(tool, field="tool")
    run_id_norm = _require_text(run_id, field="run_id")
    meta = meta or {}
    if not isinstance(meta, dict):
        raise NotifyValidationError("meta must be a JSON object")
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    payload = {
        "status": status_norm,
        "tool": tool_norm,
        "run_id": run_id_norm,
        "timestamp": ts,
        "meta": meta,
    }
    if host is not None:
        payload["host"] = str(host)
    if cwd is not None:
        payload["cwd"] = str(cwd)
    if message:
        payload["message"] = str(message)
    if version:
        payload["version"] = str(version)
    return payload
