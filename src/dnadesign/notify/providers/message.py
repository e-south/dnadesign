"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/providers/message.py

Shared text rendering for notification payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any


def format_message(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "")).upper()
    tool = payload.get("tool", "-")
    run_id = payload.get("run_id", "-")
    parts = [f"{status} — {tool}/{run_id}"]
    message = payload.get("message")
    if message:
        parts.append(str(message))
    host = payload.get("host")
    cwd = payload.get("cwd")
    if host:
        parts.append(f"host: {host}")
    if cwd:
        parts.append(f"cwd: {cwd}")
    return "\n".join(parts)
