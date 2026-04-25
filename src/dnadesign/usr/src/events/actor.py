"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/actor.py

Actor normalization helpers for USR event records.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import socket
from typing import Any, Mapping


def _default_actor() -> dict[str, Any]:
    tool = str(os.getenv("USR_ACTOR_TOOL") or "usr").strip() or "usr"
    run_id = str(os.getenv("USR_ACTOR_RUN_ID") or "").strip()
    if not run_id:
        run_id = f"usr-pid-{os.getpid()}"
    return {
        "tool": tool,
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def _normalize_actor(actor: Mapping[str, Any] | None) -> dict[str, Any]:
    if actor is None:
        return _default_actor()
    if not isinstance(actor, Mapping):
        raise TypeError("actor must be a mapping when provided")
    actor_payload = dict(actor)
    tool = str(actor_payload.get("tool") or "").strip()
    if not tool:
        raise ValueError("actor.tool must be a non-empty string")
    run_id = str(actor_payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("actor.run_id must be a non-empty string")
    actor_payload["tool"] = tool
    actor_payload["run_id"] = run_id
    return actor_payload
