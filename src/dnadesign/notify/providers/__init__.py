"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/providers/__init__.py

Provider formatters for notification payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable

from ..errors import NotifyValidationError
from .discord import format_discord
from .generic import format_generic
from .slack import format_slack

_PROVIDERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "generic": format_generic,
    "slack": format_slack,
    "discord": format_discord,
}


def format_payload(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    name = str(provider or "").strip().lower()
    if not name:
        raise NotifyValidationError("provider must be a non-empty string")
    formatter = _PROVIDERS.get(name)
    if formatter is None:
        allowed = ", ".join(sorted(_PROVIDERS.keys()))
        raise NotifyValidationError(f"provider must be one of: {allowed}")
    return formatter(payload)
