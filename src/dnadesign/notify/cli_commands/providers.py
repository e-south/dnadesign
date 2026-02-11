"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli_commands/providers.py

Provider payload mapping entrypoints for Notify CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from ..providers import format_payload


def format_for_provider(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    return format_payload(provider, payload)


__all__ = ["format_for_provider"]
