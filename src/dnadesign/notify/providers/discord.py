"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/providers/discord.py

Discord payload formatting for webhook notifications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from .message import format_message


def format_discord(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": format_message(payload)}
