"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/providers/slack.py

Slack payload formatting for webhook notifications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from .message import format_message


def format_slack(payload: dict[str, Any]) -> dict[str, Any]:
    return {"text": format_message(payload)}
