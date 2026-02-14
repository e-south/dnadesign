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


_STATUS_EMOJI = {
    "started": ":large_blue_circle:",
    "running": ":large_blue_circle:",
    "success": ":large_green_circle:",
    "failure": ":red_circle:",
}


def _status_line(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "").strip().lower()
    emoji = _STATUS_EMOJI.get(status, ":white_circle:")
    tool = str(payload.get("tool") or "-").strip()
    run_id = str(payload.get("run_id") or "-").strip()
    return f"{emoji} *{status.upper()}*  `{tool}/{run_id}`"


def _message_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = payload.get("message")
    if message is None:
        return []
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        return []
    blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": lines[0]}}]
    if len(lines) > 1:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(lines[1:12])},
            }
        )
    return blocks


def _context_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    elements: list[dict[str, Any]] = []
    host = payload.get("host")
    cwd = payload.get("cwd")
    if host:
        elements.append({"type": "mrkdwn", "text": f"*host*: `{host}`"})
    if cwd:
        elements.append({"type": "mrkdwn", "text": f"*cwd*: `{cwd}`"})
    if not elements:
        return None
    return {"type": "context", "elements": elements}


def format_slack(payload: dict[str, Any]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": _status_line(payload)},
        }
    ]
    blocks.extend(_message_blocks(payload))
    context_block = _context_block(payload)
    if context_block is not None:
        blocks.append(context_block)
    return {"text": format_message(payload), "blocks": blocks}
