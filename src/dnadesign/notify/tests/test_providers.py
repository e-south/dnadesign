"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_providers.py

Tests for notifier provider formatting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.notify.payload import build_payload
from dnadesign.notify.providers import format_payload


def test_format_payload_generic_round_trips() -> None:
    payload = build_payload(status="success", tool="densegen", run_id="demo")
    formatted = format_payload("generic", payload)
    assert formatted == payload


def test_format_payload_slack_minimum_fields() -> None:
    payload = build_payload(status="failure", tool="densegen", run_id="demo", message="oops")
    formatted = format_payload("slack", payload)
    assert "text" in formatted
    assert "blocks" in formatted
    assert isinstance(formatted["blocks"], list)
    assert formatted["blocks"][0]["type"] == "section"
    assert "*FAILURE*" in formatted["blocks"][0]["text"]["text"]


def test_format_payload_discord_minimum_fields() -> None:
    payload = build_payload(status="running", tool="densegen", run_id="demo")
    formatted = format_payload("discord", payload)
    assert "content" in formatted
