"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_payload.py

Tests for notifier payload construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

import pytest

from dnadesign.notify.errors import NotifyValidationError
from dnadesign.notify.payload import build_payload


def test_build_payload_includes_required_fields() -> None:
    payload = build_payload(status="success", tool="densegen", run_id="demo")
    assert payload["status"] == "success"
    assert payload["tool"] == "densegen"
    assert payload["run_id"] == "demo"
    assert "timestamp" in payload
    assert "host" not in payload
    assert "cwd" not in payload
    assert "meta" in payload
    assert isinstance(payload["meta"], dict)
    assert re.match(r"\d{4}-\d{2}-\d{2}T", payload["timestamp"])


def test_build_payload_rejects_invalid_status() -> None:
    with pytest.raises(NotifyValidationError):
        build_payload(status="done", tool="densegen", run_id="demo")


def test_build_payload_includes_context_when_provided() -> None:
    payload = build_payload(
        status="running",
        tool="densegen",
        run_id="demo",
        host="host-a",
        cwd="/tmp/work",
    )
    assert payload["host"] == "host-a"
    assert payload["cwd"] == "/tmp/work"
