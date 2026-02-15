"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_event_transforms.py

Behavior tests for notify event transform and validation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.event_transforms import event_message, event_meta, status_for_action, validate_usr_event


def test_status_for_action_maps_fail_and_init() -> None:
    assert status_for_action("densegen_flush_failed", event={}) == "failure"
    assert status_for_action("init", event={}) == "started"
    assert status_for_action("materialize", event={}) == "running"


def test_event_message_uses_rows_written_when_present() -> None:
    event = {
        "action": "materialize",
        "dataset": {"name": "demo"},
        "metrics": {"rows_written": 12},
    }
    assert event_message(event, run_id="run-1", duration_seconds=None) == "materialize on demo (rows_written=12)"


def test_event_meta_respects_include_flags() -> None:
    event = {
        "event_version": 1,
        "action": "materialize",
        "dataset": {"name": "demo", "root": "/tmp/usr"},
        "args": {"x": 1},
        "fingerprint": {"rows": 10},
        "registry_hash": "abc",
        "timestamp_utc": "2026-02-10T00:00:00+00:00",
    }
    meta = event_meta(
        event,
        include_args=True,
        include_raw_event=False,
        include_context=True,
    )
    assert meta["usr_dataset_name"] == "demo"
    assert meta["usr_dataset_root"] == "/tmp/usr"
    assert meta["usr_args"] == {"x": 1}
    assert "usr_event" not in meta


def test_validate_usr_event_rejects_unknown_version_without_flag() -> None:
    event = {
        "event_version": 2,
        "action": "attach",
        "dataset": {"name": "demo"},
        "actor": {"tool": "densegen", "run_id": "r1"},
    }
    with pytest.raises(NotifyConfigError, match="unknown event_version=2; expected 1"):
        validate_usr_event(event, expected_version=1, allow_unknown_version=False)
