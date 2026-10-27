"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/core/test_event_log_gardening.py

Tests for USR event-log gardening.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces
from dnadesign.usr.src.events import garden_event_log
from dnadesign.usr.src.events import gardening as event_gardening_module


def _dataset_with_events(tmp_path: Path) -> Dataset:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "events_demo")
    dataset.init(source="test", notes="event gardening test")
    for index in range(5):
        dataset.log_event(
            "materialize",
            args={"index": index},
            metrics={"rows": index},
            actor={"tool": "test", "run_id": "event-garden-fixture"},
        )
    return dataset


def test_event_log_garden_dry_run_reports_tail_without_mutating(tmp_path: Path) -> None:
    dataset = _dataset_with_events(tmp_path)
    before = dataset.events_path.read_text(encoding="utf-8")
    before_lines = len(before.splitlines())

    result = garden_event_log(dataset, retain_last=2, write=False)

    assert result.mode == "dry_run"
    assert result.total_lines == before_lines
    assert result.retained_lines == 2
    assert result.archived_lines == before_lines - 2
    assert result.archive_path is None
    assert dataset.events_path.read_text(encoding="utf-8") == before


def test_event_log_garden_write_archives_full_log_and_keeps_tail(tmp_path: Path) -> None:
    dataset = _dataset_with_events(tmp_path)
    before_lines = len(dataset.events_path.read_text(encoding="utf-8").splitlines())

    result = garden_event_log(
        dataset,
        retain_last=2,
        write=True,
        acknowledge_notify_cursor_reset=True,
        reason="unit test compaction",
    )

    assert result.mode == "write"
    assert result.archive_path is not None
    archive_path = Path(result.archive_path)
    assert archive_path.exists()
    assert len(archive_path.read_text(encoding="utf-8").splitlines()) == before_lines
    live_events = [json.loads(line) for line in dataset.events_path.read_text(encoding="utf-8").splitlines()]
    assert len(live_events) == 3
    assert live_events[-1]["action"] == "event_log_garden"
    assert live_events[-1]["metrics"]["archived_lines"] == before_lines - 2
    assert live_events[-1]["maintenance"]["notify_cursor_reset_required"] is True


def test_event_log_garden_holds_replacement_lock_through_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_events(tmp_path)
    lifecycle: list[str] = []
    replace_event_log = event_gardening_module._replace_event_log

    @contextmanager
    def tracked_lock(_event_path: Path):
        lifecycle.append("lock-enter")
        try:
            yield
        finally:
            lifecycle.append("lock-exit")

    def tracked_replace(path: Path, lines: list[str]) -> None:
        assert lifecycle == ["lock-enter"]
        replace_event_log(path, lines)
        lifecycle.append("replace")

    monkeypatch.setattr(event_gardening_module, "event_log_lock", tracked_lock)
    monkeypatch.setattr(event_gardening_module, "_replace_event_log", tracked_replace)

    garden_event_log(
        dataset,
        retain_last=2,
        write=True,
        acknowledge_notify_cursor_reset=True,
        reason="lock contract",
    )

    assert lifecycle == ["lock-enter", "replace", "lock-exit"]


def test_event_log_garden_write_requires_notify_cursor_ack(tmp_path: Path) -> None:
    dataset = _dataset_with_events(tmp_path)

    try:
        garden_event_log(dataset, retain_last=2, write=True)
    except ValueError as error:
        assert "Notify watchers" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("event-log gardening should require Notify cursor acknowledgement")
