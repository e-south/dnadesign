"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/remote/test_transfer_runtime_locks.py

Transfer contracts for host-local USR runtime locks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from dnadesign.usr.src.contracts import VerificationError
from dnadesign.usr.src.events import append as event_append_module
from dnadesign.usr.src.events import append_event_line
from dnadesign.usr.src.sync.remote import transfer as transfer_module
from dnadesign.usr.src.sync.remote.transfer import (
    capture_event_log_revision,
    collect_staged_entries,
    promote_staged_pull,
)


def test_staged_entries_exclude_runtime_locks(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"records")
    (staged / "meta.md").write_text("metadata\n", encoding="utf-8")
    (staged / ".events.log").write_text('{"event":"remote"}\n', encoding="utf-8")
    (staged / ".events.lock").write_text("remote-event-lock\n", encoding="utf-8")
    (staged / ".usr.lock").write_text("remote-dataset-lock\n", encoding="utf-8")

    entries = collect_staged_entries(staged, skip_snapshots=False)

    assert [relative.as_posix() for _source, relative in entries] == ["meta.md"]


def test_pull_promotion_preserves_local_runtime_locks(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"remote-records")
    (staged / "meta.md").write_text("remote metadata\n", encoding="utf-8")
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / ".events.lock").write_text("local-event-lock\n", encoding="utf-8")
    (destination / ".usr.lock").write_text("local-dataset-lock\n", encoding="utf-8")
    (destination / ".events.log").write_text('{"event":"local"}\n', encoding="utf-8")

    expected_event_revision = capture_event_log_revision(destination / ".events.log")
    promote_staged_pull(
        staged,
        destination,
        primary_only=False,
        skip_snapshots=False,
        expected_event_revision=expected_event_revision,
    )

    assert (destination / "records.parquet").read_bytes() == b"remote-records"
    assert (destination / "meta.md").read_text(encoding="utf-8") == "remote metadata\n"
    assert (destination / ".events.lock").read_text(encoding="utf-8") == "local-event-lock\n"
    assert (destination / ".usr.lock").read_text(encoding="utf-8") == "local-dataset-lock\n"
    assert not (destination / ".events.log").exists()


def test_primary_only_pull_leaves_event_log_outside_promotion(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"remote-records")
    (staged / ".events.log").write_text('{"event":"remote"}\n', encoding="utf-8")
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "records.parquet").write_bytes(b"local-records")
    (destination / ".events.log").write_text('{"event":"local"}\n', encoding="utf-8")

    promote_staged_pull(
        staged,
        destination,
        primary_only=True,
        skip_snapshots=False,
        expected_event_revision=None,
    )

    assert (destination / "records.parquet").read_bytes() == b"remote-records"
    assert (destination / ".events.log").read_text(encoding="utf-8") == '{"event":"local"}\n'
    assert not (destination / ".events.lock").exists()


def test_full_pull_rejects_replaced_event_log_before_destination_mutation(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"remote-records")
    (staged / ".events.log").write_text('{"event":"remote"}\n', encoding="utf-8")
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "records.parquet").write_bytes(b"local-records")
    destination_event = destination / ".events.log"
    destination_event.write_text('{"event":"local"}\n', encoding="utf-8")
    expected_event_revision = capture_event_log_revision(destination_event)

    replacement = tmp_path / "replacement.events.log"
    replacement.write_bytes(destination_event.read_bytes())
    replacement.replace(destination_event)

    with pytest.raises(VerificationError, match="event log changed while the full pull was staged"):
        promote_staged_pull(
            staged,
            destination,
            primary_only=False,
            skip_snapshots=False,
            expected_event_revision=expected_event_revision,
        )

    assert (destination / "records.parquet").read_bytes() == b"local-records"
    assert destination_event.read_text(encoding="utf-8") == '{"event":"local"}\n'


def test_full_pull_requires_pre_transfer_event_revision(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"remote-records")
    destination = tmp_path / "dataset"

    with pytest.raises(ValueError, match="requires its pre-transfer event-log revision"):
        promote_staged_pull(
            staged,
            destination,
            primary_only=False,
            skip_snapshots=False,
            expected_event_revision=None,
        )

    assert not destination.exists()


def test_pull_promotion_serializes_event_replacement_with_concurrent_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"remote-records")
    (staged / ".events.log").write_text('{"event":"remote"}\n', encoding="utf-8")
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "records.parquet").write_bytes(b"local-records")
    (destination / ".events.log").write_text('{"event":"local"}\n', encoding="utf-8")

    event_copy_started = threading.Event()
    release_event_copy = threading.Event()
    append_lock_attempted = threading.Event()
    append_completed = threading.Event()
    original_copy_file_atomic = transfer_module.copy_file_atomic
    original_event_log_lock = event_append_module.event_log_lock
    expected_event_revision = capture_event_log_revision(destination / ".events.log")

    def blocking_copy_file_atomic(source: Path, target: Path) -> None:
        if Path(target).name == ".events.log":
            event_copy_started.set()
            assert release_event_copy.wait(timeout=5)
        original_copy_file_atomic(source, target)

    monkeypatch.setattr(transfer_module, "copy_file_atomic", blocking_copy_file_atomic)

    @contextmanager
    def tracked_append_lock(event_path: str | Path) -> Iterator[None]:
        append_lock_attempted.set()
        with original_event_log_lock(event_path):
            yield

    monkeypatch.setattr(event_append_module, "event_log_lock", tracked_append_lock)

    def append_concurrently() -> None:
        append_event_line(destination / ".events.log", '{"event":"local-concurrent"}')
        append_completed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        promotion = executor.submit(
            promote_staged_pull,
            staged,
            destination,
            primary_only=False,
            skip_snapshots=False,
            expected_event_revision=expected_event_revision,
        )
        assert event_copy_started.wait(timeout=5)
        append = executor.submit(append_concurrently)
        try:
            assert append_lock_attempted.wait(timeout=5)
            assert not append_completed.wait(timeout=0.1)
        finally:
            release_event_copy.set()
        promotion.result(timeout=5)
        append.result(timeout=5)

    assert append_completed.is_set()
    assert (destination / ".events.log").read_text(encoding="utf-8").splitlines() == [
        '{"event":"remote"}',
        '{"event":"local-concurrent"}',
    ]


@pytest.mark.parametrize("blocked_target", ["records.parquet", "meta.md"])
@pytest.mark.parametrize("staged_event_present", [True, False])
def test_full_pull_holds_event_lock_before_destination_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_target: str,
    staged_event_present: bool,
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"remote-records")
    (staged / "meta.md").write_text("remote metadata\n", encoding="utf-8")
    if staged_event_present:
        (staged / ".events.log").write_text('{"event":"remote"}\n', encoding="utf-8")
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "records.parquet").write_bytes(b"local-records")
    (destination / "meta.md").write_text("local metadata\n", encoding="utf-8")
    (destination / ".events.log").write_text('{"event":"local"}\n', encoding="utf-8")

    destination_copy_started = threading.Event()
    release_destination_copy = threading.Event()
    append_lock_attempted = threading.Event()
    append_completed = threading.Event()
    original_copy_file_atomic = transfer_module.copy_file_atomic
    original_event_log_lock = event_append_module.event_log_lock
    expected_event_revision = capture_event_log_revision(destination / ".events.log")

    def blocking_copy_file_atomic(source: Path, target: Path) -> None:
        target = Path(target)
        if target.parent == destination and target.name == blocked_target:
            destination_copy_started.set()
            assert release_destination_copy.wait(timeout=5)
        original_copy_file_atomic(source, target)

    monkeypatch.setattr(transfer_module, "copy_file_atomic", blocking_copy_file_atomic)

    @contextmanager
    def tracked_append_lock(event_path: str | Path) -> Iterator[None]:
        append_lock_attempted.set()
        with original_event_log_lock(event_path):
            yield

    monkeypatch.setattr(event_append_module, "event_log_lock", tracked_append_lock)

    def append_concurrently() -> None:
        append_event_line(destination / ".events.log", '{"event":"local-concurrent"}')
        append_completed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        promotion = executor.submit(
            promote_staged_pull,
            staged,
            destination,
            primary_only=False,
            skip_snapshots=False,
            expected_event_revision=expected_event_revision,
        )
        assert destination_copy_started.wait(timeout=5)
        append = executor.submit(append_concurrently)
        try:
            assert append_lock_attempted.wait(timeout=5)
            assert not append_completed.wait(timeout=0.1)
        finally:
            release_destination_copy.set()
        promotion.result(timeout=5)
        append.result(timeout=5)

    assert append_completed.is_set()
    expected_events = ['{"event":"local-concurrent"}']
    if staged_event_present:
        expected_events.insert(0, '{"event":"remote"}')
    assert (destination / ".events.log").read_text(encoding="utf-8").splitlines() == expected_events


@pytest.mark.parametrize("staged_event_kind", ["directory", "symlink"])
def test_pull_promotion_rejects_malformed_staged_event_log_before_mutation(
    tmp_path: Path, staged_event_kind: str
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"remote-records")
    staged_events = staged / ".events.log"
    if staged_event_kind == "directory":
        staged_events.mkdir()
    else:
        staged_events.symlink_to(staged / "missing-events.log")

    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "records.parquet").write_bytes(b"local-records")
    (destination / ".events.log").write_text('{"event":"local"}\n', encoding="utf-8")

    expected_event_revision = capture_event_log_revision(destination / ".events.log")
    with pytest.raises(VerificationError, match=r"\.events\.log"):
        promote_staged_pull(
            staged,
            destination,
            primary_only=False,
            skip_snapshots=False,
            expected_event_revision=expected_event_revision,
        )

    assert (destination / "records.parquet").read_bytes() == b"local-records"
    assert (destination / ".events.log").read_text(encoding="utf-8") == '{"event":"local"}\n'
    assert (destination / ".events.lock").is_file()
