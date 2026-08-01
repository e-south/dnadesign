"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/remote/test_transfer_runtime_locks.py

Transfer contracts for host-local USR runtime locks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.usr.src.sync.remote.transfer import collect_staged_entries, promote_staged_pull


def test_staged_entries_exclude_runtime_locks(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "records.parquet").write_bytes(b"records")
    (staged / "meta.md").write_text("metadata\n", encoding="utf-8")
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

    promote_staged_pull(staged, destination, primary_only=False, skip_snapshots=False)

    assert (destination / "records.parquet").read_bytes() == b"remote-records"
    assert (destination / "meta.md").read_text(encoding="utf-8") == "remote metadata\n"
    assert (destination / ".events.lock").read_text(encoding="utf-8") == "local-event-lock\n"
    assert (destination / ".usr.lock").read_text(encoding="utf-8") == "local-dataset-lock\n"
