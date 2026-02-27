"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_remote_failures.py

Pressure tests for USR sync behavior when remote stat probing fails.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from dnadesign.usr.src import sync as sync_module
from dnadesign.usr.src.errors import RemoteUnavailableError, VerificationError
from dnadesign.usr.src.remote import RemoteDatasetStat, RemotePrimaryStat


def _write_stub_records(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")


def _remote_with_stat_failure(message: str = "remote stat unavailable"):
    class _FailingRemote:
        def __init__(self, _cfg):
            pass

        def stat_dataset(self, _dataset: str, *, verify: str = "auto", include_derived_hashes: bool = False):
            raise RemoteUnavailableError(f"{message}; verify={verify}")

        def pull_to_local(self, *_args, **_kwargs):
            raise AssertionError("pull_to_local must not be called when remote stat probing fails")

        def push_from_local(self, *_args, **_kwargs):
            raise AssertionError("push_from_local must not be called when remote stat probing fails")

    return _FailingRemote


def test_plan_diff_fails_fast_when_remote_stat_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _remote_with_stat_failure("ssh stat failure"))

    with pytest.raises(RemoteUnavailableError, match="ssh stat failure"):
        sync_module.plan_diff(tmp_path, "densegen/demo", "bu-scc", verify="auto")


def test_execute_pull_does_not_lock_or_transfer_when_remote_stat_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _remote_with_stat_failure("pull stat failure"))

    lock_called = {"value": False}

    def _lock(_path):
        @contextmanager
        def _ctx():
            lock_called["value"] = True
            yield

        return _ctx()

    monkeypatch.setattr(sync_module, "dataset_write_lock", _lock)

    with pytest.raises(RemoteUnavailableError, match="pull stat failure"):
        sync_module.execute_pull(tmp_path, "densegen/demo", "bu-scc", sync_module.SyncOptions(verify="auto"))

    assert lock_called["value"] is False


def test_execute_push_does_not_lock_or_transfer_when_remote_stat_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _remote_with_stat_failure("push stat failure"))

    lock_called = {"value": False}

    def _lock(_path):
        @contextmanager
        def _ctx():
            lock_called["value"] = True
            yield

        return _ctx()

    monkeypatch.setattr(sync_module, "dataset_write_lock", _lock)

    with pytest.raises(RemoteUnavailableError, match="push stat failure"):
        sync_module.execute_push(tmp_path, "densegen/demo", "bu-scc", sync_module.SyncOptions(verify="auto"))

    assert lock_called["value"] is False


def test_execute_pull_refuses_transfer_when_remote_primary_missing(monkeypatch, tmp_path: Path) -> None:
    local_records = tmp_path / "densegen" / "demo" / "records.parquet"
    _write_stub_records(local_records)

    transfer_called = {"value": False}
    lock_called = {"value": False}

    class _Remote:
        def __init__(self, _cfg):
            pass

        def stat_dataset(self, _dataset: str, *, verify: str = "auto", include_derived_hashes: bool = False):
            return RemoteDatasetStat(
                primary=RemotePrimaryStat(False, None, None, None, None, None),
                meta_mtime=None,
                events_lines=0,
                snapshot_names=[],
            )

        def pull_to_local(self, *_args, **_kwargs):
            transfer_called["value"] = True

    def _lock(_path):
        @contextmanager
        def _ctx():
            lock_called["value"] = True
            yield

        return _ctx()

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)
    monkeypatch.setattr(sync_module, "dataset_write_lock", _lock)

    with pytest.raises(VerificationError, match="remote records.parquet is missing"):
        sync_module.execute_pull(tmp_path, "densegen/demo", "bu-scc", sync_module.SyncOptions(verify="auto"))

    assert transfer_called["value"] is False
    assert lock_called["value"] is False


def test_execute_push_refuses_transfer_when_local_primary_missing(monkeypatch, tmp_path: Path) -> None:
    transfer_called = {"value": False}
    lock_called = {"value": False}

    class _Remote:
        def __init__(self, _cfg):
            pass

        def stat_dataset(self, _dataset: str, *, verify: str = "auto", include_derived_hashes: bool = False):
            return RemoteDatasetStat(
                primary=RemotePrimaryStat(False, None, None, None, None, None),
                meta_mtime=None,
                events_lines=0,
                snapshot_names=[],
            )

        def push_from_local(self, *_args, **_kwargs):
            transfer_called["value"] = True

    def _lock(_path):
        @contextmanager
        def _ctx():
            lock_called["value"] = True
            yield

        return _ctx()

    monkeypatch.setattr(sync_module, "get_remote", lambda _name: object())
    monkeypatch.setattr(sync_module, "SSHRemote", _Remote)
    monkeypatch.setattr(sync_module, "dataset_write_lock", _lock)

    with pytest.raises(VerificationError, match="local records.parquet is missing"):
        sync_module.execute_push(tmp_path, "densegen/demo", "bu-scc", sync_module.SyncOptions(verify="auto"))

    assert transfer_called["value"] is False
    assert lock_called["value"] is False
