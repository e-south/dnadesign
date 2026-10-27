"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/remote/test_remote_rsync_contract.py

Tests for rsync command construction on USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.usr.src.contracts import TransferError
from dnadesign.usr.src.sync.remote import locks as locks_module
from dnadesign.usr.src.sync.remote import remote as remote_module
from dnadesign.usr.src.sync.remote.config import SSHRemoteConfig
from dnadesign.usr.src.sync.remote.remote import SSHRemote


def _remote(*, batch_mode: bool) -> SSHRemote:
    return SSHRemote(
        SSHRemoteConfig(
            name="bu-scc",
            host="scc1.bu.edu",
            user="alice",
            base_dir="/project/alice/dnadesign/src/dnadesign/usr/datasets",
            batch_mode=batch_mode,
        )
    )


class _LiveProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


@contextmanager
def _active_event_lease(remote: SSHRemote, monkeypatch: pytest.MonkeyPatch):
    process = _LiveProcess()

    @contextmanager
    def _session(*_args, **_kwargs):
        yield locks_module._RemoteLockSession(process=process)

    monkeypatch.setattr(remote_module, "remote_lock_session", _session)
    with remote.event_log_transfer_lock("demo") as lease:
        yield lease, process


@contextmanager
def _active_dataset_lease(remote: SSHRemote, monkeypatch: pytest.MonkeyPatch):
    process = _LiveProcess()

    @contextmanager
    def _session(*_args, **_kwargs):
        yield locks_module._RemoteLockSession(process=process)

    monkeypatch.setattr(remote_module, "remote_lock_session", _session)
    with remote.dataset_transfer_lock("demo") as lease:
        yield lease, process


def test_rsync_cmd_avoids_host_specific_permission_metadata() -> None:
    cmd = _remote(batch_mode=True)._rsync_cmd()

    assert "-rltz" in cmd
    assert "-az" not in cmd
    assert "--no-perms" in cmd
    assert "--no-owner" in cmd
    assert "--no-group" in cmd
    assert "--omit-dir-times" in cmd


def test_rsync_cmd_respects_batch_mode_toggle() -> None:
    strict_cmd = _remote(batch_mode=True)._rsync_cmd()
    interactive_cmd = _remote(batch_mode=False)._rsync_cmd()

    strict_ssh = strict_cmd[strict_cmd.index("-e") + 1]
    interactive_ssh = interactive_cmd[interactive_cmd.index("-e") + 1]

    assert "BatchMode=yes" in strict_ssh
    assert "BatchMode=yes" not in interactive_ssh


def test_dataset_rsync_excludes_host_local_runtime_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    commands: list[list[str]] = []

    def capture_run(command: list[str]):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_module.subprocess, "run", capture_run)
    monkeypatch.setattr(remote, "_ssh_run", lambda *_args, **_kwargs: (0, "", ""))

    with _active_dataset_lease(remote, monkeypatch):
        remote.pull_to_local("demo", tmp_path / "pull")
    with _active_dataset_lease(remote, monkeypatch):
        with _active_event_lease(remote, monkeypatch) as (event_lease, _process):
            remote.push_from_local("demo", tmp_path / "source", event_lease=event_lease)

    assert len(commands) == 2
    for command in commands:
        assert any(command[index : index + 2] == ["--exclude", ".events.lock"] for index in range(len(command)))
        assert any(command[index : index + 2] == ["--exclude", ".usr.lock"] for index in range(len(command)))
        assert any(command[index : index + 2] == ["--exclude", ".usr.transfer.lock"] for index in range(len(command)))
        assert any(command[index : index + 2] == ["--exclude", ".usr.lease.*"] for index in range(len(command)))
    assert "--rsync-path" in commands[0]
    assert "--rsync-path" in commands[1]
    push_program = commands[1][commands[1].index("--rsync-path") + 1]
    assert "flock -s" in push_program
    assert event_lease.token in push_program
    assert 'kill -0 "$lease_pid"' in push_program
    assert "ps -o lstart=" in push_program


def test_full_push_requires_a_remote_event_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    commands: list[list[str]] = []

    def capture_run(command: list[str]):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_module.subprocess, "run", capture_run)
    monkeypatch.setattr(remote, "_ssh_run", lambda *_args, **_kwargs: (0, "", ""))

    with _active_dataset_lease(remote, monkeypatch):
        with pytest.raises(TransferError, match="remote event-log lease"):
            remote.push_from_local("demo", tmp_path / "source")

    assert commands == []


def test_remote_event_revision_requires_the_active_matching_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote(batch_mode=True)
    payload = b'{"action":"ready"}\n'
    monkeypatch.setattr(remote, "_remote_stat_file", lambda _path: (True, len(payload), "0"))
    monkeypatch.setattr(remote, "_remote_sha256", lambda _path: hashlib.sha256(payload).hexdigest())
    with _active_event_lease(remote, monkeypatch) as (lease, _process):
        revision = remote.event_log_revision("demo", event_lease=lease)

        assert revision.exists is True
        assert revision.size_bytes == len(payload)
        assert revision.sha256 == hashlib.sha256(payload).hexdigest()

    with pytest.raises(TransferError, match="active remote event-log lease"):
        remote.event_log_revision("demo", event_lease=lease)


def test_full_push_rejects_a_hand_constructed_event_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    monkeypatch.setattr(remote, "_ssh_run", lambda *_args, **_kwargs: (0, "", ""))

    with _active_dataset_lease(remote, monkeypatch):
        with _active_event_lease(remote, monkeypatch) as (lease, _process):
            forged = locks_module._RemoteEventLogLease(
                owner=remote,
                dataset="demo",
                token=lease.token,
                session=lease.session,
            )
            with pytest.raises(TransferError, match="active remote event-log lease"):
                remote.push_from_local("demo", tmp_path / "source", event_lease=forged)


def test_full_push_rejects_a_lease_whose_lock_session_died(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    commands: list[list[str]] = []
    monkeypatch.setattr(remote_module.subprocess, "run", lambda command: commands.append(command))
    monkeypatch.setattr(remote, "_ssh_run", lambda *_args, **_kwargs: (0, "", ""))

    with _active_dataset_lease(remote, monkeypatch):
        with _active_event_lease(remote, monkeypatch) as (lease, process):
            process.returncode = 255
            with pytest.raises(TransferError, match="was lost"):
                remote.push_from_local("demo", tmp_path / "source", event_lease=lease)

    assert commands == []


def test_full_push_dry_run_does_not_create_remote_state_or_require_a_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    commands: list[list[str]] = []

    def capture_run(command: list[str]):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_module.subprocess, "run", capture_run)
    monkeypatch.setattr(remote, "_ssh_run", lambda *_args, **_kwargs: pytest.fail("dry-run mutated remote state"))

    remote.push_from_local("demo", tmp_path / "source", dry_run=True)

    assert len(commands) == 1
    assert "--dry-run" in commands[0]
    assert "--rsync-path" not in commands[0]


def test_file_dry_runs_do_not_create_local_or_remote_parent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    commands: list[list[str]] = []
    local_destination = tmp_path / "missing" / "records.parquet"
    local_source = tmp_path / "source.parquet"

    def capture_run(command: list[str]):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_module.subprocess, "run", capture_run)
    monkeypatch.setattr(
        remote,
        "_ssh_run",
        lambda *_args, **_kwargs: pytest.fail("file dry-run mutated remote state"),
    )

    remote.pull_file("/remote/missing/records.parquet", local_destination, dry_run=True)
    remote.push_file(local_source, "/remote/missing/records.parquet", dry_run=True)

    assert not local_destination.parent.exists()
    assert len(commands) == 2
    assert all("--dry-run" in command for command in commands)


def test_pull_without_a_dataset_lease_fails_before_local_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    destination = tmp_path / "not-created"
    monkeypatch.setattr(
        remote_module.subprocess,
        "run",
        lambda _command: pytest.fail("pull without a dataset lease reached rsync"),
    )

    with pytest.raises(TransferError, match="active remote dataset lease"):
        remote.pull_to_local("demo", destination)

    assert not destination.exists()


def test_primary_only_push_does_not_acquire_remote_event_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    commands: list[list[str]] = []

    def capture_run(command: list[str]):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_module.subprocess, "run", capture_run)
    monkeypatch.setattr(remote, "_ssh_run", lambda *_args, **_kwargs: (0, "", ""))

    with _active_dataset_lease(remote, monkeypatch):
        remote.push_from_local("demo", tmp_path / "source", primary_only=True)

    assert len(commands) == 1
    assert "--rsync-path" in commands[0]
    program = commands[0][commands[0].index("--rsync-path") + 1]
    assert ".usr.transfer.lock" in program
    assert ".events.lock" not in program


def test_primary_only_push_rejects_a_dead_dataset_lease_before_rsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(batch_mode=True)
    commands: list[list[str]] = []
    monkeypatch.setattr(remote_module.subprocess, "run", lambda command: commands.append(command))
    monkeypatch.setattr(remote, "_ssh_run", lambda *_args, **_kwargs: (0, "", ""))

    with pytest.raises(TransferError, match="was lost"):
        with _active_dataset_lease(remote, monkeypatch) as (_lease, process):
            process.returncode = 255
            remote.push_from_local("demo", tmp_path / "source", primary_only=True)

    assert commands == []
