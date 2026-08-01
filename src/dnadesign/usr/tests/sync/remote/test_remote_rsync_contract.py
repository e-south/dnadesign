"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/remote/test_remote_rsync_contract.py

Tests for rsync command construction on USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

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

    remote.pull_to_local("demo", tmp_path / "pull")
    remote.push_from_local("demo", tmp_path / "source")

    assert len(commands) == 2
    for command in commands:
        assert any(command[index : index + 2] == ["--exclude", ".events.lock"] for index in range(len(command)))
        assert any(command[index : index + 2] == ["--exclude", ".usr.lock"] for index in range(len(command)))
        assert "--rsync-path" in command


def test_full_push_holds_remote_event_lock_for_rsync(
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

    remote.push_from_local("demo", tmp_path / "source")

    assert len(commands) == 1
    command = commands[0]
    rsync_path = command[command.index("--rsync-path") + 1]
    assert shlex.split(rsync_path) == [
        "flock",
        "-x",
        "-w",
        "300",
        "/project/alice/dnadesign/src/dnadesign/usr/datasets/demo/.events.lock",
        "rsync",
    ]


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

    remote.push_from_local("demo", tmp_path / "source", primary_only=True)

    assert len(commands) == 1
    assert "--rsync-path" not in commands[0]
