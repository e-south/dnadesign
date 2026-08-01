"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/remote/test_remote_inventory_paths.py

Adversarial tests for remote sidecar inventory path normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.usr.src.contracts import RemoteUnavailableError
from dnadesign.usr.src.sync.remote.config import SSHRemoteConfig
from dnadesign.usr.src.sync.remote.remote import SSHRemote


def _remote() -> SSHRemote:
    return SSHRemote(
        SSHRemoteConfig(
            name="bu-scc",
            host="scc1.bu.edu",
            user="tester",
            base_dir="/project/tester/usr",
        )
    )


def test_remote_derived_inventory_rejects_parent_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote()

    def _ssh_run(_cmd: str, check: bool = True):
        del check
        return 0, "./densegen/part-000.parquet\n../escape.parquet\n", ""

    monkeypatch.setattr(remote, "_ssh_run", _ssh_run)

    with pytest.raises(RemoteUnavailableError, match="unsafe relative path"):
        remote._remote_list_derived_files("/project/tester/usr/densegen_demo/_derived")


def test_remote_aux_inventory_rejects_absolute_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote()

    def _ssh_run(_cmd: str, check: bool = True):
        del check
        return 0, "/tmp/absolute-path.json\n", ""

    monkeypatch.setattr(remote, "_ssh_run", _ssh_run)

    with pytest.raises(RemoteUnavailableError, match="unsafe relative path"):
        remote._remote_list_aux_files("/project/tester/usr/densegen_demo")


def test_remote_inventory_normalizes_and_sorts_valid_relative_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote()

    def _ssh_run(_cmd: str, check: bool = True):
        del check
        return 0, "./densegen/part-010.parquet\n./densegen/part-001.parquet\n", ""

    monkeypatch.setattr(remote, "_ssh_run", _ssh_run)

    assert remote._remote_list_derived_files("/project/tester/usr/densegen_demo/_derived") == [
        "densegen/part-001.parquet",
        "densegen/part-010.parquet",
    ]


def test_remote_aux_inventory_excludes_runtime_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote()
    commands: list[str] = []

    def capture_probe(command: str):
        commands.append(command)
        return 0, "", ""

    monkeypatch.setattr(remote, "_ssh_probe", capture_probe)

    assert remote._remote_list_aux_files("/project/tester/usr/densegen_demo") == []
    assert len(commands) == 1
    assert "! -path './.events.lock'" in commands[0]
    assert "! -path './.usr.lock'" in commands[0]
