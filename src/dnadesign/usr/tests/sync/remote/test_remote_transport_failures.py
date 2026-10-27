"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/remote/test_remote_transport_failures.py

Regression tests for SSH transport/auth failures during remote dataset probes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from subprocess import CompletedProcess

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


def _ssh_auth_failure(_cmd, **_kwargs):
    return CompletedProcess(
        args=_cmd,
        returncode=255,
        stdout="",
        stderr="Permission denied (keyboard-interactive,hostbased)",
    )


def test_stat_dataset_raises_on_ssh_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote()
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.remote.subprocess.run", _ssh_auth_failure)

    with pytest.raises(RemoteUnavailableError, match="keyboard-interactive"):
        remote.stat_dataset("densegen_demo", verify="auto")


def test_stat_file_raises_on_ssh_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = _remote()
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.remote.subprocess.run", _ssh_auth_failure)

    with pytest.raises(RemoteUnavailableError, match="keyboard-interactive"):
        remote.stat_file("/project/tester/usr/densegen_demo/records.parquet", verify="auto")
