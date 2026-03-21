"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_remote_control_session.py

Tests for SSH control-socket status and warm-auth bootstrap on USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dnadesign.usr.src.config import SSHRemoteConfig
from dnadesign.usr.src.errors import RemoteUnavailableError
from dnadesign.usr.src.remote import SSHRemote


class _FakeTTY:
    def __init__(self, *, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def _remote() -> SSHRemote:
    return SSHRemote(
        SSHRemoteConfig(
            name="cluster",
            host="scc1.bu.edu",
            user="alice",
            base_dir="/project/alice/usr_datasets",
            batch_mode=False,
        )
    )


def test_control_session_status_reports_live_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    socket_path = tmp_path / "cm.sock"
    socket_path.write_text("", encoding="utf-8")

    def _run(cmd, **kwargs):
        if "-G" in cmd:
            return CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=(
                    "user alice\n"
                    "hostname scc1.bu.edu\n"
                    "controlmaster auto\n"
                    f"controlpath {socket_path}\n"
                    "controlpersist 600\n"
                ),
                stderr="",
            )
        if "-O" in cmd:
            return CompletedProcess(args=cmd, returncode=0, stdout="Master running\n", stderr="")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    monkeypatch.setattr("dnadesign.usr.src.remote.subprocess.run", _run)

    status = _remote().control_session_status()
    assert status.multiplex_enabled is True
    assert status.socket_exists is True
    assert status.socket_live is True
    assert status.control_path == str(socket_path)


def test_warm_auth_requires_tty_when_socket_not_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    socket_path = tmp_path / "cm.sock"

    def _run(cmd, **kwargs):
        if "-G" in cmd:
            return CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=(
                    "user alice\n"
                    "hostname scc1.bu.edu\n"
                    "controlmaster auto\n"
                    f"controlpath {socket_path}\n"
                    "controlpersist 600\n"
                ),
                stderr="",
            )
        if "-O" in cmd:
            return CompletedProcess(args=cmd, returncode=255, stdout="", stderr="No existing master")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    monkeypatch.setattr("dnadesign.usr.src.remote.subprocess.run", _run)
    monkeypatch.setattr("dnadesign.usr.src.remote.sys.stdin", _FakeTTY(is_tty=False))
    monkeypatch.setattr("dnadesign.usr.src.remote.sys.stdout", _FakeTTY(is_tty=False))

    with pytest.raises(RemoteUnavailableError, match="requires a TTY"):
        _remote().warm_auth_session()


def test_warm_auth_bootstraps_missing_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    socket_path = tmp_path / "cm.sock"

    def _run(cmd, **kwargs):
        if "-G" in cmd:
            return CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=(
                    "user alice\n"
                    "hostname scc1.bu.edu\n"
                    "controlmaster auto\n"
                    f"controlpath {socket_path}\n"
                    "controlpersist 600\n"
                ),
                stderr="",
            )
        if "-O" in cmd:
            return CompletedProcess(args=cmd, returncode=0, stdout="Master running\n", stderr="")
        if "-MNf" in cmd:
            socket_path.write_text("", encoding="utf-8")
            return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected ssh command: {cmd}")

    monkeypatch.setattr("dnadesign.usr.src.remote.subprocess.run", _run)
    monkeypatch.setattr("dnadesign.usr.src.remote.sys.stdin", _FakeTTY(is_tty=True))
    monkeypatch.setattr("dnadesign.usr.src.remote.sys.stdout", _FakeTTY(is_tty=True))

    status = _remote().warm_auth_session()
    assert status.socket_live is True
    assert status.control_path == str(socket_path)
