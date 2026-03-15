"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_remote_lock_handshake.py

Tests for remote lock handshake noise tolerance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.usr.src.config import SSHRemoteConfig
from dnadesign.usr.src.errors import TransferError
from dnadesign.usr.src.remote import SSHRemote


class _FakeStdout:
    def __init__(self, lines: list[str]):
        self._lines = list(lines)

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


class _FakeStderr:
    def __init__(self, text: str = ""):
        self._text = text

    def read(self) -> str:
        return self._text


class _FakeStdin:
    def __init__(self):
        self.writes: list[str] = []

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProc:
    def __init__(self, *, stdout_lines: list[str], stderr_text: str = ""):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self.stderr = _FakeStderr(stderr_text)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout: int | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.returncode = 1


def _remote() -> SSHRemote:
    return SSHRemote(
        SSHRemoteConfig(
            name="bu-scc",
            host="scc1.bu.edu",
            user="alice",
            base_dir="/project/alice/usr_datasets",
        )
    )


def test_dataset_transfer_lock_ignores_stdout_noise_before_marker(monkeypatch) -> None:
    fake_proc = _FakeProc(stdout_lines=["AGENT_MANAGE_RUNTIME_SKILLS=1\n", "USR_REMOTE_LOCK_ACQUIRED\n"])
    monkeypatch.setattr("dnadesign.usr.src.remote.subprocess.Popen", lambda *args, **kwargs: fake_proc)

    with _remote().dataset_transfer_lock("densegen/demo"):
        pass

    assert fake_proc.stdin.writes == ["release\n"]


def test_dataset_transfer_lock_still_raises_timeout_after_noise(monkeypatch) -> None:
    fake_proc = _FakeProc(stdout_lines=["AGENT_MANAGE_RUNTIME_SKILLS=1\n", "USR_REMOTE_LOCK_TIMEOUT\n"])
    monkeypatch.setattr("dnadesign.usr.src.remote.subprocess.Popen", lambda *args, **kwargs: fake_proc)

    try:
        with _remote().dataset_transfer_lock("densegen/demo", timeout_seconds=7):
            pass
    except TransferError as exc:
        assert "timeout" in str(exc).lower()
        return
    raise AssertionError("expected timeout marker to raise TransferError")
