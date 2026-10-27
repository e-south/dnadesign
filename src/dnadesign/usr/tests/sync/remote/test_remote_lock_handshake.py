"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/sync/remote/test_remote_lock_handshake.py

Tests for remote lock handshake noise tolerance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
import threading

import pytest

from dnadesign.usr.src.contracts import TransferError
from dnadesign.usr.src.sync.remote import locks as locks_module
from dnadesign.usr.src.sync.remote.config import SSHRemoteConfig
from dnadesign.usr.src.sync.remote.remote import SSHRemote

_PORTABLE_TEST_SHELLS = tuple(
    dict.fromkeys(path for name in ("sh", "dash", "bash") if (path := shutil.which(name)) is not None)
)


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


def _fixed_lease_token(
    monkeypatch,
    token: str = "0123456789abcdef0123456789abcdef",  # pragma: allowlist secret
) -> str:
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.remote.secrets.token_hex", lambda _size: token)
    return token


def _bound(marker: str, token: str) -> str:
    return f"{marker}:{token}\n"


def test_dataset_transfer_lock_ignores_stdout_noise_before_marker(monkeypatch) -> None:
    token = _fixed_lease_token(monkeypatch)
    fake_proc = _FakeProc(stdout_lines=["AGENT_MANAGE_RUNTIME_SKILLS=1\n", _bound("USR_REMOTE_LOCK_ACQUIRED", token)])
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: fake_proc)

    with _remote().dataset_transfer_lock("densegen_demo"):
        pass

    assert fake_proc.stdin.writes == ["release\n"]


def test_dataset_transfer_lock_still_raises_timeout_after_noise(monkeypatch) -> None:
    token = _fixed_lease_token(monkeypatch)
    fake_proc = _FakeProc(stdout_lines=["AGENT_MANAGE_RUNTIME_SKILLS=1\n", _bound("USR_REMOTE_LOCK_TIMEOUT", token)])
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: fake_proc)

    try:
        with _remote().dataset_transfer_lock("densegen_demo", timeout_seconds=7):
            pass
    except TransferError as exc:
        assert "timeout" in str(exc).lower()
        return
    raise AssertionError("expected timeout marker to raise TransferError")


def test_failed_handshake_does_not_echo_remote_stdout_noise(monkeypatch) -> None:
    _fixed_lease_token(monkeypatch)
    fake_proc = _FakeProc(stdout_lines=["PRIVATE_VALUE=do-not-echo\n"])
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: fake_proc)

    with pytest.raises(TransferError) as captured:
        with _remote().dataset_transfer_lock("densegen_demo"):
            pass

    assert "PRIVATE_VALUE" not in str(captured.value)
    assert "stdout_noise_lines=1" in str(captured.value)


def test_event_log_transfer_lock_yields_only_one_active_lease(monkeypatch) -> None:
    token = _fixed_lease_token(monkeypatch)
    fake_proc = _FakeProc(stdout_lines=[_bound("USR_REMOTE_EVENT_LOCK_ACQUIRED", token)])
    commands: list[list[str]] = []

    def _popen(command, **_kwargs):
        commands.append(command)
        return fake_proc

    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", _popen)

    remote = _remote()
    with remote.event_log_transfer_lock("densegen_demo") as lease:
        assert lease.owner is remote
        assert lease.dataset == "densegen_demo"
        assert lease.token in commands[0][-1]
        assert "flock -s" in commands[0][-1]

    assert fake_proc.stdin.writes == ["release\n"]


def test_event_log_transfer_lock_reports_its_own_timeout(monkeypatch) -> None:
    token = _fixed_lease_token(monkeypatch)
    fake_proc = _FakeProc(stdout_lines=[_bound("USR_REMOTE_EVENT_LOCK_TIMEOUT", token)])
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: fake_proc)

    try:
        with _remote().event_log_transfer_lock("densegen_demo", timeout_seconds=11):
            pass
    except TransferError as exc:
        assert "event-log lock" in str(exc)
        assert "11" in str(exc)
        return
    raise AssertionError("expected event-log timeout marker to raise TransferError")


class _BlockingStdout:
    def __init__(self) -> None:
        self.released = threading.Event()

    def readline(self) -> str:
        self.released.wait(timeout=5)
        return ""


class _BlockingProc(_FakeProc):
    def __init__(self) -> None:
        super().__init__(stdout_lines=[])
        self.stdout = _BlockingStdout()

    def kill(self) -> None:
        super().kill()
        self.stdout.released.set()


def test_lock_handshake_has_a_local_deadline() -> None:
    proc = _BlockingProc()

    with pytest.raises(TransferError, match="exceeded 1 seconds"):
        locks_module._read_lock_handshake(
            proc,
            acquired_marker="ACQUIRED",
            timeout_marker="TIMEOUT",
            deadline_seconds=1,
        )

    assert proc.returncode == 1


def test_failed_lock_handshake_terminates_before_draining_stderr(monkeypatch) -> None:
    _fixed_lease_token(monkeypatch)
    proc = _FakeProc(stdout_lines=[])

    class _GuardedStderr:
        def read(self) -> str:
            assert proc.returncode is not None
            return "ssh failed"

    proc.stderr = _GuardedStderr()
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: proc)

    with pytest.raises(TransferError, match="remote stderr suppressed"):
        with _remote().dataset_transfer_lock("densegen_demo"):
            pass

    assert proc.returncode == 1


def test_lock_session_releases_before_propagating_body_exception(monkeypatch) -> None:
    token = _fixed_lease_token(monkeypatch)
    proc = _FakeProc(stdout_lines=[_bound("USR_REMOTE_LOCK_ACQUIRED", token)])
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: proc)

    with pytest.raises(RuntimeError, match="body failed"):
        with _remote().dataset_transfer_lock("densegen_demo"):
            raise RuntimeError("body failed")

    assert proc.stdin.writes == ["release\n"]
    assert proc.returncode == 0


def test_fixed_startup_marker_cannot_forge_a_lock_handshake(monkeypatch) -> None:
    token = _fixed_lease_token(monkeypatch)
    proc = _FakeProc(
        stdout_lines=[
            "USR_REMOTE_LOCK_ACQUIRED\n",
            _bound("USR_REMOTE_LOCK_ACQUIRED", token),
        ]
    )
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: proc)

    with _remote().dataset_transfer_lock("densegen_demo"):
        pass

    assert proc.stdin.writes == ["release\n"]


def test_fixed_startup_marker_alone_is_rejected(monkeypatch) -> None:
    _fixed_lease_token(monkeypatch)
    proc = _FakeProc(stdout_lines=["USR_REMOTE_LOCK_ACQUIRED\n"])
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: proc)

    with pytest.raises(TransferError, match="missing lock handshake marker"):
        with _remote().dataset_transfer_lock("densegen_demo"):
            pass

    assert proc.stdin.writes == []


def test_failed_handshake_never_exposes_remote_stderr(monkeypatch) -> None:
    _fixed_lease_token(monkeypatch)
    secret = "SECRET_TOKEN=" + ("sensitive-value-" * 10_000)  # pragma: allowlist secret
    proc = _FakeProc(stdout_lines=[], stderr_text=secret)
    popen_kwargs: dict[str, object] = {}

    def _popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return proc

    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", _popen)

    with pytest.raises(TransferError) as captured:
        with _remote().dataset_transfer_lock("densegen_demo"):
            pass

    message = str(captured.value)
    assert "SECRET_TOKEN" not in message
    assert "sensitive-value" not in message
    assert len(message) < 300
    assert popen_kwargs["stderr"] is subprocess.DEVNULL


def test_failed_release_never_exposes_remote_stderr(monkeypatch) -> None:
    token = _fixed_lease_token(monkeypatch)

    class _ReleaseFailureProc(_FakeProc):
        def wait(self, timeout: int | None = None) -> int:
            self.returncode = 9
            return self.returncode

    secret = "SECRET_TOKEN=" + ("sensitive-value-" * 10_000)  # pragma: allowlist secret
    proc = _ReleaseFailureProc(
        stdout_lines=[_bound("USR_REMOTE_LOCK_ACQUIRED", token)],
        stderr_text=secret,
    )
    monkeypatch.setattr("dnadesign.usr.src.sync.remote.locks.subprocess.Popen", lambda *args, **kwargs: proc)

    with pytest.raises(TransferError) as captured:
        with _remote().dataset_transfer_lock("densegen_demo"):
            pass

    message = str(captured.value)
    assert "SECRET_TOKEN" not in message
    assert "sensitive-value" not in message
    assert "exited with code 9" in message
    assert len(message) < 300


def test_remote_shell_command_preserves_the_script_as_one_argument() -> None:
    script = "set -eu; value='two words'; printf '%s\\n' \"$value\""

    completed = subprocess.run(
        ["sh", "-c", locks_module._remote_shell_command(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == "two words\n"
    assert completed.stderr == ""


def test_remote_lease_scripts_use_stable_process_start_identity() -> None:
    dataset_path = "/project/alice/usr_datasets/densegen_demo"
    dataset_token = "DatasetLease123"
    event_token = "EventLease456"
    scripts = (
        locks_module.dataset_lock_script(
            dataset_path,
            timeout_seconds=30,
            lease_token=dataset_token,
        ),
        locks_module.event_log_lock_script(
            dataset_path,
            timeout_seconds=30,
            lease_token=event_token,
        ),
        locks_module.leased_rsync_program(
            dataset_path,
            dataset_lease_token=dataset_token,
            timeout_seconds=30,
            event_mode="shared",
            event_lease_token=event_token,
        ),
    )

    for script in scripts:
        assert "LC_ALL=C TZ=UTC0 ps -o lstart=" in script
        assert script.count("ps -o lstart=") == script.count("LC_ALL=C TZ=UTC0 ps -o lstart=")


@pytest.mark.parametrize("shell", _PORTABLE_TEST_SHELLS)
@pytest.mark.parametrize("kind", ["dataset", "event"])
def test_lease_owner_script_rejects_a_preexisting_symlink(tmp_path, shell: str, kind: str) -> None:
    dataset_path = tmp_path / "dataset"
    redirect_target = tmp_path / "redirect-target"
    dataset_path.mkdir()
    redirect_target.mkdir()
    redirected_owner = redirect_target / "owner"
    redirected_owner.write_text("do-not-truncate\n", encoding="utf-8")

    token = "SafeLease123"
    lease_path = dataset_path / f".usr.lease.{kind}.{token}"
    lease_path.symlink_to(redirected_owner)
    script = locks_module._lease_owner_script(str(dataset_path), kind, token)

    completed = subprocess.run(
        [shell, "-c", f"set -eu; {script}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75
    assert lease_path.is_symlink()
    assert redirected_owner.read_text(encoding="utf-8") == "do-not-truncate\n"


def test_lease_owner_uses_its_open_descriptor_across_a_path_swap(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    token = "SafeLease123"
    lease_path = dataset_path / f".usr.lease.dataset.{token}"
    displaced_path = dataset_path / "displaced-lease"
    replacement_path = dataset_path / "replacement"
    replacement_path.write_text("do-not-modify\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_flock.chmod(0o755)
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/bin/sh\n"
        'mv "$LEASE_PATH" "$DISPLACED_PATH"\n'
        'ln -s "$REPLACEMENT_PATH" "$LEASE_PATH"\n'
        "printf 'Mon Jan  1 00:00:00 2024\\n'\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)
    fake_stat = fake_bin / "stat"
    fake_stat.write_text(
        "#!/bin/sh\ncase \"$*\" in\n  *\"$LEASE_PATH\"*) printf '2:2\\n' ;;\n  *) printf '1:1\\n' ;;\nesac\n",
        encoding="utf-8",
    )
    fake_stat.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "LEASE_PATH": str(lease_path),
        "DISPLACED_PATH": str(displaced_path),
        "REPLACEMENT_PATH": str(replacement_path),
    }

    completed = subprocess.run(
        ["sh", "-c", f"set -eu; {locks_module._lease_owner_script(str(dataset_path), 'dataset', token)}"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert lease_path.is_symlink()
    assert replacement_path.read_text(encoding="utf-8") == "do-not-modify\n"
    assert displaced_path.read_text(encoding="utf-8").startswith(f"{token} ")


def test_lease_owner_removes_the_unchanged_marker_on_normal_exit(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    token = "SafeLease123"
    lease_path = dataset_path / f".usr.lease.dataset.{token}"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in {
        "flock": "#!/bin/sh\nexit 0\n",
        "stat": "#!/bin/sh\nprintf '1:1\\n'\n",
    }.items():
        executable = fake_bin / name
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    completed = subprocess.run(
        ["sh", "-c", f"set -eu; {locks_module._lease_owner_script(str(dataset_path), 'dataset', token)}"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert not lease_path.exists()


@pytest.mark.skipif(shutil.which("flock") is None, reason="requires util-linux flock")
def test_lease_validation_rejects_a_post_creation_path_swap(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    token = "SafeLease123"
    lease_path = dataset_path / f".usr.lease.dataset.{token}"
    displaced_path = dataset_path / "displaced-owner"
    replacement_path = dataset_path / "replacement-owner"
    replacement_path.write_text(f"{token} {os.getpid()} forged-start\n", encoding="utf-8")

    holder = subprocess.Popen(
        [
            "sh",
            "-c",
            f"set -eu; {locks_module._lease_owner_script(str(dataset_path), 'dataset', token)}"
            "printf 'ready\\n'; IFS= read -r _release",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline() == "ready\n"
        lease_path.rename(displaced_path)
        lease_path.symlink_to(replacement_path)

        completed = subprocess.run(
            ["sh", "-c", f"set -eu; {locks_module._lease_validation_script(str(dataset_path), 'dataset', token)}"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 74
        assert replacement_path.read_text(encoding="utf-8").endswith("forged-start\n")
    finally:
        if holder.stdin is not None:
            holder.stdin.write("release\n")
            holder.stdin.flush()
        holder.wait(timeout=5)


def test_lease_cleanup_does_not_remove_a_post_creation_replacement(tmp_path) -> None:
    script = locks_module._lease_owner_script(str(tmp_path), "dataset", "SafeLease123")

    assert "rmdir " not in script
    assert 'path_identity" = "$fd_identity' in script
    assert "excluded runtime marker for offline cleanup" in (locks_module._lease_owner_script.__doc__ or "")


@pytest.mark.parametrize("shell", _PORTABLE_TEST_SHELLS)
def test_lease_validation_normalizes_a_missing_marker_to_contract_exit_code(tmp_path, shell: str) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    completed = subprocess.run(
        [
            shell,
            "-c",
            f"set -eu; {locks_module._lease_validation_script(str(dataset_path), 'dataset', 'SafeLease123')}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 74


def test_lease_validation_rejects_a_non_numeric_process_id(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    token = "SafeLease123"
    dataset_path.mkdir(parents=True)
    lease_path = dataset_path / f".usr.lease.dataset.{token}"
    lease_path.write_text(f"{token} --help forged-start\n", encoding="utf-8")

    completed = subprocess.run(
        ["sh", "-c", f"set -eu; {locks_module._lease_validation_script(str(dataset_path), 'dataset', token)}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 74


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("flock") is None or shutil.which("ps") is None,
    reason="requires Linux procfs, procps, and util-linux flock",
)
def test_dataset_transfer_fence_survives_outer_owner_loss(tmp_path) -> None:
    dataset_path = tmp_path / "usr dataset"
    owner_token = "OwnerLease123"
    contender_token = "ContenderLease456"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text(
        "#!/bin/sh\nprintf 'USR_TEST_RSYNC_STARTED\\n'\nIFS= read -r _release || true\n",
        encoding="utf-8",
    )
    fake_rsync.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    owner = subprocess.Popen(
        [
            "sh",
            "-c",
            locks_module.dataset_lock_script(
                str(dataset_path),
                timeout_seconds=8,
                lease_token=owner_token,
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    wrapper: subprocess.Popen[str] | None = None
    contender: subprocess.Popen[str] | None = None
    try:
        assert owner.stdout is not None
        assert owner.stdout.readline() == _bound("USR_REMOTE_LOCK_ACQUIRED", owner_token)

        wrapper_program = locks_module.leased_rsync_program(
            str(dataset_path),
            dataset_lease_token=owner_token,
            timeout_seconds=8,
        )
        wrapper = subprocess.Popen(
            ["sh", "-c", f"{wrapper_program} sentinel"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert wrapper.stdout is not None
        assert wrapper.stdout.readline() == "USR_TEST_RSYNC_STARTED\n"

        owner.kill()
        owner.wait(timeout=5)
        contender = subprocess.Popen(
            [
                "sh",
                "-c",
                locks_module.dataset_lock_script(
                    str(dataset_path),
                    timeout_seconds=8,
                    lease_token=contender_token,
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert contender.stdout is not None
        readable, _, _ = select.select([contender.stdout], [], [], 0.3)
        assert readable == []

        assert wrapper.stdin is not None
        wrapper.stdin.write("release\n")
        wrapper.stdin.flush()
        wrapper.stdin.close()
        assert wrapper.wait(timeout=5) == 0

        assert contender.stdout.readline() == _bound("USR_REMOTE_LOCK_ACQUIRED", contender_token)
        assert contender.stdin is not None
        contender.stdin.write("release\n")
        contender.stdin.flush()
        contender.stdin.close()
        assert contender.wait(timeout=5) == 0
    finally:
        for process in (wrapper, contender, owner):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
