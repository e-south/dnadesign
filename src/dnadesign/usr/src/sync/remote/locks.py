"""Persistent SSH lock sessions for remote dataset transactions."""

from __future__ import annotations

import shlex
import subprocess
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from queue import Empty, Queue

from ...contracts import TransferError


@dataclass(frozen=True, slots=True)
class _RemoteLockSession:
    process: subprocess.Popen[str]

    def require_alive(self, *, failure_label: str, ssh_target: str) -> None:
        if self.process.poll() is not None:
            raise TransferError(f"Remote {failure_label} was lost on {ssh_target}")


@dataclass(frozen=True, slots=True)
class _RemoteEventLogLease:
    owner: object
    dataset: str
    token: str
    session: _RemoteLockSession


@dataclass(frozen=True, slots=True)
class _RemoteDatasetLease:
    owner: object
    dataset: str
    token: str
    session: _RemoteLockSession


def _lease_path(dataset_path: str, kind: str, token: str) -> str:
    if kind not in {"dataset", "event"}:
        raise ValueError(f"Unsupported remote lease kind: {kind}")
    if not token or not token.isascii() or not token.isalnum():
        raise ValueError("Remote lease token must be non-empty ASCII alphanumeric text")
    return str(PurePosixPath(dataset_path) / f".usr.lease.{kind}.{token}")


def _lease_owner_script(dataset_path: str, kind: str, token: str) -> str:
    """Create an fd-held lease marker with identity-checked cleanup.

    Normal exit removes the marker only while its pathname still resolves to
    fd 6's inode. SIGKILL, a missing Linux procfs fd view, or an identity
    mismatch can leave one small excluded runtime marker for offline cleanup.
    """

    lease_path = shlex.quote(_lease_path(dataset_path, kind, token))
    quoted_token = shlex.quote(token)
    return (
        f"lease_path={lease_path}; "
        "set -C; "
        'if ! exec 6>"$lease_path"; then exit 75; fi; '
        "set +C; "
        "cleanup() { "
        "path_identity=$(stat -L -c '%d:%i' -- \"$lease_path\" 2>/dev/null || true); "
        "fd_identity=$(stat -L -c '%d:%i' -- \"/proc/$$/fd/6\" 2>/dev/null || true); "
        'if [ -n "$path_identity" ] && [ "$path_identity" = "$fd_identity" ]; then '
        'rm -f -- "$lease_path"; '
        "fi; "
        "}; "
        "trap cleanup EXIT; "
        "if ! flock -x -n 6; then exit 75; fi; "
        "trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM; "
        "lease_started=$(LC_ALL=C TZ=UTC0 ps -o lstart= -p \"$$\" | tr -d '[:space:]'); "
        '[ -n "$lease_started" ] || exit 75; '
        f'printf \'%s %s %s\\n\' {quoted_token} "$$" "$lease_started" >&6; '
    )


def _lease_validation_script(dataset_path: str, kind: str, token: str) -> str:
    """Validate the exact marker inode held by the lease owner."""

    lease_path = shlex.quote(_lease_path(dataset_path, kind, token))
    quoted_token = shlex.quote(token)
    return (
        f"lease_path={lease_path}; "
        'if ! exec 6<"$lease_path"; then exit 74; fi; '
        # A swapped pathname opens a different inode and therefore cannot carry
        # the holder's exclusive lock.  From this point on, fd 6 is the lease
        # capability; metadata is never read through the pathname.
        "if flock -x -n 6; then flock -u 6; exit 74; fi; "
        "if ! IFS=' ' read -r lease_token lease_pid lease_started <&6; then exit 74; fi; "
        f'[ "$lease_token" = {quoted_token} ] || exit 74; '
        '[ -n "$lease_started" ] || exit 74; '
        "case \"$lease_pid\" in ''|*[!0-9]*) exit 74 ;; esac; "
        'kill -0 "$lease_pid" 2>/dev/null || exit 74; '
        'current_started=$(LC_ALL=C TZ=UTC0 ps -o lstart= -p "$lease_pid" 2>/dev/null | '
        "tr -d '[:space:]') || exit 74; "
        '[ "$current_started" = "$lease_started" ] || exit 74; '
    )


def dataset_lock_script(dataset_path: str, *, timeout_seconds: int, lease_token: str) -> str:
    dataset_dir = shlex.quote(dataset_path)
    lock_path = shlex.quote(str(PurePosixPath(dataset_path) / ".usr.lock"))
    transfer_lock_path = shlex.quote(str(PurePosixPath(dataset_path) / ".usr.transfer.lock"))
    timeout = max(1, int(timeout_seconds))
    return (
        "set -eu; "
        f"mkdir -p {dataset_dir}; "
        "umask 077; "
        f"exec 9>>{lock_path}; "
        f"if ! flock -x -w {timeout} 9; then printf 'USR_REMOTE_LOCK_TIMEOUT:%s\\n' {lease_token}; exit 73; fi; "
        f"exec 7>>{transfer_lock_path}; "
        f"if ! flock -x -w {timeout} 7; then printf 'USR_REMOTE_LOCK_TIMEOUT:%s\\n' {lease_token}; exit 73; fi; "
        "flock -s 7; "
        f"{_lease_owner_script(dataset_path, 'dataset', lease_token)}"
        f"printf 'USR_REMOTE_LOCK_ACQUIRED:%s\\n' {lease_token}; "
        "IFS= read -r _usr_sync_unlock || true"
    )


def event_log_lock_script(dataset_path: str, *, timeout_seconds: int, lease_token: str) -> str:
    dataset_dir = shlex.quote(dataset_path)
    lock_path = shlex.quote(str(PurePosixPath(dataset_path) / ".events.lock"))
    timeout = max(1, int(timeout_seconds))
    return (
        "set -eu; "
        f"mkdir -p {dataset_dir}; "
        "umask 077; "
        f"exec 8>>{lock_path}; "
        f"if ! flock -s -w {timeout} 8; then printf 'USR_REMOTE_EVENT_LOCK_TIMEOUT:%s\\n' {lease_token}; exit 73; fi; "
        f"{_lease_owner_script(dataset_path, 'event', lease_token)}"
        f"printf 'USR_REMOTE_EVENT_LOCK_ACQUIRED:%s\\n' {lease_token}; "
        "IFS= read -r _usr_sync_unlock || true"
    )


def leased_rsync_program(
    dataset_path: str,
    *,
    dataset_lease_token: str,
    timeout_seconds: int,
    event_mode: str | None = None,
    event_lease_token: str | None = None,
) -> str:
    """Build an rsync server command bound to live dataset and event locks."""

    if event_mode not in {None, "shared", "exclusive"}:
        raise ValueError(f"Unsupported event-lock mode: {event_mode}")
    if (event_mode == "shared") != (event_lease_token is not None):
        raise ValueError("Shared event-lock mode requires exactly one event lease token")
    transfer_lock_path = shlex.quote(str(PurePosixPath(dataset_path) / ".usr.transfer.lock"))
    event_lock_path = shlex.quote(str(PurePosixPath(dataset_path) / ".events.lock"))
    timeout = max(1, int(timeout_seconds))
    script = (
        "set -eu; "
        f"exec 7>>{transfer_lock_path}; "
        f"if ! flock -s -w {timeout} 7; then exit 73; fi; "
        f"{_lease_validation_script(dataset_path, 'dataset', dataset_lease_token)}"
    )
    if event_mode is not None:
        flock_mode = "-s" if event_mode == "shared" else "-x"
        script += f"exec 8>>{event_lock_path}; if ! flock {flock_mode} -w {timeout} 8; then exit 73; fi; "
    if event_lease_token is not None:
        script += _lease_validation_script(dataset_path, "event", event_lease_token)
    script += 'exec rsync "$@"'
    return f"sh -c {shlex.quote(script)} sh"


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def _read_lock_handshake(
    proc: subprocess.Popen[str],
    *,
    acquired_marker: str,
    timeout_marker: str,
    deadline_seconds: int,
) -> tuple[str, int]:
    result: Queue[tuple[str, int] | BaseException] = Queue(maxsize=1)

    def _read() -> None:
        marker = ""
        noise_count = 0
        try:
            if proc.stdout is not None:
                while True:
                    line = proc.stdout.readline()
                    if line == "":
                        break
                    marker = line.strip()
                    if marker in {acquired_marker, timeout_marker}:
                        break
                    if marker:
                        noise_count += 1
            result.put((marker, noise_count))
        except BaseException as exc:  # pragma: no cover - defensive pipe boundary
            result.put(exc)

    reader = threading.Thread(target=_read, name="usr-remote-lock-handshake", daemon=True)
    reader.start()
    try:
        handshake = result.get(timeout=max(1, int(deadline_seconds)))
    except Empty as exc:
        _terminate_process(proc)
        reader.join(timeout=2)
        raise TransferError(f"Remote lock handshake exceeded {max(1, int(deadline_seconds))} seconds") from exc
    if isinstance(handshake, BaseException):
        raise TransferError(f"Remote lock handshake failed: {handshake}") from handshake
    return handshake


def _remote_shell_command(script: str) -> str:
    """Preserve one script argument across OpenSSH's remote-command flattening."""

    return f"sh -lc {shlex.quote(script)}"


@contextmanager
def remote_lock_session(
    ssh_command: Sequence[str],
    script: str,
    *,
    acquired_marker: str,
    timeout_marker: str,
    handshake_token: str,
    timeout_seconds: int,
    failure_label: str,
    ssh_target: str,
) -> Iterator[_RemoteLockSession]:
    """Hold one remote flock until the caller leaves the context."""

    if not handshake_token or not handshake_token.isascii() or not handshake_token.isalnum():
        raise ValueError("Remote lock handshake token must be non-empty ASCII alphanumeric text")
    acquired_marker = f"{acquired_marker}:{handshake_token}"
    timeout_marker = f"{timeout_marker}:{handshake_token}"
    proc = subprocess.Popen(
        [*ssh_command, _remote_shell_command(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline_seconds = max(1, int(timeout_seconds)) + 15
    try:
        marker, noise_count = _read_lock_handshake(
            proc,
            acquired_marker=acquired_marker,
            timeout_marker=timeout_marker,
            deadline_seconds=deadline_seconds,
        )
    except TransferError as exc:
        _terminate_process(proc)
        raise TransferError(f"Failed to acquire remote {failure_label} on {ssh_target}: {exc}") from exc
    if marker != acquired_marker:
        _terminate_process(proc)
        if marker == timeout_marker:
            raise TransferError(
                f"Remote {failure_label} timeout on {ssh_target} after {max(1, int(timeout_seconds))} seconds"
            )
        detail = "missing lock handshake marker; remote stderr suppressed"
        if noise_count:
            detail = f"{detail}; stdout_noise_lines={noise_count}"
        raise TransferError(f"Failed to acquire remote {failure_label} on {ssh_target}: {detail}")

    release_error: str | None = None
    body_raised = False
    session = _RemoteLockSession(process=proc)
    try:
        yield session
    except BaseException:
        body_raised = True
        raise
    finally:
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write("release\n")
                    proc.stdin.flush()
                    proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        if proc.returncode not in (0, None):
            release_error = f"remote lock session exited with code {proc.returncode}; remote stderr suppressed"
        if release_error is not None and not body_raised:
            raise TransferError(f"Remote {failure_label} release failed on {ssh_target}: {release_error}")


__all__ = [
    "dataset_lock_script",
    "event_log_lock_script",
    "leased_rsync_program",
    "remote_lock_session",
]
