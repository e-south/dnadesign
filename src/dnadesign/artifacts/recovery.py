"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/recovery.py

Recover interrupted artifact-publication directories without trusting names.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import stat
import time
from pathlib import Path

import psutil

from .errors import PublicationError
from .owned_directory import (
    owner_matches_descriptor,
    read_owner_from_descriptor,
    remove_owned_directory,
)

_OWNER_FILE = ".dnadesign-publication-owner.json"
_PUBLICATION_OWNER_SCHEMA = "dnadesign.artifact_publication_owner.v2"
_ROLLBACK_OWNER_SCHEMA = "dnadesign.artifact_rollback_owner.v2"
_PRE_START_TOKEN_PUBLICATION_OWNER_SCHEMA = "dnadesign.artifact_publication_owner.v1"
_PUBLICATION_RECOVERY_SCHEMAS = (
    _PUBLICATION_OWNER_SCHEMA,
    _PRE_START_TOKEN_PUBLICATION_OWNER_SCHEMA,
)
_MAX_STALE_CANDIDATES = 64
_MAX_SUPPORTED_PID = (1 << 31) - 1
_PRIVATE_FILE_MODE = 0o600


def _format_process_start_token(created_unix: float) -> str:
    if created_unix <= 0 or not math.isfinite(created_unix):
        raise ValueError("Process start time must be finite and positive")
    return f"{created_unix:.9f}"


def _canonical_process_start_token(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        canonical = _format_process_start_token(float(value))
    except (OverflowError, ValueError):
        return None
    return canonical if value == canonical else None


def _current_process_start_token() -> str:
    try:
        return _format_process_start_token(psutil.Process(os.getpid()).create_time())
    except (OSError, OverflowError, ValueError, psutil.Error) as exc:
        raise PublicationError("Artifact publication could not establish process identity") from exc


def _owner_process_is_active(pid: int, expected_start_token: object) -> bool | None:
    """Return whether the original owner is active, or None when identity is unknown."""

    if pid <= 0 or pid > _MAX_SUPPORTED_PID:
        return None
    canonical_start_token = _canonical_process_start_token(expected_start_token)
    if canonical_start_token is None:
        return None
    try:
        observed_start_token = _format_process_start_token(psutil.Process(pid).create_time())
    except psutil.NoSuchProcess:
        return False
    except (OSError, OverflowError, ValueError, psutil.Error):
        return None
    return observed_start_token == canonical_start_token


def _owner_process_is_definitely_absent(pid: int) -> bool:
    """Return True only when the operating system reports that no such PID exists."""

    if pid <= 0 or pid > _MAX_SUPPORTED_PID:
        return False
    try:
        psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    except (OSError, OverflowError, ValueError, psutil.Error):
        return False
    return False


def _owner_payload(final: Path) -> dict[str, object]:
    return {
        "schema": _PUBLICATION_OWNER_SCHEMA,
        "target_sha256": hashlib.sha256(os.fsencode(final)).hexdigest(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "pid": os.getpid(),
        "process_start_token": _current_process_start_token(),
        "host": socket.gethostname(),
        "created_unix": time.time(),
    }


def _rollback_owner_payload(owner: dict[str, object]) -> dict[str, object]:
    return {**owner, "schema": _ROLLBACK_OWNER_SCHEMA}


def _write_owner(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(_PRIVATE_FILE_MODE)


def _ensure_owner_on_descriptor(
    descriptor: int,
    payload: dict[str, object],
    *,
    accepted_existing: dict[str, object] | None = None,
) -> None:
    if owner_matches_descriptor(descriptor, payload, owner_file=_OWNER_FILE) or (
        accepted_existing is not None
        and owner_matches_descriptor(descriptor, accepted_existing, owner_file=_OWNER_FILE)
    ):
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        owner_descriptor = os.open(
            _OWNER_FILE,
            flags,
            _PRIVATE_FILE_MODE,
            dir_fd=descriptor,
        )
    except FileExistsError as exc:
        raise PublicationError("Artifact bundle rollback owner sentinel is unsafe") from exc
    try:
        remaining = memoryview((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        while remaining:
            written = os.write(owner_descriptor, remaining)
            if written <= 0:
                raise OSError("Artifact bundle rollback owner sentinel write made no progress")
            remaining = remaining[written:]
        os.fsync(owner_descriptor)
    except BaseException:
        try:
            os.unlink(_OWNER_FILE, dir_fd=descriptor)
        except OSError:
            pass
        raise
    finally:
        os.close(owner_descriptor)


def _remove_owner_from_descriptor(descriptor: int) -> None:
    try:
        os.unlink(_OWNER_FILE, dir_fd=descriptor)
    except FileNotFoundError:
        return


def _owner_payload_is_recoverable(
    payload: dict[str, object],
    *,
    final: Path,
    uid: int | None,
    owner_schema: str | tuple[str, ...],
) -> bool:
    owner_pid = payload.get("pid")
    if isinstance(owner_pid, bool) or not isinstance(owner_pid, int):
        return False
    if owner_pid <= 0 or owner_pid > _MAX_SUPPORTED_PID:
        return False
    schemas = (owner_schema,) if isinstance(owner_schema, str) else owner_schema
    observed_schema = payload.get("schema")
    owner_context_matches = (
        observed_schema in schemas
        and payload.get("target_sha256") == hashlib.sha256(os.fsencode(final)).hexdigest()
        and payload.get("uid") == uid
        and payload.get("host") == socket.gethostname()
    )
    if not owner_context_matches:
        return False
    if observed_schema == _PRE_START_TOKEN_PUBLICATION_OWNER_SCHEMA:
        # v1 was emitted by released code before process-start identity was
        # recorded. It is accepted only for one-way cleanup when the recorded
        # PID is absent; an existing or indeterminate PID is preserved.
        return _owner_process_is_definitely_absent(owner_pid)
    return _owner_process_is_active(owner_pid, payload.get("process_start_token")) is False


def _is_recoverable_directory(
    path: Path,
    *,
    final: Path,
    uid: int | None,
    owner_schema: str | tuple[str, ...],
) -> bool:
    try:
        entry_stat = path.lstat()
        if not stat.S_ISDIR(entry_stat.st_mode) or (uid is not None and entry_stat.st_uid != uid):
            return False
        owner_path = path / _OWNER_FILE
        owner_stat = owner_path.lstat()
        if not stat.S_ISREG(owner_stat.st_mode):
            return False
        payload = json.loads(owner_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and _owner_payload_is_recoverable(
        payload,
        final=final,
        uid=uid,
        owner_schema=owner_schema,
    )


def _open_recoverable_owned_directory(
    parent_descriptor: int,
    name: str,
    *,
    final: Path,
    uid: int | None,
    owner_schema: str | tuple[str, ...],
) -> tuple[int, dict[str, object]] | None:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError:
        return None
    keep_open = False
    try:
        entry_stat = os.fstat(descriptor)
        if uid is not None and entry_stat.st_uid != uid:
            return None
        observed_owner = read_owner_from_descriptor(descriptor, owner_file=_OWNER_FILE)
        if observed_owner is None or not _owner_payload_is_recoverable(
            observed_owner,
            final=final,
            uid=uid,
            owner_schema=owner_schema,
        ):
            return None
        keep_open = True
        return descriptor, observed_owner
    finally:
        if not keep_open:
            os.close(descriptor)


def _remove_recoverable_owned_directory(
    parent_descriptor: int,
    name: str,
    *,
    final: Path,
    uid: int | None,
    owner_schema: str | tuple[str, ...],
) -> bool:
    opened = _open_recoverable_owned_directory(
        parent_descriptor,
        name,
        final=final,
        uid=uid,
        owner_schema=owner_schema,
    )
    if opened is None:
        return False
    descriptor, observed_owner = opened
    try:
        return remove_owned_directory(
            parent_descriptor,
            name,
            descriptor,
            observed_owner,
            owner_file=_OWNER_FILE,
        )
    finally:
        os.close(descriptor)


def _bounded_named_candidates(directory: Path, *, prefix: str) -> list[Path]:
    candidates: list[Path] = []
    for candidate in directory.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        candidates.append(candidate)
        if len(candidates) >= _MAX_STALE_CANDIDATES:
            break
    return candidates


def _recover_owned_adjacent_directories(
    parent_descriptor: int,
    directory: Path,
    *,
    prefix: str,
    final: Path,
    uid: int | None,
    owner_schema: str | tuple[str, ...],
) -> None:
    for candidate in _bounded_named_candidates(directory, prefix=prefix):
        if _is_recoverable_directory(
            candidate,
            final=final,
            uid=uid,
            owner_schema=owner_schema,
        ):
            _remove_recoverable_owned_directory(
                parent_descriptor,
                candidate.name,
                final=final,
                uid=uid,
                owner_schema=owner_schema,
            )


__all__ = []
