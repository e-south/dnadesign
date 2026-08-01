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
import os
import socket
import stat
import time
from pathlib import Path

from .errors import PublicationError
from .owned_directory import (
    owner_matches_descriptor,
    read_owner_from_descriptor,
    remove_owned_directory,
)

_OWNER_FILE = ".dnadesign-publication-owner.json"
_PUBLICATION_OWNER_SCHEMA = "dnadesign.artifact_publication_owner.v1"
_ROLLBACK_OWNER_SCHEMA = "dnadesign.artifact_rollback_owner.v1"
_MAX_STALE_CANDIDATES = 64
_PRIVATE_FILE_MODE = 0o600


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OverflowError, PermissionError):
        return True
    return True


def _owner_payload(final: Path) -> dict[str, object]:
    return {
        "schema": _PUBLICATION_OWNER_SCHEMA,
        "target_sha256": hashlib.sha256(os.fsencode(final)).hexdigest(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "pid": os.getpid(),
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
    try:
        owner_pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        return False
    schemas = (owner_schema,) if isinstance(owner_schema, str) else owner_schema
    return (
        payload.get("schema") in schemas
        and payload.get("target_sha256") == hashlib.sha256(os.fsencode(final)).hexdigest()
        and payload.get("uid") == uid
        and payload.get("host") == socket.gethostname()
        and not _pid_is_alive(owner_pid)
    )


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
