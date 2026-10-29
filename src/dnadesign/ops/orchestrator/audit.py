"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/audit.py

Private, atomic execution progress and prior-attempt preservation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from uuid import uuid4


def _private_open(path: Path, flags: int) -> int:
    descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"Audit storage must be a regular file with one link: {path}")
    return descriptor


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    contents = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ExecutionAudit:
    """Reserve one audit path; uncertain attempts must be reviewed before reuse."""

    def __init__(self, path: Path, plan: dict[str, object]) -> None:
        self.path = path
        self._lock: int | None = None
        self.execution: dict[str, object] = {
            "ok": False,
            "failed_phase": None,
            "status": "prepared",
            "active_command": None,
            "commands": [],
            "retry_requires_review": False,
        }
        self.payload = {"attempt_id": str(uuid4()), "plan": plan, "execution": self.execution}

    def __enter__(self) -> ExecutionAudit:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock = _private_open(self.path.with_name(f".{self.path.name}.lock"), os.O_CREAT | os.O_RDWR)
            os.fchmod(self._lock, 0o600)
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._preserve_previous()
            self._save()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args) -> None:
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def _preserve_previous(self) -> None:
        try:
            descriptor = _private_open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return
        with os.fdopen(descriptor, "rb") as handle:
            contents = handle.read()
        previous = json.loads(contents)
        execution = previous.get("execution", {})
        if not execution or (
            execution.get("active_command") is not None
            or execution.get("retry_requires_review", not execution.get("ok", False))
        ):
            raise ValueError(
                "Review the prior execution audit before retrying: commands may have taken effect. "
                "Reconcile their outcomes, then use a new audit path for any intended execution."
            )
        history = self.path.parent / f"{self.path.stem}.attempts"
        if history.is_symlink():
            raise ValueError("Audit history directory must not be a symbolic link.")
        history.mkdir(mode=0o700, exist_ok=True)
        destination = history / f"{hashlib.sha256(contents).hexdigest()}.json"
        try:
            descriptor = _private_open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            with os.fdopen(_private_open(destination, os.O_RDONLY), "rb") as handle:
                if handle.read() != contents:
                    raise ValueError("Prior audit history differs from the recorded attempt.")
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())

    def _save(self) -> None:
        _write_atomic(self.path, self.payload)

    def begin_command(self, *, phase: str, command: str, index: int) -> None:
        self.execution.update(
            status="running",
            active_command={"phase": phase, "command": command, "index": index},
            retry_requires_review=True,
        )
        self._save()

    def record_command(self, command: dict[str, object]) -> None:
        self.execution["commands"].append(command)
        self.execution["active_command"] = None
        self._save()

    def finish(self, *, ok: bool, failed_phase: str | None) -> None:
        commands = self.execution["commands"]
        self.execution.update(
            ok=ok,
            failed_phase=failed_phase,
            status="completed" if ok else "failed",
            retry_requires_review=not ok and any(command["outcome"] != "not_started" for command in commands),
        )
        self._save()
