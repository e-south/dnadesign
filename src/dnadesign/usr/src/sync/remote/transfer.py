"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sync/remote/transfer.py

Staging and atomic promotion helpers for USR dataset sync transfers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ...contracts import VerificationError
from ...events.append import MAX_EVENT_RECORD_BYTES, event_log_lock

EVENT_LOG_FILENAME = ".events.log"


@dataclass(frozen=True, slots=True)
class EventLogRevision:
    """One descriptor-validated local event-log revision."""

    exists: bool
    device: int | None = None
    inode: int | None = None
    size_bytes: int | None = None
    sha256: str | None = None

    def content_revision(self) -> EventLogContentRevision:
        """Project descriptor identity into transport-comparable content."""

        return EventLogContentRevision(
            exists=self.exists,
            size_bytes=int(self.size_bytes or 0),
            sha256=self.sha256,
        )


@dataclass(frozen=True, slots=True)
class EventLogContentRevision:
    """One event-log content identity shared by local and remote transports."""

    exists: bool
    size_bytes: int
    sha256: str | None


def _event_log_revision_locked(event_path: Path) -> EventLogRevision:
    """Read one event-log revision while its stable sidecar lock is held."""

    event_path = Path(event_path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(event_path, flags)
    except FileNotFoundError:
        return EventLogRevision(exists=False)
    except OSError as exc:
        raise VerificationError(f"Local event log is unavailable for pull revision capture: {event_path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise VerificationError(f"Local event log must be one regular file: {event_path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 16):
            digest.update(chunk)
        completed = os.fstat(descriptor)
        try:
            current = event_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise VerificationError(f"Local event log changed during pull revision capture: {event_path}") from exc
        opened_identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(completed.st_mode)
            or opened_identity != (completed.st_dev, completed.st_ino)
            or opened.st_size != completed.st_size
            or opened_identity != (current.st_dev, current.st_ino)
        ):
            raise VerificationError(f"Local event log changed during pull revision capture: {event_path}")
        return EventLogRevision(
            exists=True,
            device=completed.st_dev,
            inode=completed.st_ino,
            size_bytes=completed.st_size,
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def capture_event_log_revision(event_path: Path) -> EventLogRevision:
    """Capture the local event-log precondition for one staged full pull."""

    event_path = Path(event_path)
    with event_log_lock(event_path):
        return _event_log_revision_locked(event_path)


def capture_locked_event_log_content_revision(event_path: Path) -> EventLogContentRevision:
    """Capture event content while the caller holds its stable sidecar lock."""

    return _event_log_revision_locked(Path(event_path)).content_revision()


def capture_validated_event_log_revision(event_path: Path) -> EventLogRevision:
    """Capture one stable, complete UTF-8 JSONL revision without mutation."""

    event_path = Path(event_path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(event_path, flags)
    except FileNotFoundError:
        return EventLogRevision(exists=False)
    except OSError as exc:
        raise VerificationError(f"Event log is unavailable for validation: {event_path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise VerificationError(f"Event log must be one regular file: {event_path}")
        digest = hashlib.sha256()
        pending = bytearray()
        line_number = 0
        while chunk := os.read(descriptor, 1 << 16):
            digest.update(chunk)
            pending.extend(chunk)
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    if len(pending) >= MAX_EVENT_RECORD_BYTES:
                        raise VerificationError(
                            f"Event log record {line_number + 1} exceeds the "
                            f"{MAX_EVENT_RECORD_BYTES}-byte encoded-record limit: {event_path}"
                        )
                    break
                if newline + 1 > MAX_EVENT_RECORD_BYTES:
                    raise VerificationError(
                        f"Event log record {line_number + 1} exceeds the "
                        f"{MAX_EVENT_RECORD_BYTES}-byte encoded-record limit: {event_path}"
                    )
                line_number += 1
                raw_line = bytes(pending[:newline])
                del pending[: newline + 1]
                if not raw_line:
                    raise VerificationError(
                        f"Event log contains a blank JSONL record at line {line_number}: {event_path}"
                    )
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise VerificationError(
                        f"Event log is not valid UTF-8 at line {line_number}: {event_path}"
                    ) from exc
                except json.JSONDecodeError as exc:
                    raise VerificationError(
                        f"Event log contains malformed JSON at line {line_number}: {event_path}"
                    ) from exc
                if not isinstance(record, dict):
                    raise VerificationError(f"Event log record {line_number} is not a JSON object: {event_path}")
        if pending:
            raise VerificationError(f"Event log ends with a partial JSONL record: {event_path}")
        completed = os.fstat(descriptor)
        try:
            current = event_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise VerificationError(f"Event log changed during validation: {event_path}") from exc
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(completed.st_mode)
            or identity != (completed.st_dev, completed.st_ino)
            or opened.st_size != completed.st_size
            or identity != (current.st_dev, current.st_ino)
        ):
            raise VerificationError(f"Event log changed during validation: {event_path}")
        return EventLogRevision(
            exists=True,
            device=completed.st_dev,
            inode=completed.st_ino,
            size_bytes=completed.st_size,
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def capture_validated_event_log_content_revision(event_path: Path) -> EventLogContentRevision:
    """Project one validated local event-log revision into transport identity."""

    return capture_validated_event_log_revision(event_path).content_revision()


def digest_event_log_prefix(event_path: Path, size_bytes: int) -> str:
    """Hash one exact complete-line prefix from a stable regular event log."""

    event_path = Path(event_path)
    if size_bytes < 0:
        raise VerificationError("Event-log prefix size must be non-negative")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(event_path, flags)
    except FileNotFoundError as exc:
        raise VerificationError(f"Local event log is missing during prefix verification: {event_path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise VerificationError(f"Local event log must be one regular file: {event_path}")
        if opened.st_size < size_bytes:
            raise VerificationError("Remote event log is not a prefix of the locked local event log")
        digest = hashlib.sha256()
        remaining = size_bytes
        final_byte = b""
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 16))
            if not chunk:
                raise VerificationError("Remote event log is not a prefix of the locked local event log")
            digest.update(chunk)
            final_byte = chunk[-1:]
            remaining -= len(chunk)
        if size_bytes and final_byte != b"\n":
            raise VerificationError("Remote event log is not a prefix of the locked local event log: partial line")
        completed = os.fstat(descriptor)
        try:
            current = event_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise VerificationError(f"Local event log changed during prefix verification: {event_path}") from exc
        opened_identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(completed.st_mode)
            or opened_identity != (completed.st_dev, completed.st_ino)
            or opened.st_size != completed.st_size
            or opened_identity != (current.st_dev, current.st_ino)
        ):
            raise VerificationError(f"Local event log changed during prefix verification: {event_path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def make_pull_staging_dir(root: Path, dataset: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset.replace("/", "__")
    return Path(tempfile.mkdtemp(prefix=f".usr-pull-{safe_dataset}-", dir=str(root)))


def copy_file_atomic(src: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.usr-sync-", dir=str(dst.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def collect_staged_entries(staged: Path, *, skip_snapshots: bool) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    for src_path in sorted(Path(staged).rglob("*")):
        rel = src_path.relative_to(staged)
        if not rel.parts:
            continue
        rel_text = rel.as_posix()
        if rel_text in {
            "records.parquet",
            EVENT_LOG_FILENAME,
            ".events.lock",
            ".usr.lock",
            ".usr.transfer.lock",
        }:
            continue
        if rel.parts[0].startswith(".usr.lease."):
            continue
        if skip_snapshots and rel.parts[0] == "_snapshots":
            continue
        if src_path.is_symlink():
            raise VerificationError(f"Staged pull payload contains symlink entry: {rel_text}")
        if src_path.is_dir() or src_path.is_file():
            entries.append((src_path, rel))
            continue
        raise VerificationError(f"Staged pull payload contains unsupported entry type: {rel_text}")
    return entries


def _staged_event_log(staged: Path) -> Path | None:
    event_path = Path(staged) / EVENT_LOG_FILENAME
    try:
        event_stat = event_path.lstat()
    except FileNotFoundError:
        return None
    if event_path.is_symlink():
        raise VerificationError(f"Staged pull payload contains symlink entry: {EVENT_LOG_FILENAME}")
    if not stat.S_ISREG(event_stat.st_mode):
        raise VerificationError(f"Staged pull payload contains unsupported event-log entry: {EVENT_LOG_FILENAME}")
    return event_path


def _promote_event_log_locked(staged_event: Path | None, destination: Path) -> None:
    """Install the staged event log while the destination event lock is held."""

    destination = Path(destination)
    if staged_event is not None:
        copy_file_atomic(staged_event, destination)
        return
    try:
        destination.unlink()
    except FileNotFoundError:
        pass


def promote_staged_pull(
    staged: Path,
    dest: Path,
    *,
    primary_only: bool,
    skip_snapshots: bool,
    expected_event_revision: EventLogRevision | None,
) -> None:
    """Promote one validated pull under an explicit event-log precondition.

    Primary-only pulls require no event revision because they never touch the
    event log. Full pulls require the revision captured before transfer and
    abort before destination mutation if that local state changed. Promotion
    does not merge event histories or retry against a newer local revision.
    """

    if primary_only and expected_event_revision is not None:
        raise ValueError("Primary-only pull promotion must not declare an event-log revision.")
    if not primary_only and expected_event_revision is None:
        raise ValueError("Full pull promotion requires its pre-transfer event-log revision.")

    staged = Path(staged)
    dest = Path(dest)
    staged_primary = staged / "records.parquet"
    if not staged_primary.exists():
        raise VerificationError(f"Staged pull payload missing records.parquet: {staged_primary}")
    if staged_primary.is_symlink():
        raise VerificationError(f"Staged pull payload contains symlink entry: {staged_primary.name}")
    if not staged_primary.is_file():
        raise VerificationError(f"Staged pull payload contains unsupported records entry: {staged_primary.name}")

    staged_event = _staged_event_log(staged)
    staged_entries = collect_staged_entries(staged, skip_snapshots=skip_snapshots)

    if primary_only:
        dest.mkdir(parents=True, exist_ok=True)
        copy_file_atomic(staged_primary, dest / "records.parquet")
        return

    destination_event = dest / EVENT_LOG_FILENAME
    with event_log_lock(destination_event):
        if (
            expected_event_revision is not None
            and _event_log_revision_locked(destination_event) != expected_event_revision
        ):
            raise VerificationError(
                "Local event log changed while the full pull was staged; refusing destination promotion."
            )
        dest.mkdir(parents=True, exist_ok=True)
        copy_file_atomic(staged_primary, dest / "records.parquet")

        kept_paths: set[str] = {"records.parquet"}
        for src_path, rel in staged_entries:
            rel_text = rel.as_posix()
            kept_paths.add(rel_text)
            dst_path = dest / rel
            if src_path.is_dir():
                dst_path.mkdir(parents=True, exist_ok=True)
                continue
            copy_file_atomic(src_path, dst_path)

        keep_with_parents: set[str] = {
            EVENT_LOG_FILENAME,
            ".events.lock",
            ".usr.lock",
            ".usr.transfer.lock",
        }
        for rel_text in kept_paths:
            keep_with_parents.add(rel_text)
            parent = Path(rel_text).parent
            while str(parent) != ".":
                keep_with_parents.add(parent.as_posix())
                parent = parent.parent

        for local_path in sorted(dest.rglob("*"), key=lambda p: (len(p.parts), p.as_posix()), reverse=True):
            rel = local_path.relative_to(dest)
            rel_text = rel.as_posix()
            if rel_text in keep_with_parents:
                continue
            if rel.parts and rel.parts[0].startswith(".usr.lease."):
                continue
            if skip_snapshots and rel.parts and rel.parts[0] == "_snapshots":
                continue
            if local_path.is_file() or local_path.is_symlink():
                local_path.unlink()
                continue
            try:
                local_path.rmdir()
            except OSError:
                pass

        _promote_event_log_locked(staged_event, destination_event)
