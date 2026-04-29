"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/gardening.py

Offline gardening helpers for long-lived USR dataset event logs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..storage.parquet import now_utc


@dataclass(frozen=True)
class EventLogGardenResult:
    dataset_id: str
    events_path: str
    archive_path: str | None
    total_lines: int
    retained_lines: int
    archived_lines: int
    before_size_bytes: int
    after_size_bytes: int
    source_sha256: str | None
    mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def garden_event_log(
    dataset: Any,
    *,
    retain_last: int = 1000,
    write: bool = False,
    acknowledge_notify_cursor_reset: bool = False,
    archive_dir_name: str = ".events.archive",
    reason: str = "",
) -> EventLogGardenResult:
    """Archive a long USR `.events.log` and retain only its operational tail.

    Event gardening is an offline maintenance operation. It rewrites the live
    `.events.log`, so callers must stop Notify watchers and reseed cursors
    before using `write=True`.
    """

    if int(retain_last) < 1:
        raise ValueError("retain_last must be >= 1.")
    events_path = Path(dataset.events_path)
    if not events_path.exists():
        return EventLogGardenResult(
            dataset_id=str(dataset.name),
            events_path=events_path.as_posix(),
            archive_path=None,
            total_lines=0,
            retained_lines=0,
            archived_lines=0,
            before_size_bytes=0,
            after_size_bytes=0,
            source_sha256=None,
            mode="write" if write else "dry_run",
        )
    total_lines, source_sha256, tail_lines = _scan_event_log(events_path, retain_last=int(retain_last))
    before_size = events_path.stat().st_size
    retained_lines = min(total_lines, int(retain_last))
    archived_lines = max(0, total_lines - retained_lines)
    if not write:
        return EventLogGardenResult(
            dataset_id=str(dataset.name),
            events_path=events_path.as_posix(),
            archive_path=None,
            total_lines=total_lines,
            retained_lines=retained_lines,
            archived_lines=archived_lines,
            before_size_bytes=before_size,
            after_size_bytes=before_size,
            source_sha256=source_sha256,
            mode="dry_run",
        )
    if not acknowledge_notify_cursor_reset:
        raise ValueError(
            "Refusing to rewrite .events.log without acknowledge_notify_cursor_reset=True; "
            "stop Notify watchers and reseed cursors after gardening."
        )

    timestamp = _safe_timestamp(now_utc())
    archive_dir = Path(dataset.dir) / archive_dir_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"events-{timestamp}-{source_sha256[:12]}.jsonl"
    if archive_path.exists():
        raise FileExistsError(f"Event archive already exists: {archive_path}")
    shutil.copy2(events_path, archive_path)
    _replace_event_log(events_path, tail_lines)
    dataset.log_event(
        "event_log_garden",
        args={
            "reason": str(reason),
            "retain_last": int(retain_last),
            "archive": archive_path.relative_to(Path(dataset.dir)).as_posix(),
            "acknowledge_notify_cursor_reset": True,
        },
        metrics={
            "total_lines": total_lines,
            "retained_lines": retained_lines,
            "archived_lines": archived_lines,
            "before_size_bytes": before_size,
            "after_size_bytes_before_event": events_path.stat().st_size,
        },
        artifacts={"source_sha256": source_sha256},
        maintenance={"offline_required": True, "notify_cursor_reset_required": True},
        actor={"tool": "usr", "run_id": "event-log-garden"},
    )
    return EventLogGardenResult(
        dataset_id=str(dataset.name),
        events_path=events_path.as_posix(),
        archive_path=archive_path.as_posix(),
        total_lines=total_lines,
        retained_lines=retained_lines,
        archived_lines=archived_lines,
        before_size_bytes=before_size,
        after_size_bytes=events_path.stat().st_size,
        source_sha256=source_sha256,
        mode="write",
    )


def _scan_event_log(path: Path, retain_last: int) -> tuple[int, str, list[str]]:
    digest = hashlib.sha256()
    tail: deque[str] = deque(maxlen=retain_last)
    count = 0
    with Path(path).open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            line = raw.decode("utf-8")
            stripped = line.strip()
            if stripped:
                try:
                    json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON event at {path}:{count + 1}") from exc
            tail.append(line)
            count += 1
    return count, digest.hexdigest(), list(tail)


def _replace_event_log(path: Path, lines: list[str]) -> None:
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=".events.",
        suffix=".log",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.writelines(lines)
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _safe_timestamp(value: str) -> str:
    return value.replace(":", "").replace("+", "Z").replace(".", "")
