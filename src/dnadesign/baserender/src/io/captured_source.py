"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/io/captured_source.py

Descriptor-captured source bytes for reproducible BaseRender execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapturedSource:
    """One regular file's bytes and identity captured through one descriptor."""

    path: Path
    content: bytes | None
    sha256: str
    size: int
    fingerprint: tuple[int, int, int, int, int]

    @classmethod
    def capture(cls, path: str | Path) -> CapturedSource:
        source_path = Path(path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(source_path, flags)
        except OSError as exc:
            raise ValueError(f"Render source is unavailable or unsafe: {source_path}") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"Render source is unavailable or unsafe: {source_path}")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                content = handle.read()
            after = os.fstat(descriptor)
            before_fingerprint = _fingerprint(before)
            after_fingerprint = _fingerprint(after)
            if before_fingerprint != after_fingerprint:
                raise ValueError(f"Render source changed while evidence was captured: {source_path}")
            return cls(
                path=source_path,
                content=content,
                sha256=hashlib.sha256(content).hexdigest(),
                size=after.st_size,
                fingerprint=after_fingerprint,
            )
        finally:
            os.close(descriptor)

    def portable(self) -> dict[str, int | str]:
        return {"sha256": self.sha256, "bytes": self.size}

    def verify_unchanged(self) -> None:
        try:
            current = type(self).capture(self.path)
        except ValueError as exc:
            raise ValueError(f"Render source changed during execution: {self.path}") from exc
        if current.sha256 != self.sha256 or current.fingerprint != self.fingerprint:
            raise ValueError(f"Render source changed during execution: {self.path}")

    def without_content(self) -> CapturedSource:
        return CapturedSource(
            path=self.path,
            content=None,
            sha256=self.sha256,
            size=self.size,
            fingerprint=self.fingerprint,
        )


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


__all__ = ["CapturedSource"]
