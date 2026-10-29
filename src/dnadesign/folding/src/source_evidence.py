"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/source_evidence.py

Descriptor-captured source bytes for immutable Folding publications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .errors import FoldingConfigError


def _lexical_absolute(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


@dataclass(frozen=True)
class CapturedSource:
    """Bytes and identity captured through one no-follow regular-file descriptor."""

    path: Path
    content: bytes
    sha256: str
    fingerprint: tuple[int, int, int, int, int]

    @classmethod
    def capture(cls, path: str | Path, *, label: str) -> CapturedSource:
        source_path = _lexical_absolute(path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(source_path, flags)
        except OSError as exc:
            raise FoldingConfigError(f"{label} is unavailable or unsafe: {source_path}") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise FoldingConfigError(f"{label} is unavailable or unsafe: {source_path}")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                content = handle.read()
            after = os.fstat(descriptor)
            before_fingerprint = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_fingerprint = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_fingerprint != after_fingerprint:
                raise FoldingConfigError(f"{label} changed while its bytes were captured: {source_path}")
            return cls(
                path=source_path,
                content=content,
                sha256=hashlib.sha256(content).hexdigest(),
                fingerprint=after_fingerprint,
            )
        finally:
            os.close(descriptor)

    @property
    def portable_ref(self) -> str:
        return f"sha256:{self.sha256}"

    def verify_unchanged(self, *, label: str) -> None:
        try:
            current = type(self).capture(self.path, label=label)
        except FoldingConfigError as exc:
            raise FoldingConfigError(f"{label} changed during SVG publication: {self.path}") from exc
        if current.sha256 != self.sha256 or current.fingerprint != self.fingerprint:
            raise FoldingConfigError(f"{label} changed during SVG publication: {self.path}")


__all__ = ["CapturedSource"]
