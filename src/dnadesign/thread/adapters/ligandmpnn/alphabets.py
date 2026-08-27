"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/alphabets.py

Deterministic residue-alphabet sidecars for LigandMPNN.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnRequest


@dataclass(frozen=True)
class LigandMpnnResidueAlphabetSidecar:
    """Digest-bound receipt for one materialized omission JSON sidecar."""

    request_id: str
    path: Path
    sha256: str
    residue_count: int
    materialized_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.sha256.startswith("sha256:") or len(self.sha256) != 71:
            raise ValueError("sha256 must be a prefixed SHA256 digest")
        _require_json_path(self.path, field_name="path")
        if self.materialized_path is not None:
            _require_json_path(self.materialized_path, field_name="materialized_path")
        if isinstance(self.residue_count, bool) or self.residue_count <= 0:
            raise ValueError("residue_count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "thread.ligandmpnn.residue_alphabet_sidecar",
            "schema_version": 1,
            "request_id": self.request_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "residue_count": self.residue_count,
        }

    def validate_for(self, request: LigandMpnnRequest) -> None:
        """Fail if this receipt is not the exact sidecar for ``request``."""

        self._validate_request_binding(request)
        materialized_path = self.materialized_path or self.path
        materialized_bytes = _read_regular_file_bytes(materialized_path, label="residue alphabet sidecar")
        if materialized_bytes is None or _digest(materialized_bytes) != self.sha256:
            raise ValueError("residue alphabet sidecar file SHA256 does not match receipt")

    def validate_execution_file(self, request: LigandMpnnRequest) -> None:
        """Validate the final command-bound file after staging is promoted."""

        self._validate_request_binding(request)
        execution_bytes = _read_regular_file_bytes(self.path, label="execution residue alphabet sidecar")
        if execution_bytes is None or _digest(execution_bytes) != self.sha256:
            raise ValueError("execution residue alphabet sidecar SHA256 does not match receipt")

    def _validate_request_binding(self, request: LigandMpnnRequest) -> None:
        if self.request_id != request.request_id:
            raise ValueError("residue alphabet sidecar request_id does not match request")
        expected = _canonical_bytes(request)
        if self.residue_count != len(request.residue_alphabets):
            raise ValueError("residue alphabet sidecar residue_count does not match request")
        if self.sha256 != _digest(expected):
            raise ValueError("residue alphabet sidecar digest does not match request")


def materialize_residue_alphabet_sidecar(
    request: LigandMpnnRequest,
    path: Path,
    *,
    write_path: Path | None = None,
) -> LigandMpnnResidueAlphabetSidecar:
    """Materialize omission JSON while binding the eventual execution path."""

    if not request.residue_alphabets:
        raise ValueError("request has no residue alphabets to materialize")
    _require_json_path(path, field_name="path")
    target = write_path or path
    _require_json_path(target, field_name="write_path")
    content = _canonical_bytes(request)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_new_regular_file_or_validate_existing(target, content)
    return LigandMpnnResidueAlphabetSidecar(
        request_id=request.request_id,
        path=path,
        sha256=_digest(content),
        residue_count=len(request.residue_alphabets),
        materialized_path=target,
    )


def _canonical_bytes(request: LigandMpnnRequest) -> bytes:
    payload = {
        alphabet.residue.upstream_id: alphabet.omitted_amino_acids
        for alphabet in sorted(request.residue_alphabets, key=lambda item: item.residue.upstream_id)
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _require_json_path(path: object, *, field_name: str) -> None:
    if not isinstance(path, Path) or path.suffix.lower() != ".json":
        raise ValueError(f"{field_name} must be a Path ending in .json")
    if str(path).startswith("~"):
        raise ValueError(f"{field_name} must not begin with '~'")


def _write_new_regular_file_or_validate_existing(target: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o644)
    except FileExistsError:
        existing = _read_regular_file_bytes(target, label="sidecar target")
        if existing is None:
            raise ValueError("sidecar target changed during materialization")
        if existing != content:
            raise FileExistsError(f"refusing to overwrite different residue alphabet sidecar: {target}")
        return
    except OSError as error:
        raise ValueError("sidecar target must be a regular file") from error
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("sidecar write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_file_bytes(path: Path, *, label: str) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"{label} must be a regular file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)
