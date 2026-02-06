"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/artifacts/npz_store.py

Atomic read/write helpers for DenseGen NPZ artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ArtifactInfo:
    ref: str
    sha256: str
    bytes: int


class ArtifactCorruptError(RuntimeError):
    """Raised when an NPZ artifact is missing, unreadable, or mismatched."""


class ArtifactWriteError(RuntimeError):
    """Raised when an NPZ artifact cannot be safely written."""


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    dir_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_npz_atomic(arrays: Mapping[str, np.ndarray], out_path: Path) -> ArtifactInfo:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd: int | None = None
    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{out_path.name}.",
            suffix=".tmp",
            dir=out_path.parent,
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(tmp_fd, "wb") as handle:
            tmp_fd = None
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
        _fsync_dir(out_path.parent)
        size_bytes = int(out_path.stat().st_size)
        sha256 = _sha256_file(out_path)
        return ArtifactInfo(ref=str(out_path), sha256=sha256, bytes=size_bytes)
    except Exception as exc:
        raise ArtifactWriteError(f"Failed to atomically write NPZ artifact at {out_path}: {exc}") from exc
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def validate_npz(path: Path, expected_sha256: str | None = None) -> None:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ArtifactCorruptError(f"NPZ artifact does not exist: {artifact_path}")
    try:
        with np.load(artifact_path, allow_pickle=True) as npz:
            for key in npz.files:
                _ = npz[key].shape
    except Exception as exc:
        raise ArtifactCorruptError(f"Failed to validate NPZ artifact: {artifact_path}") from exc
    if expected_sha256:
        actual = _sha256_file(artifact_path)
        if actual != expected_sha256:
            raise ArtifactCorruptError(
                f"NPZ artifact sha256 mismatch for {artifact_path}: expected={expected_sha256} actual={actual}"
            )


def describe_npz(path: Path, *, expected_sha256: str | None = None) -> ArtifactInfo:
    artifact_path = Path(path)
    validate_npz(artifact_path, expected_sha256=expected_sha256)
    return ArtifactInfo(
        ref=str(artifact_path),
        sha256=_sha256_file(artifact_path),
        bytes=int(artifact_path.stat().st_size),
    )
