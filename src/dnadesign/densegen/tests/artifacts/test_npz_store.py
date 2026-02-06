"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/artifacts/test_npz_store.py

Tests atomic NPZ artifact writing and validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.artifacts import npz_store
from dnadesign.densegen.src.artifacts.npz_store import (
    ArtifactCorruptError,
    ArtifactWriteError,
    write_npz_atomic,
)


def test_write_npz_atomic_commits_via_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_path = tmp_path / "artifacts" / "record.npz"
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def _track_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(npz_store.os, "replace", _track_replace)

    info = write_npz_atomic({"arr": np.array([1, 2, 3], dtype=np.int64)}, out_path)

    assert out_path.exists()
    assert out_path.with_suffix(".npz.tmp").exists() is False
    assert calls and calls[-1][1] == out_path
    assert calls[-1][0] != out_path.with_suffix(".npz.tmp")
    assert info.sha256
    assert info.bytes == out_path.stat().st_size


def test_validate_npz_detects_corruption(tmp_path: Path) -> None:
    out_path = tmp_path / "artifacts" / "record.npz"
    write_npz_atomic({"arr": np.array([1, 2, 3], dtype=np.int64)}, out_path)
    out_path.write_bytes(b"not-an-npz")

    with pytest.raises(ArtifactCorruptError):
        npz_store.validate_npz(out_path)


def test_write_npz_atomic_fails_on_directory_fsync_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_path = tmp_path / "artifacts" / "record.npz"

    def _fail_open(*_args, **_kwargs):
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(npz_store.os, "open", _fail_open)

    with pytest.raises(ArtifactWriteError, match="directory fsync unavailable"):
        write_npz_atomic({"arr": np.array([1, 2, 3], dtype=np.int64)}, out_path)
