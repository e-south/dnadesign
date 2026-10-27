"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/artifacts/test_npz_store.py

Tests atomic NPZ artifact writing and validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
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


class _ExecutableArrayValue:
    def __reduce__(self):
        return print, ("NPZ_OBJECT_EXECUTED",)


@pytest.mark.parametrize("checksum", ["none", "correct", "wrong"])
def test_validate_npz_never_executes_object_arrays(tmp_path: Path, capsys, checksum: str) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, payload=np.array([_ExecutableArrayValue()], dtype=object))
    expected = {
        "none": None,
        "correct": hashlib.sha256(path.read_bytes()).hexdigest(),
        "wrong": "0" * 64,
    }[checksum]
    with pytest.raises(ArtifactCorruptError):
        npz_store.validate_npz(path, expected_sha256=expected)
    assert capsys.readouterr().out == ""


def test_validate_npz_checks_integrity_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "invalid.npz"
    path.write_bytes(b"not an archive")
    with pytest.raises(ArtifactCorruptError, match="sha256 mismatch"):
        npz_store.validate_npz(path, expected_sha256="0" * 64)


@pytest.mark.parametrize("dtype", [object, [("value", object)]])
def test_write_npz_rejects_object_dtype_without_replacing_file(tmp_path: Path, dtype) -> None:
    path = tmp_path / "record.npz"
    path.write_bytes(b"existing")
    with pytest.raises(ArtifactWriteError, match="object"):
        write_npz_atomic({"payload": np.empty(1, dtype=dtype)}, path)
    assert path.read_bytes() == b"existing"


def test_npz_roundtrips_numeric_and_text_arrays_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "numeric.npz"
    info = write_npz_atomic({"numbers": np.arange(3), "text": np.array(["α", "DNA"])}, path)
    npz_store.validate_npz(path, expected_sha256=info.sha256)
    with np.load(path, allow_pickle=False) as data:
        assert data["numbers"].tolist() == [0, 1, 2]
        assert data["text"].tolist() == ["α", "DNA"]
