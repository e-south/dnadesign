"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_verify_mode.py

Tests for verification mode resolution and strictness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.usr.src.diff import FileStat, resolve_verify_mode
from dnadesign.usr.src.errors import VerificationError


def test_resolve_verify_mode_hash_requires_sha() -> None:
    remote = FileStat(exists=True, size=10, sha256=None, rows=None, cols=None, mtime="0")
    with pytest.raises(VerificationError):
        resolve_verify_mode("hash", remote)


def test_resolve_verify_mode_auto_falls_back_to_size() -> None:
    remote = FileStat(exists=True, size=10, sha256=None, rows=None, cols=None, mtime="0")
    mode, notes = resolve_verify_mode("auto", remote)
    assert mode == "size"
    assert notes and "Falling back" in notes[0]


def test_resolve_verify_mode_auto_uses_hash_when_available() -> None:
    remote = FileStat(exists=True, size=10, sha256="abc", rows=None, cols=None, mtime="0")
    mode, notes = resolve_verify_mode("auto", remote)
    assert mode == "hash"
    assert notes == []


def test_resolve_verify_mode_parquet_requires_shape() -> None:
    remote = FileStat(exists=True, size=10, sha256=None, rows=None, cols=None, mtime="0")
    with pytest.raises(VerificationError):
        resolve_verify_mode("parquet", remote)


def test_resolve_verify_mode_auto_falls_back_to_parquet() -> None:
    remote = FileStat(exists=True, size=None, sha256=None, rows=7, cols=3, mtime="0")
    mode, notes = resolve_verify_mode("auto", remote)
    assert mode == "parquet"
    assert notes and "Falling back" in notes[0]
