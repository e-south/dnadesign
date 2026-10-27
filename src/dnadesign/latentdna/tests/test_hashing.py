"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_hashing.py

Hashing helper contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

from dnadesign.latentdna.src.io.hashing import _sha256_file_for_stat, sha256_file


def test_sha256_file_reuses_digest_for_unchanged_file_stat(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"stable payload")
    _sha256_file_for_stat.cache_clear()

    first = sha256_file(path)
    second = sha256_file(path)
    cache_info = _sha256_file_for_stat.cache_info()

    assert first == second
    assert cache_info.hits == 1
    assert cache_info.misses == 1


def test_sha256_file_invalidates_cache_when_file_stat_changes(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"stable payload")
    _sha256_file_for_stat.cache_clear()

    first = sha256_file(path)
    path.write_bytes(b"changed payload with different size")
    second = sha256_file(path)

    assert first != second
    assert _sha256_file_for_stat.cache_info().misses == 2


def test_sha256_file_detects_equal_size_edits_with_restored_mtime(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"first")
    stat = path.stat()
    first = sha256_file(path)
    path.write_bytes(b"other")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert sha256_file(path) != first
