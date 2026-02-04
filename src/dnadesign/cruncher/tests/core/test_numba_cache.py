"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_numba_cache.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dnadesign.cruncher.utils.numba_cache import (
    ensure_numba_cache_dir,
    temporary_numba_cache_dir,
)


def test_numba_cache_requires_explicit_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="NUMBA_CACHE_DIR"):
        ensure_numba_cache_dir(tmp_path)


def test_numba_cache_accepts_relative_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    cache_dir = ensure_numba_cache_dir(tmp_path, cache_dir=Path(".cruncher") / "numba_cache")
    expected = (tmp_path / ".cruncher" / "numba_cache").resolve()
    assert cache_dir == expected
    assert expected.exists()
    assert os.environ.get("NUMBA_CACHE_DIR") == str(expected)


def test_numba_cache_respects_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_dir = tmp_path / "custom_cache"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(custom_dir))
    cache_dir = ensure_numba_cache_dir(tmp_path)
    assert cache_dir == custom_dir
    assert custom_dir.exists()


def test_numba_cache_rejects_mismatched_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dir = tmp_path / "env_cache"
    override_dir = tmp_path / "override_cache"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(env_dir))
    with pytest.raises(ValueError, match="NUMBA_CACHE_DIR"):
        ensure_numba_cache_dir(tmp_path, cache_dir=override_dir)


def test_numba_cache_requires_writable_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    readonly_dir = tmp_path / "readonly_cache"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o500)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(readonly_dir))
    try:
        with pytest.raises(RuntimeError):
            ensure_numba_cache_dir(tmp_path)
    finally:
        readonly_dir.chmod(0o700)


def test_temporary_numba_cache_dir_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    with temporary_numba_cache_dir() as cache_dir:
        assert cache_dir.exists()
        assert os.environ.get("NUMBA_CACHE_DIR") == str(cache_dir)
    assert not cache_dir.exists()
