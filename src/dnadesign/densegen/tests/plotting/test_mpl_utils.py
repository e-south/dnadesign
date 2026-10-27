"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_mpl_utils.py

Tests for Matplotlib cache configuration helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from dnadesign.densegen.src.utils.mpl_utils import ensure_mpl_cache_dir


def test_ensure_mpl_cache_dir_sets_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    dest = tmp_path / "mpl"
    result = ensure_mpl_cache_dir(dest)
    assert result == dest
    assert os.environ["MPLCONFIGDIR"] == str(dest)


def test_ensure_mpl_cache_dir_requires_writable_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    dest = tmp_path / "readonly"
    dest.mkdir()
    try:
        dest.chmod(stat.S_IREAD | stat.S_IEXEC)
    except PermissionError:
        pytest.skip("chmod not supported for this filesystem")
    if os.access(dest, os.W_OK):
        pytest.skip("unable to make directory read-only on this platform")
    with pytest.raises(RuntimeError, match="not writable"):
        ensure_mpl_cache_dir(dest)


def test_ensure_mpl_cache_dir_defaults_to_repo_cache(monkeypatch) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    result = ensure_mpl_cache_dir()
    repo_root = Path(__file__).resolve().parents[5]
    expected = repo_root / ".cache" / "matplotlib" / "densegen"
    assert result == expected
    assert os.environ["MPLCONFIGDIR"] == str(expected)
