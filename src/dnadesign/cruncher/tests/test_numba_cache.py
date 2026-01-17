"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_numba_cache.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dnadesign.cruncher.utils.numba_cache import ensure_numba_cache_dir


def test_numba_cache_defaults_to_cruncher_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text('[project]\nname = "dnadesign"\n')
    cruncher_root = repo_root / "src" / "dnadesign" / "cruncher"
    cruncher_root.mkdir(parents=True)
    workspace = repo_root / "workspaces" / "demo"
    workspace.mkdir(parents=True)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    cache_dir = ensure_numba_cache_dir(workspace)
    expected = cruncher_root / ".cruncher" / "numba_cache"
    assert cache_dir == expected
    assert expected.exists()
    assert os.environ.get("NUMBA_CACHE_DIR") == str(expected)


def test_numba_cache_falls_back_to_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text('[project]\nname = "dnadesign"\n')
    workspace = repo_root / "workspaces" / "demo"
    workspace.mkdir(parents=True)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    cache_dir = ensure_numba_cache_dir(workspace)
    expected = repo_root / ".cruncher" / "numba_cache"
    assert cache_dir == expected
    assert expected.exists()
    assert os.environ.get("NUMBA_CACHE_DIR") == str(expected)


def test_numba_cache_errors_without_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError):
        ensure_numba_cache_dir(tmp_path)


def test_numba_cache_respects_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_dir = tmp_path / "custom_cache"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(custom_dir))
    cache_dir = ensure_numba_cache_dir(tmp_path)
    assert cache_dir == custom_dir
    assert custom_dir.exists()


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
