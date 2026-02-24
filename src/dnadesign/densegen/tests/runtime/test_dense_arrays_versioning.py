"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/densegen/tests/runtime/test_dense_arrays_versioning.py

Validate strict dense-arrays version resolution contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import builtins
import importlib.metadata
from pathlib import Path

import pytest

from dnadesign.densegen.src.core.pipeline import versioning


def test_dense_arrays_version_from_uv_lock_rejects_invalid_toml(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("not = [")

    with pytest.raises(ValueError, match="uv.lock"):
        versioning._dense_arrays_version_from_uv_lock(tmp_path)


def test_dense_arrays_version_from_pyproject_rejects_invalid_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("project = { dependencies = [")

    with pytest.raises(ValueError, match="pyproject.toml"):
        versioning._dense_arrays_version_from_pyproject(tmp_path)


def test_resolve_dense_arrays_version_surfaces_metadata_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_import(name, *args, **kwargs):
        if name == "dense_arrays":
            raise ModuleNotFoundError("dense_arrays not installed")
        return _original_import(name, *args, **kwargs)

    def _raise_metadata_error(_name: str) -> str:
        raise RuntimeError("metadata backend unavailable")

    _original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(versioning.importlib.metadata, "version", _raise_metadata_error)
    monkeypatch.setattr(versioning, "_find_project_root", lambda _start: None)

    with pytest.raises(RuntimeError, match="installed package metadata"):
        versioning._resolve_dense_arrays_version(tmp_path / "config.yaml")


def test_resolve_dense_arrays_version_rejects_unresolved_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_import(name, *args, **kwargs):
        if name == "dense_arrays":
            raise ModuleNotFoundError("dense_arrays not installed")
        return _original_import(name, *args, **kwargs)

    def _package_not_found(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    _original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(versioning.importlib.metadata, "version", _package_not_found)
    monkeypatch.setattr(versioning, "_find_project_root", lambda _start: None)

    with pytest.raises(RuntimeError, match="Unable to resolve dense-arrays version"):
        versioning._resolve_dense_arrays_version(tmp_path / "config.yaml")
