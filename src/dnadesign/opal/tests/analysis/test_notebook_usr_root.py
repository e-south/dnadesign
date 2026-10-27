"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_notebook_usr_root.py

Tests exact USR coordinate binding for generated OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.opal.src.analysis.notebook_scope import resolve_notebook_usr_root


def test_notebook_usr_root_accepts_matching_invocation(monkeypatch, tmp_path: Path) -> None:
    operator_root = tmp_path / "operator-data"
    operator_root.mkdir()
    monkeypatch.setenv("OPAL_NOTEBOOK_USR_ROOT", str(operator_root))

    assert resolve_notebook_usr_root(operator_root) == operator_root.resolve()


def test_notebook_usr_root_uses_serialized_coordinate_without_invocation(monkeypatch, tmp_path: Path) -> None:
    operator_root = tmp_path / "operator-data"
    operator_root.mkdir()
    monkeypatch.delenv("OPAL_NOTEBOOK_USR_ROOT", raising=False)

    assert resolve_notebook_usr_root(operator_root) == operator_root.resolve()


def test_notebook_usr_root_rejects_invocation_without_serialized_coordinate(monkeypatch, tmp_path: Path) -> None:
    operator_root = tmp_path / "operator-data"
    operator_root.mkdir()
    monkeypatch.setenv("OPAL_NOTEBOOK_USR_ROOT", str(operator_root))

    with pytest.raises(RuntimeError, match="regenerate it with opal --usr-root"):
        resolve_notebook_usr_root(None)


def test_notebook_usr_root_rejects_mismatched_invocation(monkeypatch, tmp_path: Path) -> None:
    serialized_root = tmp_path / "serialized-data"
    invocation_root = tmp_path / "invocation-data"
    serialized_root.mkdir()
    invocation_root.mkdir()
    monkeypatch.setenv("OPAL_NOTEBOOK_USR_ROOT", str(invocation_root))

    with pytest.raises(RuntimeError, match="does not match the generated notebook"):
        resolve_notebook_usr_root(serialized_root)
