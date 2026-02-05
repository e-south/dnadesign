"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_dataset_layout.py

Tests for namespaced dataset layout and resolution behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pytest

from dnadesign.usr.src.cli import _resolve_dataset_name_interactive, _resolve_existing_dataset_id, list_datasets
from dnadesign.usr.src.dataset import Dataset


def _init_dataset(root: Path, name: str) -> None:
    ds = Dataset(root, name)
    ds.init(source="test")


def test_list_datasets_supports_namespaces(tmp_path: Path) -> None:
    _init_dataset(tmp_path, "demo")
    _init_dataset(tmp_path, "ns/demo")

    names = list_datasets(tmp_path)
    assert "demo" in names
    assert "ns/demo" in names


def test_resolve_from_cwd_namespaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_dataset(tmp_path, "ns/demo")
    monkeypatch.chdir(tmp_path / "ns" / "demo")

    resolved = _resolve_dataset_name_interactive(tmp_path, None, False)
    assert resolved == "ns/demo"


def test_resolve_unqualified_unique_namespace(tmp_path: Path) -> None:
    _init_dataset(tmp_path, "ns/demo")

    resolved = _resolve_existing_dataset_id(tmp_path, "demo")
    assert resolved == "ns/demo"


def test_resolve_unqualified_ambiguous(tmp_path: Path) -> None:
    _init_dataset(tmp_path, "demo")
    _init_dataset(tmp_path, "ns/demo")

    with pytest.raises(SystemExit):
        _resolve_existing_dataset_id(tmp_path, "demo")


def test_resolve_rejects_parent_segments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _resolve_existing_dataset_id(tmp_path, "../demo")
