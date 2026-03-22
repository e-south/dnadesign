"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/package/test_prune_module_layout.py

Layout contract tests for infer prune delegation into USR overlay maintenance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
from pathlib import Path

from dnadesign.infer.src.prune import prune_usr_overlay


def test_prune_usr_overlay_delegates_to_usr_overlay_maintenance() -> None:
    source = inspect.getsource(prune_usr_overlay)
    assert "Dataset(" in source
    assert ".remove_overlay(" in source
    assert "dnadesign.usr.src." not in source


def test_prune_usr_overlay_normalizes_usr_package_root(tmp_path: Path, monkeypatch) -> None:
    usr_pkg_root = tmp_path / "usr_pkg"
    usr_pkg_root.mkdir(parents=True, exist_ok=True)
    (usr_pkg_root / "__init__.py").write_text("# test package root\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeDataset:
        def __init__(self, dataset_root, dataset_name):
            captured["root"] = Path(dataset_root)
            captured["name"] = dataset_name
            self.name = dataset_name

        def remove_overlay(self, namespace: str, mode: str = "archive") -> dict[str, object]:
            captured["namespace"] = namespace
            captured["mode"] = mode
            return {"removed": True}

    monkeypatch.setattr("dnadesign.infer.src.prune.Dataset", _FakeDataset)

    result = prune_usr_overlay(dataset="demo", usr_root=usr_pkg_root, mode="archive")

    assert captured["root"] == (usr_pkg_root / "datasets").resolve()
    assert captured["namespace"] == "infer"
    assert captured["mode"] == "archive"
    assert result["dataset"] == "demo"
