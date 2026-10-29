"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_seqledger_interface.py

The sequence-store interface changes without changing stored record formats.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import importlib
import tomllib
from pathlib import Path


def test_seqledger_is_the_canonical_installable_interface():
    root = Path(__file__).resolve().parents[5]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["scripts"]["seqledger"] == "dnadesign._entrypoints:seqledger"
    assert "usr" not in project["scripts"]
    assert "seqledger" in project["optional-dependencies"]
    assert "usr" not in project["optional-dependencies"]
    api = importlib.import_module("dnadesign.seqledger")
    assert callable(api.Dataset)
    assert callable(api.default_seqledger_root)


def test_seqledger_explicit_existing_dataset_root_is_not_rewritten(tmp_path):
    api = importlib.import_module("dnadesign.seqledger")
    existing = tmp_path / "usr_datasets"
    existing.mkdir()
    assert api.normalize_seqledger_root(existing) == existing.resolve()
