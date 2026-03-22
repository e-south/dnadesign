"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_overlay_maintenance_module.py

Layout and contract tests for dataset-targeted overlay maintenance helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pyarrow as pa

from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def test_overlay_maintenance_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.overlay_maintenance")
    assert hasattr(module, "remove_dataset_overlay")


def test_remove_dataset_overlay_archives_named_overlay(tmp_path: Path) -> None:
    module = importlib.import_module("dnadesign.usr.src.overlay_maintenance")
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="audit", columns_spec="audit__score:float64")

    ds = Dataset(root, "demo")
    ds.init(source="unit")
    ds.import_rows(
        [{"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"}],
        source="unit",
    )
    ids = ds.head(1)["id"].tolist()
    ds.write_overlay_part("audit", pa.table({"id": ids, "audit__score": [0.1]}), key="id")

    result = module.remove_dataset_overlay(root, "demo", "audit", mode="archive")

    assert result["removed"] is True
    assert result["namespace"] == "audit"
    assert result["dataset"] == "demo"
    assert result["mode"] == "archive"
