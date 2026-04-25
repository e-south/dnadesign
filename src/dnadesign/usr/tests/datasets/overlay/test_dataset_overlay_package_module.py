"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/overlay/test_dataset_overlay_package_module.py

Layout contract tests for Dataset overlay package decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from dnadesign.testsupport.usr import register_test_namespace
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.datasets.overlay.policy import _overlay_table_from_registry
from dnadesign.usr.src.registry import load_registry, registry_entry


def test_dataset_overlay_package_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.datasets.overlay")


def test_dataset_attach_and_write_overlay_delegate_to_overlay_package() -> None:
    attach_source = inspect.getsource(Dataset.attach)
    write_overlay_source = inspect.getsource(Dataset.write_overlay)
    write_part_source = inspect.getsource(Dataset.write_overlay_part)

    assert "attach_dataset(" in attach_source
    assert "write_overlay_dataset(" in write_overlay_source
    assert "write_overlay_part_dataset(" in write_part_source


def test_overlay_table_from_registry_preserves_ndarray_lists_and_string_nulls(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(
        root,
        namespace="mock",
        columns_spec=(
            "mock__score:float64,mock__vec:list<float64>,mock__tags:list<string>,mock__provider_version:string"
        ),
    )
    entry = registry_entry(load_registry(root, required=True), "mock")
    overlay_df = pd.DataFrame(
        {
            "id": ["row-1", "row-2"],
            "mock__score": [1.0, 2.0],
            "mock__vec": [
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([3.0, 4.0], dtype=np.float64),
            ],
            "mock__tags": [
                np.array(["seq_mean", "anchor_mean"], dtype=object),
                ["anchor_mean"],
            ],
            "mock__provider_version": [np.nan, "v1"],
        }
    )

    table = _overlay_table_from_registry(overlay_df, entry=entry, key="id")

    assert table.schema.field("mock__vec").type == pa.list_(pa.float64())
    assert table.schema.field("mock__tags").type == pa.list_(pa.string())
    assert table.column("mock__vec").to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
    assert table.column("mock__tags").to_pylist() == [["seq_mean", "anchor_mean"], ["anchor_mean"]]
    assert table.column("mock__provider_version").to_pylist() == [None, "v1"]
