"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/views/test_scopes.py

Scoped matrix access tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from dnadesign.latentdna.src.io.matrix_io import write_matrix
from dnadesign.latentdna.src.io.parquet_io import write_table
from dnadesign.latentdna.src.views import scopes
from dnadesign.latentdna.src.views.scopes import clear_scope_caches, resolve_view_scope
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    view_dir = workspace_dir / "outputs" / "views" / "demo_view"
    sample_dir = workspace_dir / "outputs" / "samples" / "demo_sample"
    view_dir.mkdir(parents=True)
    sample_dir.mkdir(parents=True)
    write_matrix(
        view_dir / "matrix.npy",
        np.asarray(
            [
                [1.0, 1.5, 2.0],
                [2.0, 2.5, 3.0],
                [3.0, 3.5, 4.0],
                [4.0, 4.5, 5.0],
            ],
            dtype=np.float32,
        ),
    )
    rows = pa.table({"id": ["row_01", "row_02", "row_03", "row_04"], "group": ["a", "a", "b", "b"]})
    write_table(rows, view_dir / "rows.parquet")
    write_table(rows.take(pa.array([1, 3], type=pa.int64())), sample_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(json.dumps({"params": {"record_key": "id"}}), encoding="utf-8")
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  demo_source:
    kind: parquet
    path: ./inputs/demo.parquet
    record_key: id
    subject_key: id
metadata:
  include: []
views:
  demo_view:
    source: demo_source
    vector: {kind: column, name: embedding}
    coordinate_space_id: demo_space
    tags: {model: demo}
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    return load_workspace_config(workspace_dir)


def test_sample_scope_reuses_cached_exact_matrix_until_sample_rows_change(tmp_path, monkeypatch) -> None:
    clear_scope_caches()
    context = _write_workspace(tmp_path)
    read_count = 0
    original_read_matrix = scopes.read_matrix

    def _counting_read_matrix(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        return original_read_matrix(*args, **kwargs)

    monkeypatch.setattr(scopes, "read_matrix", _counting_read_matrix)

    first_matrix, first_rows, _, _ = resolve_view_scope(
        context,
        view_id="demo_view",
        sample_id="demo_sample",
        alignment_id=None,
    )
    second_matrix, second_rows, _, _ = resolve_view_scope(
        context,
        view_id="demo_view",
        sample_id="demo_sample",
        alignment_id=None,
    )

    np.testing.assert_array_equal(first_matrix, np.asarray([[2.0, 2.5, 3.0], [4.0, 4.5, 5.0]], dtype=np.float32))
    np.testing.assert_array_equal(second_matrix, first_matrix)
    assert first_rows.to_pydict()["id"] == ["row_02", "row_04"]
    assert second_rows.to_pydict()["id"] == ["row_02", "row_04"]
    assert read_count == 1
    assert not second_matrix.flags.writeable

    sample_dir = context.output_root / "samples" / "demo_sample"
    write_table(pa.table({"id": ["row_01"], "group": ["a"]}), sample_dir / "rows.parquet")

    third_matrix, third_rows, _, _ = resolve_view_scope(
        context,
        view_id="demo_view",
        sample_id="demo_sample",
        alignment_id=None,
    )

    np.testing.assert_array_equal(third_matrix, np.asarray([[1.0, 1.5, 2.0]], dtype=np.float32))
    assert third_rows.to_pydict()["id"] == ["row_01"]
    assert read_count == 2
