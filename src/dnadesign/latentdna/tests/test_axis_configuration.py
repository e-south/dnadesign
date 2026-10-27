"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_axis_configuration.py

Explicit cohort-axis and association-bin contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars.builders.tf_axis_orientation import build_tf_axis_orientation_audit_scalar


def test_configured_axes_use_arbitrary_cohort_and_association_labels(tmp_path):
    view = tmp_path / "views" / "example"
    view.mkdir(parents=True)
    np.save(view / "matrix.npy", np.array([[0, 0], [2, 0], [0, 2], [0.1, 1.5]], dtype=np.float32))
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b", "c", "d"],
                "cohort": ["low", "left", "right", "query"],
                "binding_a": [False, True, False, True],
                "binding_b": [False, False, True, True],
            }
        ),
        view / "rows.parquet",
    )
    context = SimpleNamespace(output_root=tmp_path)
    params = {
        "view_id": "example",
        "cohort_column": "cohort",
        "centroid_groups": {"baseline": ["low"], "a": ["left"], "b": ["right"]},
        "tf_columns": {"a": ["binding_a"], "b": ["binding_b"]},
        "tf_bin_labels": {"a": "A-only", "b": "B-only"},
        "axes": {"a_margin": {"target": "a", "background": "baseline"}},
        "centroid_distances": {"distance_a": "a"},
        "output_filter": {"column": "cohort", "in": ["left", "query"]},
    }
    artifact = tmp_path / "audit"
    artifact.mkdir()
    result = build_tf_axis_orientation_audit_scalar(context, artifact_dir=artifact, params=params)
    rows = pq.read_table(artifact / "table.parquet").to_pylist()
    assert [row["tf_bin"] for row in rows] == ["A-only", "mixed"]
    assert rows[0]["a_margin"] > rows[1]["a_margin"]
    assert rows[0]["distance_a"] == pytest.approx(0, abs=1e-6)
    assert "ethanolness" not in result.columns
    with pytest.raises(ContractViolationError, match="unknown.*centroid"):
        build_tf_axis_orientation_audit_scalar(
            context, artifact_dir=artifact, params={**params, "axes": {"bad": {"target": "missing", "background": "a"}}}
        )
