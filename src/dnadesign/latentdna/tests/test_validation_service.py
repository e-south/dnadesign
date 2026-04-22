from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.latentdna.src.contracts.errors import WorkspaceValidationError
from dnadesign.latentdna.src.services.validation_service import validate_workspace


def _write_validation_workspace(tmp_path, *, role: str) -> tuple[object, object]:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    source_path = inputs_dir / "anchor60.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["row_01", "row_02"],
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
                "score": [0.1, 0.2],
            }
        ),
        source_path,
    )
    (workspace_dir / "config.yaml").write_text(
        f"""
schema_version: latentdna.workspace.v1
workspace:
  id: validation_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: ./inputs/anchor60.parquet
    record_key: id
    subject_key: id
metadata:
  include: [demo_metric]
  derivations:
    demo_metric:
      kind: copy
      source: score
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
    tags: {{model: demo}}
    role: {role}
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    view_dir = workspace_dir / "outputs" / "views" / "z20_60"
    view_dir.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["row_01", "row_02"]}), view_dir / "rows.parquet")
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    return workspace_dir, view_dir


def test_deep_validate_workspace_skips_hidden_materialized_view_row_contract_drift(tmp_path) -> None:
    workspace_dir, _ = _write_validation_workspace(tmp_path, role="hidden")

    payload = validate_workspace(workspace_dir, deep=True)

    detail = next(item for item in payload["view_details"] if item["view_id"] == "z20_60")
    assert detail["materialized"] is True
    assert detail["materialized_contract_status"] == "skipped_hidden"
    assert detail["missing_materialized_row_columns"] == ["demo_metric"]


def test_deep_validate_workspace_still_fails_for_primary_materialized_view_row_contract_drift(tmp_path) -> None:
    workspace_dir, _ = _write_validation_workspace(tmp_path, role="primary")

    try:
        validate_workspace(workspace_dir, deep=True)
    except WorkspaceValidationError as exc:
        assert "materialized view rows are missing configured metadata columns" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected deep validation to fail for surfaced primary view row drift")
