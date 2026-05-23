from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.latentdna.src.contracts.errors import WorkspaceValidationError
from dnadesign.latentdna.src.services.validation_service import validate_workspace
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces


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


def test_deep_validate_workspace_fails_for_materialized_view_row_count_drift(tmp_path) -> None:
    workspace_dir, view_dir = _write_validation_workspace(tmp_path, role="primary")
    pq.write_table(pa.table({"id": ["row_01"], "demo_metric": [0.1]}), view_dir / "rows.parquet")
    np.save(view_dir / "matrix.npy", np.asarray([[0.0, 1.0]], dtype=np.float32))

    try:
        validate_workspace(workspace_dir, deep=True)
    except WorkspaceValidationError as exc:
        assert "materialized view row count no longer matches source schema" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected deep validation to fail for materialized row-count drift")


def test_deep_validate_workspace_uses_each_views_own_source_row_count(tmp_path) -> None:
    workspace_dir, view_dir = _write_validation_workspace(tmp_path, role="primary")
    pq.write_table(
        pa.table(
            {
                "id": ["unused_01"],
                "embedding": pa.array([[1.0, 1.0]], type=pa.list_(pa.float32())),
                "score": [1.0],
            }
        ),
        workspace_dir / "inputs" / "z_unused.parquet",
    )
    config_path = workspace_dir / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace(
            "metadata:\n",
            """
  z_unused:
    kind: parquet
    path: ./inputs/z_unused.parquet
    record_key: id
    subject_key: id
metadata:
""",
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"id": ["row_01", "row_02"], "demo_metric": [0.1, 0.2]}),
        view_dir / "rows.parquet",
    )

    payload = validate_workspace(workspace_dir, deep=True)

    detail = next(item for item in payload["view_details"] if item["view_id"] == "z20_60")
    assert detail["materialized_row_count"] == 2
    assert detail["materialized_matrix_shape"] == [2, 2]


def test_deep_validate_workspace_fails_declared_fixed_length_mismatch(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["row_01", "row_02"],
                "length": [60, 71],
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "mixed.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: semantic_length_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  declared_core60:
    kind: parquet
    path: ./inputs/mixed.parquet
    record_key: id
    subject_key: id
    sequence_scope: analysis_window
    emitted_length_bp: 60
views:
  mixed_view:
    source: declared_core60
    vector: {kind: column, name: embedding}
    coordinate_space_id: demo_space
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        validate_workspace(workspace_dir, deep=True)
    except WorkspaceValidationError as exc:
        assert "declares emitted_length_bp=60" in str(exc)
        assert "60:1" in str(exc)
        assert "71:1" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected deep validation to fail for mixed lengths under fixed contract")


def test_deep_validate_workspace_allows_mixed_length_when_labeled_explicitly(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["row_01", "row_02"],
                "length": [60, 71],
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "mixed.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: semantic_mixed_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  merged_anchor_insert:
    kind: parquet
    path: ./inputs/mixed.parquet
    record_key: id
    subject_key: id
    sequence_scope: source_insert
    source_interval_length_bp: mixed
    pooling_span_bp: full_sequence
views:
  mixed_view:
    source: merged_anchor_insert
    vector: {kind: column, name: embedding}
    coordinate_space_id: demo_space
    tags: {scope: merged_anchor_insert_seq_mean}
candidate_sets:
  mixed_candidates:
    label: Mixed candidates
    views: [mixed_view]
    panel_titles: {mixed_view: Anchor-source insert mean}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    payload = validate_workspace(workspace_dir, deep=True)

    assert payload["status"] == "ok"
    detail = next(item for item in payload["sequence_semantic_details"] if item["source_id"] == "merged_anchor_insert")
    assert detail["length_counts"] == {60: 1, 71: 1}


def test_deep_validate_workspace_fails_mixed_length_fixed_60_panel_label(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["row_01", "row_02"],
                "length": [60, 71],
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "mixed.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: semantic_label_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  merged_anchor_insert:
    kind: parquet
    path: ./inputs/mixed.parquet
    record_key: id
    subject_key: id
    sequence_scope: source_insert
views:
  mixed_view:
    source: merged_anchor_insert
    vector: {kind: column, name: embedding}
    coordinate_space_id: demo_space
    tags: {scope: merged_anchor_insert_seq_mean}
candidate_sets:
  mixed_candidates:
    label: Mixed candidates
    views: [mixed_view]
    panel_titles: {mixed_view: 60 bp anchor}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        validate_workspace(workspace_dir, deep=True)
    except WorkspaceValidationError as exc:
        assert "labels mixed_view as '60 bp anchor'" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected deep validation to fail for fixed-60bp panel label on mixed source")


def test_deep_validate_workspace_marks_old_materialized_view_source_contract_stale(tmp_path) -> None:
    workspace_dir, view_dir = _write_validation_workspace(tmp_path, role="primary")
    (view_dir / "manifest.json").write_text(
        """
{
  "params": {
    "source": "old_anchor60",
    "vector_column": "old_embedding"
  }
}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    payload = validate_workspace(workspace_dir, deep=True)

    detail = next(item for item in payload["view_details"] if item["view_id"] == "z20_60")
    assert detail["materialized"] is True
    assert detail["materialized_contract_status"] == "stale_source_contract"
    assert detail["materialized_source"] == "stale"
    assert detail["missing_materialized_row_columns"] == ["demo_metric"]


def test_deep_validate_workspace_allows_missing_planned_source(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: validation_planned_source_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  future_core60_features:
    kind: parquet
    path: ./inputs/not_yet_materialized.parquet
    record_key: id
    subject_key: id
    role: planned
views:
  future_core60_geometry:
    source: future_core60_features
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
    tags: {model: demo}
    role: planned
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    payload = validate_workspace(workspace_dir, deep=True)

    assert payload["status"] == "ok"
    source_detail = next(item for item in payload["source_details"] if item["source_id"] == "future_core60_features")
    assert source_detail["validation_status"] == "skipped_planned"
    view_detail = next(item for item in payload["view_details"] if item["view_id"] == "future_core60_geometry")
    assert view_detail["validation_status"] == "skipped_planned"


def test_deep_validate_workspace_allows_empty_planned_infer_source_with_sequence_semantics(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = workspace_dir / "usr"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "planned_infer_sidecar")
    dataset.init(source="test", notes="planned infer sidecar validation test")
    dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test")
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: validation_planned_infer_source_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  future_infer_features:
    kind: infer_feature_sidecar
    root: ./usr
    dataset: planned_infer_sidecar
    record_key: id
    subject_key: id
    role: planned
    sequence_scope: source_record
    emitted_length_bp: 60
    where:
      model_name: evo2_7b
      representation_kind: intermediate_embedding
      pooling_operation: seq_mean
      view_name: future_view
views:
  future_geometry:
    source: future_infer_features
    vector:
      kind: column
      name: value
    coordinate_space_id: demo_space
    tags: {model: demo}
    role: planned
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    payload = validate_workspace(workspace_dir, deep=True)

    assert payload["status"] == "ok"
    semantic_detail = next(
        item for item in payload["sequence_semantic_details"] if item["source_id"] == "future_infer_features"
    )
    assert semantic_detail["materialization_status"] == "planned_empty"
    assert semantic_detail["row_count"] == 0
    assert semantic_detail["length_counts"] == {}


def test_deep_validate_workspace_fails_for_malformed_planned_source(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(pa.table({"wrong_id": ["row_01"]}), inputs_dir / "malformed.parquet")
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: validation_malformed_planned_source_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  malformed_planned:
    kind: parquet
    path: ./inputs/malformed.parquet
    record_key: id
    subject_key: id
    role: planned
views:
  malformed_geometry:
    source: malformed_planned
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
    tags: {model: demo}
    role: planned
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        validate_workspace(workspace_dir, deep=True)
    except WorkspaceValidationError as exc:
        assert "source malformed_planned is missing required columns" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected malformed planned source to fail")


def test_deep_validate_accepts_sig35_cohort_from_sequence_annotation(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["row_01"],
                "usr_label__primary": ["J23104"],
                "sequence": ["TTGACATATGCTAGCTAGCTAGCTAGCTAGCTAGC"],
                "seq_annot__features": pa.array(
                    [
                        [
                            {
                                "label": "-35",
                                "role_hint": "sigma70_minus35",
                                "qualifiers": [{"key": "note", "value": "feature_sequence=TTGACA"}],
                            }
                        ]
                    ]
                ),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "anchor60.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: validation_sig35_annotation_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: ./inputs/anchor60.parquet
    record_key: id
    subject_key: id
metadata:
  include: [sig35_variant]
  derivations:
    sig35_variant:
      kind: annotation
      source: row
      handler: dnadesign.latentdna.src.views.promoter_metadata:derive_promoter_metadata_value
      derive: sig35_variant
      required_columns: [usr_label__primary]
      any_required_column_groups:
        - [seq_annot__features]
      missing_policy: error
      value_type: string
views:
  z7_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
    tags: {model: demo}
    role: planned
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    payload = validate_workspace(workspace_dir, deep=True)

    assert payload["status"] == "ok"
