"""Native RegulonDB promoter metadata profile coverage for LatentDNA."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.services.validation_service import validate_workspace
from dnadesign.latentdna.src.views.materialize import materialize_view_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def test_native_regulondb_promoter_cohorts_materialize_without_densegen_or_sig35(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "native_regulondb_workspace"
    source_path = workspace_dir / "inputs" / "native_promoters.parquet"
    source_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["usr_a", "usr_b"],
                "subject_id": ["usr_a", "usr_b"],
                "embedding": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32())),
                "regulondb__sigma_factor_set": pa.array(
                    [["sigma70"], ["sigma38", "sigma70"]],
                    type=pa.list_(pa.string()),
                ),
                "regulondb__regulator_composition": ["activator", "mixed"],
                "regulondb__box_pattern": ["-35/-10", "-10_only"],
                "regulondb__confidence_level_set": pa.array(
                    [["strong"], ["weak"]],
                    type=pa.list_(pa.string()),
                ),
                "regulondb__metadata_completeness_class": ["complete", "partial"],
            }
        ),
        source_path,
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: native_regulondb_profile_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
sources:
  native_full:
    kind: parquet
    path: inputs/native_promoters.parquet
    record_key: id
    subject_key: subject_id
views:
  native_full_7b:
    source: native_full
    vector:
      kind: column
      name: embedding
    coordinate_space_id: evo2_7b_native_full
cohorts:
  regulondb__sigma_factor_set:
    kind: promoter_metadata
    source: native_full
    derive: regulondb__sigma_factor_set
  regulondb__regulator_composition:
    kind: promoter_metadata
    source: native_full
    derive: regulondb__regulator_composition
  regulondb__box_pattern:
    kind: promoter_metadata
    source: native_full
    derive: regulondb__box_pattern
  regulondb__confidence_level_set:
    kind: promoter_metadata
    source: native_full
    derive: regulondb__confidence_level_set
  regulondb__metadata_completeness_class:
    kind: promoter_metadata
    source: native_full
    derive: regulondb__metadata_completeness_class
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, row_count, dims, _, row_columns, processing_columns = materialize_view_artifact(
        context,
        view_id="native_full_7b",
    )

    assert row_count == 2
    assert dims == 2
    assert "densegen__plan" not in processing_columns
    assert "densegen__used_tfbs_detail" not in processing_columns
    assert "sig35_variant" not in row_columns
    rows = read_table(artifact_dir / "rows.parquet")
    assert "sig35_variant" not in rows.column_names
    assert "regulondb__sigma_factor_set" in rows.column_names
    assert rows.column("regulondb__regulator_composition").to_pylist() == ["activator", "mixed"]
    validation = validate_workspace(workspace_dir, deep=True)
    assert validation["status"] == "ok"
    assert {
        detail["derive"]
        for detail in validation["cohort_details"]
        if detail["source"] == "native_full" and detail["kind"] == "promoter_metadata"
    } == {
        "regulondb__sigma_factor_set",
        "regulondb__regulator_composition",
        "regulondb__box_pattern",
        "regulondb__confidence_level_set",
        "regulondb__metadata_completeness_class",
    }
