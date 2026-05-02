"""Native RegulonDB promoter metadata profile coverage for LatentDNA."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.services.validation_service import validate_workspace
from dnadesign.latentdna.src.views.materialize import materialize_view_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


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


def test_live_regulondb_workspace_declares_representation_health_review_path() -> None:
    workspace = _repo_root() / "src" / "dnadesign" / "latentdna" / "workspaces" / "regulondb_native_promoter_panel"
    context = load_workspace_config(workspace)

    notebook = context.config.notebooks["latent_geometry_browser"]
    recipe_steps = {step.id: step for step in context.config.recipes["regulondb_review_recipe"].steps}

    assert notebook.default_deliverable == "representation_health_summary"
    assert notebook.ordered_plots[:5] == [
        "representation_health_summary",
        "native_core60_shift_summary",
        "sigma_factor_structure_summary",
        "sigma_umap_intermediate_embedding_7b_native_source_record_seq_mean",
        "sigma_umap_intermediate_embedding_7b_core60_tss_upstream",
    ]
    assert context.config.deliverables["representation_health_summary"].recipe == "regulondb_review_recipe"
    assert context.config.deliverables["native_core60_shift_summary"].recipe == "regulondb_review_recipe"
    assert context.config.deliverables["sigma_factor_structure_summary"].recipe == "regulondb_review_recipe"
    assert context.config.deliverables["sigma_umap_panel"].recipe == "regulondb_review_recipe"
    assert context.config.plots["representation_health_summary"].kind == "metric_panel_grid"
    assert context.config.plots["native_core60_shift_summary"].kind == "metric_panel_grid"
    assert context.config.plots["sigma_factor_structure_summary"].kind == "metric_panel_grid"
    assert context.config.alignments["intermediate_embedding_7b_native_to_core60"].left_on == [
        "alignment_parent_sequence_id"
    ]
    assert context.config.alignments["output_layer_mean_7b_native_to_core60"].right_on == [
        "alignment_parent_sequence_id"
    ]
    assert context.config.views["output_layer_mean_7b_native_source_record_seq_mean"].coordinate_space_id == (
        "evo2_7b_output_layer_mean"
    )
    assert context.config.views["output_layer_mean_7b_core60_tss_upstream"].coordinate_space_id == (
        "evo2_7b_output_layer_mean"
    )
    assert "materialize_native_source_record_output_layer" in recipe_steps
    assert "materialize_core60_tss_upstream_output_layer" in recipe_steps
    assert "build_representation_health_summary_metrics" in recipe_steps
    assert "build_sigma_factor_structure_summary_metrics" in recipe_steps
    assert "build_native_core60_shift_summary_metrics" in recipe_steps
    assert "render_representation_health_summary" in recipe_steps
    assert "render_native_core60_shift_summary" in recipe_steps
    assert "render_sigma_factor_structure_summary" in recipe_steps
    assert "generate_latent_geometry_browser" in recipe_steps
