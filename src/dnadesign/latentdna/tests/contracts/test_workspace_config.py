"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/contracts/test_workspace_config.py

Workspace contract validation tests for latentdna.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil

import pytest
from pydantic import ValidationError

from dnadesign.latentdna import load_workspace_config
from dnadesign.latentdna.api import CoordinateSpaceError, WorkspaceValidationError
from dnadesign.latentdna.src.workspaces.loader import builtin_templates_dir


def test_load_workspace_config_rejects_cross_space_vector_difference(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
views:
  left_view:
    source: missing_source
    vector:
      kind: column
      name: left_embedding
    coordinate_space_id: left_space
    tags: {model: left}
    role: primary
  right_view:
    source: missing_source
    vector:
      kind: column
      name: right_embedding
    coordinate_space_id: right_space
    tags: {model: right}
    role: primary
  bad_delta:
    derive:
      kind: vector_difference
      left: left_view
      right: right_view
      alignment: anchor_ctx
    coordinate_space_id: left_space
    tags: {operation: difference}
    role: primary
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CoordinateSpaceError, match="coordinate space"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_alignment_reference(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  left_view:
    source: anchor60
    vector:
      kind: column
      name: left_embedding
    coordinate_space_id: shared_space
    tags: {model: left}
    role: primary
  right_view:
    source: anchor60
    vector:
      kind: column
      name: right_embedding
    coordinate_space_id: shared_space
    tags: {model: right}
    role: primary
  good_delta:
    derive:
      kind: vector_difference
      left: left_view
      right: right_view
      alignment: anchor_ctx
    coordinate_space_id: shared_space
    tags: {operation: difference}
    role: primary
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="unknown alignment"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_accepts_matrix_bundle_and_extended_derivations(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  bundle_source:
    kind: matrix_bundle
    path: inputs/bundle
    record_key: id
    subject_key: subject_id
  context_source:
    kind: parquet
    path: inputs/context.parquet
    record_key: id
    subject_key: subject_id
    context_key: context_id
metadata:
  include: []
views:
  bundle_view:
    source: bundle_source
    vector:
      kind: bundle_matrix
    coordinate_space_id: bundle_space
    tags: {model: demo}
    role: primary
  normalized_bundle:
    derive:
      kind: normalize
      view: bundle_view
      method: l2
    coordinate_space_id: bundle_space
    tags: {operation: normalize}
    role: primary
  context_view:
    source: context_source
    vector:
      kind: column
      name: embedding
    coordinate_space_id: context_space
    tags: {model: demo}
    role: primary
  context_by_subject:
    derive:
      kind: aggregate_by_key
      view: context_view
      key: subject_key
      aggregation: mean
    coordinate_space_id: context_space
    tags: {operation: aggregate}
    role: primary
  reduced_bundle:
    derive:
      kind: apply_reducer
      view: bundle_view
      reducer: bundle_pca
    coordinate_space_id: bundle_space_pca
    tags: {operation: apply_reducer}
    role: primary
  reduced_bundle_norm:
    derive:
      kind: normalize
      view: reduced_bundle
      method: l2
    coordinate_space_id: bundle_space_pca
    tags: {operation: normalize}
    role: primary
  concatenated_bundle:
    derive:
      kind: concatenate
      inputs: [reduced_bundle, reduced_bundle_norm]
    coordinate_space_id: concatenated_space
    tags: {operation: concatenate}
    role: primary
scalars:
  selected_scalars:
    derive:
      kind: select_columns
      source: raw_scalars
      columns: [score]
  renamed_scalars:
    derive:
      kind: rename_columns
      source: selected_scalars
      renames:
        score: normalized_score
  joined_scalars:
    derive:
      kind: join_tables
      sources: [selected_scalars, renamed_scalars]
      "on": [id]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    assert context.config.views["bundle_view"].vector.kind == "bundle_matrix"
    assert context.config.views["normalized_bundle"].derive.kind == "normalize"
    assert context.config.views["context_by_subject"].derive.kind == "aggregate_by_key"
    assert context.config.views["reduced_bundle"].derive.kind == "apply_reducer"
    assert context.config.views["reduced_bundle_norm"].derive.kind == "normalize"
    assert context.config.views["concatenated_bundle"].derive.kind == "concatenate"
    assert context.config.scalars["selected_scalars"].derive.kind == "select_columns"
    assert context.config.scalars["renamed_scalars"].derive.kind == "rename_columns"
    assert context.config.scalars["joined_scalars"].derive.kind == "join_tables"


def test_load_workspace_config_rejects_unknown_view_key(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
    typo_field: true
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_cross_space_concatenate_of_raw_views(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z7_60:
    source: anchor60
    vector:
      kind: column
      name: embedding_7b
    coordinate_space_id: evo2_7b_space
    tags: {model: 7b}
    role: primary
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding_20b
    coordinate_space_id: evo2_20b_space
    tags: {model: 20b}
    role: primary
  bad_concat:
    derive:
      kind: concatenate
      inputs: [z7_60, z20_60]
    coordinate_space_id: mixed_space
    tags: {operation: concatenate}
    role: primary
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CoordinateSpaceError, match="concatenate"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_cohort_source(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
cohorts:
  plan:
    kind: column
    source: missing_source
    column: densegen__plan
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="cohort plan references unknown source"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_cyclic_recipe_graph(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
recipes:
  bad_recipe:
    steps:
      - id: first_step
        op: view.materialize
        depends_on: [second_step]
        params:
          view: z20_60
      - id: second_step
        op: sample.build
        depends_on: [first_step]
        params:
          sample_id: atlas_sample
          view: z20_60
          strategy: all
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="cycle"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_deliverable_recipe(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
deliverables:
  atlas_demo:
    kind: projection_panel
    description: Demo deliverable.
    recipe: missing_recipe
    requires:
      views: [z20_60]
    outputs:
      projections: [umap_z20_60]
      plots: [atlas_demo_plot]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="unknown recipe"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_config_backed_deliverable_output(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
recipes:
  atlas_recipe:
    steps:
      - id: materialize_view
        op: view.materialize
        params:
          view: z20_60
deliverables:
  atlas_demo:
    kind: projection_panel
    description: Demo deliverable.
    recipe: atlas_recipe
    requires:
      views: [z20_60]
    outputs:
      views: [missing_view]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="unknown view"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_declared_deliverable_output_missing_from_recipe(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
  delta20:
    derive:
      kind: normalize
      view: z20_60
      method: l2
    coordinate_space_id: shared_space
    tags: {operation: normalize}
    role: primary
scalars:
  delta20_norm:
    derive:
      kind: vector_norm
      view: delta20
      norm: l2
recipes:
  atlas_recipe:
    steps:
      - id: materialize_view
        op: view.materialize
        params:
          view: z20_60
deliverables:
  atlas_demo:
    kind: projection_panel
    description: Demo deliverable.
    recipe: atlas_recipe
    requires:
      views: [z20_60]
    outputs:
      views: [delta20]
      scalars: [delta20_norm]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="linked recipe does not produce"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_accepts_landmark_atlas_committee_template(tmp_path) -> None:
    template_dir = builtin_templates_dir() / "landmark_atlas_committee"
    workspace_dir = tmp_path / "workspace"
    shutil.copytree(template_dir, workspace_dir)

    context = load_workspace_config(workspace_dir)

    assert context.workspace_id == "stress_ethanol_cipro_latent_atlas"
    assert "atlas_2x2_intermediate" in context.config.deliverables
    assert "control_neighborhood_enrichment" in context.config.deliverables
    assert "context_shift_primary" in context.config.deliverables
    assert "agreement_7b_vs_20b" in context.config.deliverables
    assert "x2_primary_20b" in context.config.deliverables


def test_load_workspace_config_accepts_notebook_declarations(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
notebooks:
  atlas_review:
    kind: artifact_review
    title: Demo artifact review
    artifacts:
      - kind: view
        id: z20_60
      - kind: projection
        id: umap_z20_60
        alias: atlas_projection
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    notebook = context.require_notebook("atlas_review")
    assert notebook.title == "Demo artifact review"
    assert [artifact.id for artifact in notebook.artifacts] == ["z20_60", "umap_z20_60"]


def test_load_workspace_config_accepts_plot_registry(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
plots:
  atlas_scatter:
    kind: projection_scatter
    projection: umap_z20_60
    color_column: usr_label__primary
    label_column: usr_label__primary
    label_values: [spyP, sulAp]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    plot = context.require_plot("atlas_scatter")
    assert plot.kind == "projection_scatter"
    assert plot.projection == "umap_z20_60"
    assert plot.label_column == "usr_label__primary"
    assert plot.label_values == ["spyP", "sulAp"]


def test_load_workspace_config_rejects_projection_grid_with_misaligned_panel_titles(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
plots:
  atlas_grid:
    kind: projection_grid
    projections: [umap_a, umap_b]
    panel_titles: [left only]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="projection_grid panel_titles must match projections length"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_projection_scatter_label_values_without_column(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
plots:
  atlas_scatter:
    kind: projection_scatter
    projection: umap_z20_60
    label_values: [spyP]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="projection_scatter label_values require label_column"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_distribution_plot_with_multiple_inputs(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
plots:
  bad_distribution:
    kind: distribution
    scalar: delta20_norm
    distance: primary_landmark_distances
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="distribution plots require exactly one artifact input"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_notebook_artifact_kind(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
notebooks:
  atlas_review:
    kind: artifact_review
    title: Demo artifact review
    artifacts:
      - kind: made_up_kind
        id: z20_60
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="unsupported notebook artifact kind"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_export_block_alignment(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/latentdna
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources:
  anchor60:
    kind: parquet
    path: inputs/anchor60.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  z20_60:
    source: anchor60
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
    tags: {model: demo}
    role: primary
exports:
  x_demo:
    row_basis: anchor_ctx
    blocks:
      - kind: table_columns
        block_id: distances
        source: demo_distances
        columns: [d_spy_p]
        alignment: missing_alignment
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="unknown alignment"):
        load_workspace_config(workspace_dir)
