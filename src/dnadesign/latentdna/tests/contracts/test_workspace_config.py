"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/contracts/test_workspace_config.py

Workspace contract validation tests for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil

import pytest
from pydantic import ValidationError

from dnadesign.latentdna.src.contracts.errors import (
    ContractViolationError,
    CoordinateSpaceError,
    WorkspaceValidationError,
)
from dnadesign.latentdna.src.workspaces import load_workspace_config
from dnadesign.latentdna.src.workspaces.paths import builtin_templates_dir


def test_load_workspace_config_rejects_cross_space_vector_difference(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  output_root: ./outputs
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


def test_load_workspace_config_rejects_projection_recipe_without_explicit_seed(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  projection_recipe:
    steps:
      - id: fit_projection
        op: projection.fit
        params:
          view: z20_60
          sample: atlas_sample
          run_id: umap_z20_60
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="projection.fit.*explicit seed"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_deliverable_recipe(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
    title: Demo deliverable
    section: Atlas
    question: Does the demo deliverable validate?
    summary: Minimal deliverable fixture for recipe validation.
    recipe: missing_recipe
    requires:
      views: [z20_60]
    outputs:
      projections: [umap_z20_60]
      plots: [atlas_demo_plot]
    docs_refs: []
    acceptance_checks: []
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
    title: Demo deliverable
    section: Atlas
    question: Does the demo deliverable validate?
    summary: Minimal deliverable fixture for config-backed output validation.
    recipe: atlas_recipe
    requires:
      views: [z20_60]
    outputs:
      views: [missing_view]
    docs_refs: []
    acceptance_checks: []
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
  context_shift_demo:
    derive:
      kind: normalize
      view: z20_60
      method: l2
    coordinate_space_id: shared_space
    tags: {operation: normalize}
    role: primary
scalars:
  context_shift_l2_demo:
    derive:
      kind: vector_norm
      view: context_shift_demo
      norm: l2
recipes:
  review_recipe:
    steps:
      - id: materialize_view
        op: view.materialize
        params:
          view: z20_60
deliverables:
  review_bundle:
    title: Demo deliverable
    section: Review
    question: Does the demo deliverable validate?
    summary: Minimal deliverable fixture for recipe-output validation.
    recipe: review_recipe
    requires:
      views: [z20_60]
    outputs:
      views: [context_shift_demo]
      scalars: [context_shift_l2_demo]
    docs_refs: []
    acceptance_checks: []
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="linked recipe does not produce"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_accepts_promoter_reference_margin_benchmark_template(tmp_path) -> None:
    template_dir = builtin_templates_dir() / "promoter_reference_margin_benchmark"
    workspace_dir = tmp_path / "workspace"
    shutil.copytree(template_dir, workspace_dir)

    context = load_workspace_config(workspace_dir)

    assert context.workspace_id == "promoter_reference_margin_workspace"
    assert "dataset_overview" in context.config.deliverables
    assert "representation_health_summary" in context.config.deliverables
    assert "design_structure_summary" in context.config.deliverables
    assert "sigma35_ordinal_audit" in context.config.deliverables
    assert "context_robustness_summary" in context.config.deliverables
    assert "appendix_geometry_review" in context.config.deliverables
    assert "appendix_umap_gallery" in context.config.deliverables


def test_load_workspace_config_accepts_notebook_declarations(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "plot_semantics").mkdir()
    (workspace_dir / "plot_semantics" / "appendix_umap_gallery.yaml").write_text(
        """
plot_id: appendix_umap_gallery
question: What appendix projection is available for the demo notebook?
decision_role: appendix
encoding: Demo projection scatter plot for notebook wiring validation.
scope: Full population.
guardrails:
  - This fixture only validates workspace loading.
caption: Demo appendix plot semantics fixture.
alt_text: Demo appendix plot semantics fixture.
preprocessing_md: Fixture semantics do not declare additional preprocessing.
math_md: Fixture semantics do not declare a mathematical definition.
rationale_md: Fixture semantics exist only to validate workspace loading.
limitations_md: Fixture semantics are not a study-facing scientific contract.
failure_modes_md: Replace fixture semantics before using the plot outside tests.
        """.strip()
        + "\n",
        encoding="utf-8",
    )
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
  latent_geometry_browser:
    kind: workspace
    title: Demo workspace notebook
    default_deliverable: appendix_umap_gallery
plots:
  appendix_umap_gallery:
    kind: projection_scatter
    projection: umap_z20_60
    semantics_ref: plot_semantics/appendix_umap_gallery.yaml
deliverables:
  appendix_umap_gallery:
    title: Demo workspace deliverable
    section: Appendix
    question: Does the browser render cleanly?
    summary: Minimal workspace notebook surface.
    recipe: notebook_recipe
    requires:
      views: [z20_60]
    outputs:
      plots: [appendix_umap_gallery]
      notebooks: [latent_geometry_browser]
    docs_refs: []
    acceptance_checks: []
recipes:
  notebook_recipe:
    steps:
      - id: render_appendix
        op: plot.render
        params:
          plot: appendix_umap_gallery
      - id: generate_notebook
        op: notebook.generate
        depends_on: [render_appendix]
        params:
          notebook: latent_geometry_browser
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    notebook = context.require_notebook("latent_geometry_browser")
    assert notebook.title == "Demo workspace notebook"
    assert notebook.default_deliverable == "appendix_umap_gallery"


def test_load_workspace_config_accepts_candidate_set_declarations(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "plot_semantics").mkdir()
    (workspace_dir / "plot_semantics" / "appendix_umap_gallery.yaml").write_text(
        """
plot_id: appendix_umap_gallery
question: What appendix projection is available for the demo notebook?
decision_role: appendix
encoding: Demo projection scatter plot for notebook wiring validation.
scope: Full population.
guardrails:
  - This fixture only validates workspace loading.
caption: Demo appendix plot semantics fixture.
alt_text: Demo appendix plot semantics fixture.
preprocessing_md: Fixture semantics do not declare additional preprocessing.
math_md: Fixture semantics do not declare a mathematical definition.
rationale_md: Fixture semantics exist only to validate workspace loading.
limitations_md: Fixture semantics are not a study-facing scientific contract.
failure_modes_md: Replace fixture semantics before using the plot outside tests.
        """.strip()
        + "\n",
        encoding="utf-8",
    )
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
    tags: {model: demo, family: intermediate_embedding, scope: anchor_60bp}
    role: primary
candidate_sets:
  demo_x:
    label: Demo X
    include_tags: {family: intermediate_embedding}
notebooks:
  latent_geometry_browser:
    kind: workspace
    title: Demo workspace notebook
    default_deliverable: appendix_umap_gallery
    candidate_sets: [demo_x]
    default_candidate_set: demo_x
plots:
  appendix_umap_gallery:
    kind: projection_scatter
    projection: umap_z20_60
    semantics_ref: plot_semantics/appendix_umap_gallery.yaml
deliverables:
  appendix_umap_gallery:
    title: Demo workspace deliverable
    section: Appendix
    question: Does the browser render cleanly?
    summary: Minimal workspace notebook surface.
    recipe: notebook_recipe
    requires:
      views: [z20_60]
    outputs:
      plots: [appendix_umap_gallery]
      notebooks: [latent_geometry_browser]
    docs_refs: []
    acceptance_checks: []
recipes:
  notebook_recipe:
    steps:
      - id: render_appendix
        op: plot.render
        params:
          plot: appendix_umap_gallery
      - id: generate_notebook
        op: notebook.generate
        depends_on: [render_appendix]
        params:
          notebook: latent_geometry_browser
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    assert context.config.candidate_sets["demo_x"].label == "Demo X"
    assert context.require_notebook("latent_geometry_browser").default_candidate_set == "demo_x"


def test_load_workspace_config_rejects_legacy_deliverable_shape(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
    description: Legacy deliverable.
    recipe: atlas_recipe
    requires:
      views: [z20_60]
    outputs:
      views: [z20_60]
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_legacy_study_binding_shape(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
study_binding:
  study_dir: src/dnadesign/studies/studies/stress_ethanol_cipro_growth
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_legacy_study_binding_docs_root(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
study_binding:
  study_id: stress_ethanol_cipro_growth
  docs_root: src/dnadesign/studies/studies/stress_ethanol_cipro_growth
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_retired_output_logit_family(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  old_output_surface:
    source: anchor60
    vector:
      kind: column
      name: value
    coordinate_space_id: evo2_7b_output_layer_mean
    tags: {model: 7b, family: pooled_logits}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="retired representation family"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_retired_output_logit_coordinate_space(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  output_layer_mean_7b_anchor_60bp:
    source: anchor60
    vector:
      kind: column
      name: value
    coordinate_space_id: evo2_7b_pooled_logits
    tags: {model: 7b, family: output_layer_mean}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="retired representation term"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_retired_output_logit_view_id(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  pooled_logits_7b_anchor_60bp:
    source: anchor60
    vector:
      kind: column
      name: value
    coordinate_space_id: evo2_7b_output_layer_mean
    tags: {model: 7b, family: output_layer_mean}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="retired representation term"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_retired_family_candidate_selector(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
candidate_sets:
  old_outputs:
    label: Old outputs
    include_tags: {family: pooled_logits}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="retired representation family"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_docs_ref_path_traversal(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    record_root = tmp_path / "docs" / "studies" / "demo_study"
    record_root.mkdir(parents=True)
    for file_name in ("campaign.yaml", "datasets.yaml", "ops.study.yaml"):
        (record_root / file_name).write_text("version: 1\n", encoding="utf-8")
    (record_root / "status.md").write_text("## Demo\n", encoding="utf-8")
    deliverable_docs_root = tmp_path / "study_docs"
    deliverable_docs_root.mkdir()
    (deliverable_docs_root / "study.yaml").write_text("study_id: demo_study\n", encoding="utf-8")
    (deliverable_docs_root / "deliverables").mkdir()
    (deliverable_docs_root / "deliverables" / "review.md").write_text("# Review\n", encoding="utf-8")
    (tmp_path / "outside.md").write_text("# Outside\n", encoding="utf-8")
    (workspace_dir / "config.yaml").write_text(
        f"""
schema_version: latentdna.workspace.v1
workspace:
  id: demo
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
    tags: {{model: demo}}
    role: primary
recipes:
  review_recipe:
    steps:
      - id: materialize_view
        op: view.materialize
        params:
          view: z20_60
deliverables:
  review_bundle:
    title: Demo deliverable
    section: Review
    question: Does the demo deliverable validate?
    summary: Minimal deliverable fixture for docs-ref validation.
    recipe: review_recipe
    requires:
      views: [z20_60]
    outputs:
      views: [z20_60]
    docs_refs: [study:demo_study/../outside]
    acceptance_checks: []
study_binding:
  study_id: demo_study
  record_root: {record_root.as_posix()}
  deliverable_docs_root: {deliverable_docs_root.as_posix()}
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="stay under the study docs root"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_noncanonical_output_root_location(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: demo
  output_root: ./outputs/nested
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="workspace output_root must resolve to"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_accepts_plot_registry(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "plot_semantics").mkdir()
    (workspace_dir / "plot_semantics" / "atlas_scatter.yaml").write_text(
        """
plot_id: atlas_scatter
question: Does the demo plot registry preserve scatter settings?
decision_role: debug
encoding: Demo scatter plot used for plot-registry validation.
scope: Full population.
guardrails:
  - Fixture semantics for config validation only.
caption: Demo plot registry semantics.
alt_text: Demo plot registry semantics.
preprocessing_md: Fixture semantics do not declare additional preprocessing.
math_md: Fixture semantics do not declare a mathematical definition.
rationale_md: Fixture semantics exist only to validate plot-registry loading.
limitations_md: Fixture semantics are not a study-facing scientific contract.
failure_modes_md: Replace fixture semantics before using the plot outside tests.
        """.strip()
        + "\n",
        encoding="utf-8",
    )
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
    semantics_ref: plot_semantics/atlas_scatter.yaml
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


def test_load_workspace_config_can_skip_plot_semantics_sidecar_validation(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
plots:
  atlas_scatter:
    kind: projection_scatter
    projection: umap_z20_60
    semantics_ref: plot_semantics/missing.yaml
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)

    assert context.require_plot("atlas_scatter").semantics_ref == "plot_semantics/missing.yaml"


def test_load_workspace_config_can_require_plot_semantics_sidecars_explicitly(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
plots:
  atlas_scatter:
    kind: projection_scatter
    projection: umap_z20_60
    semantics_ref: plot_semantics/missing.yaml
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractViolationError, match="plot semantics sidecar does not exist"):
        load_workspace_config(workspace_dir, validate_plot_semantics=True)


def test_load_workspace_config_rejects_projection_grid_with_misaligned_panel_titles(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  plot_formats: [svg, png]
  neighbor_backend: auto
sources: {}
metadata:
  include: []
plots:
  atlas_grid:
    kind: projection_grid
    projections: [umap_a, umap_b]
    semantics_ref: plot_semantics/atlas_grid.yaml
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
  output_root: ./outputs
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
    semantics_ref: plot_semantics/atlas_scatter.yaml
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
  output_root: ./outputs
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
    semantics_ref: plot_semantics/bad_distribution.yaml
    scalar: context_shift_l2_demo
    distance: primary_landmark_distances
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="distribution plots require exactly one artifact input"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_notebook_default_deliverable(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
    kind: workspace
    title: Demo workspace notebook
    default_deliverable: missing_bundle
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="references unknown default deliverable"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_misaligned_notebook_candidate_panel_titles(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  latent_geometry_browser:
    kind: workspace
    title: Demo workspace notebook
    default_deliverable: appendix_umap_gallery
    candidate_grid_views: [z20_60]
    candidate_grid_panel_titles: [first, second]
deliverables:
  appendix_umap_gallery:
    title: Demo workspace deliverable
    section: Appendix
    question: Does the browser render cleanly?
    summary: Minimal workspace notebook surface.
    recipe: notebook_recipe
    requires:
      views: [z20_60]
    outputs:
      notebooks: [latent_geometry_browser]
    docs_refs: []
    acceptance_checks: []
recipes:
  notebook_recipe:
    steps:
      - id: generate_notebook
        op: notebook.generate
        params:
          notebook: latent_geometry_browser
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="candidate_grid_panel_titles must match candidate_grid_views length"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_notebook_candidate_set(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  latent_geometry_browser:
    kind: workspace
    title: Demo workspace notebook
    default_deliverable: appendix_umap_gallery
    candidate_sets: [missing_x]
deliverables:
  appendix_umap_gallery:
    title: Demo workspace deliverable
    section: Appendix
    question: Does the browser render cleanly?
    summary: Minimal workspace notebook surface.
    recipe: notebook_recipe
    requires:
      views: [z20_60]
    outputs:
      notebooks: [latent_geometry_browser]
    docs_refs: []
    acceptance_checks: []
recipes:
  notebook_recipe:
    steps:
      - id: generate_notebook
        op: notebook.generate
        params:
          notebook: latent_geometry_browser
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="unknown candidate_set"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_notebook_default_reference_set(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
  latent_geometry_browser:
    kind: workspace
    title: Demo workspace notebook
    default_deliverable: appendix_umap_gallery
    default_reference_set: missing_reference_set
deliverables:
  appendix_umap_gallery:
    title: Demo workspace deliverable
    section: Appendix
    question: Does the browser render cleanly?
    summary: Minimal workspace notebook surface.
    recipe: notebook_recipe
    requires:
      views: [z20_60]
    outputs:
      notebooks: [latent_geometry_browser]
    docs_refs: []
    acceptance_checks: []
recipes:
  notebook_recipe:
    steps:
      - id: generate_notebook
        op: notebook.generate
        params:
          notebook: latent_geometry_browser
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceValidationError, match="unknown default_reference_set"):
        load_workspace_config(workspace_dir)


def test_load_workspace_config_rejects_unknown_export_block_alignment(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
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
