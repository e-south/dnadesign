"""
Workspace config validation helpers for latentdna.
"""

from __future__ import annotations

from typing import Any

from ..contracts.deliverable import ARTIFACT_REFERENCE_CATEGORIES, SUPPORTED_DELIVERABLE_REFERENCE_CATEGORIES
from ..contracts.errors import CoordinateSpaceError, WorkspaceValidationError
from ..contracts.ids import validate_identifier
from ..contracts.notebook import NotebookConfig, WorkspaceNotebookConfig
from ..contracts.recipe import SUPPORTED_RECIPE_OPS, expected_step_artifacts, topological_step_order
from ..contracts.representations import validate_representation_identity
from ..contracts.workspace import (
    AcceptanceCheckConfig,
    DeliverableConfig,
    DerivedViewConfig,
    MatrixBundleSourceConfig,
    RecipeConfig,
    SourceBackedViewConfig,
    WorkspaceConfig,
)
from ..studies.docs_refs import resolve_docs_ref_path
from ..studies.study_binding import (
    REQUIRED_STUDY_DELIVERABLE_DOC_FILES,
    REQUIRED_STUDY_RECORD_FILES,
    missing_required_files,
)
from .paths import resolve_repo_path


def _view_declares_reduced_space(config: WorkspaceConfig, view_id: str, *, _seen: set[str] | None = None) -> bool:
    if _seen is None:
        _seen = set()
    if view_id in _seen:
        return False
    _seen.add(view_id)

    view = config.views[view_id]
    if not isinstance(view, DerivedViewConfig):
        return False
    if view.derive.kind == "apply_reducer":
        return True
    if view.derive.kind in {"normalize", "aggregate_by_key"}:
        return _view_declares_reduced_space(config, view.derive.view, _seen=_seen)
    if view.derive.kind == "vector_difference":
        return _view_declares_reduced_space(
            config,
            view.derive.left,
            _seen=set(_seen),
        ) and _view_declares_reduced_space(
            config,
            view.derive.right,
            _seen=set(_seen),
        )
    if view.derive.kind == "concatenate":
        return all(
            _view_declares_reduced_space(config, input_view, _seen=set(_seen)) for input_view in view.derive.inputs
        )
    return False


def validate_workspace_config(config: WorkspaceConfig) -> None:
    validate_identifier(config.workspace.id, label="workspace.id")
    for source_id in config.sources:
        validate_identifier(source_id, label="source id")
    for view_id in config.views:
        validate_identifier(view_id, label="view id")
        _validate_representation_identity(view_id, owner="view id")
    for alignment_id in config.alignments:
        validate_identifier(alignment_id, label="alignment id")
    for scalar_id in config.scalars:
        validate_identifier(scalar_id, label="scalar id")
    for landmark_id in config.landmarks:
        validate_identifier(landmark_id, label="landmark id")
    for plot_id in config.plots:
        validate_identifier(plot_id, label="plot id")
    for cohort_id in config.cohorts:
        validate_identifier(cohort_id, label="cohort id")
    for export_id in config.exports:
        validate_identifier(export_id, label="export id")
    for notebook_id in config.notebooks:
        validate_identifier(notebook_id, label="notebook id")
    for recipe_id in config.recipes:
        validate_identifier(recipe_id, label="recipe id")
    for deliverable_id in config.deliverables:
        validate_identifier(deliverable_id, label="deliverable id")

    for column_name, derivation in config.metadata.derivations.items():
        validate_identifier(column_name, label="metadata derivation id")
        if derivation.kind == "lookup" and derivation.source not in config.sources:
            raise WorkspaceValidationError(
                f"metadata derivation {column_name!r} lookup references unknown source {derivation.source!r}"
            )

    for alignment_id, alignment in config.alignments.items():
        if alignment.left not in config.views and alignment.left not in config.sources:
            raise WorkspaceValidationError(f"alignment {alignment_id} references unknown left input {alignment.left!r}")
        if alignment.right not in config.views and alignment.right not in config.sources:
            raise WorkspaceValidationError(
                f"alignment {alignment_id} references unknown right input {alignment.right!r}"
            )
        if alignment.on is not None and isinstance(alignment.on, list) and not alignment.on:
            raise WorkspaceValidationError(f"alignment {alignment_id} must declare at least one key column in 'on'")
        if alignment.left_on is not None:
            if not alignment.left_on or not alignment.right_on:
                raise WorkspaceValidationError(
                    f"alignment {alignment_id} must declare non-empty left_on/right_on columns together"
                )
            if len(alignment.left_on) != len(alignment.right_on):
                raise WorkspaceValidationError(
                    f"alignment {alignment_id} left_on/right_on must declare the same number of columns"
                )

    for view_id, view in config.views.items():
        if isinstance(view, DerivedViewConfig):
            derive = view.derive
            if derive.kind == "vector_difference":
                if derive.left not in config.views or derive.right not in config.views:
                    raise WorkspaceValidationError(f"derived view {view_id} references unknown input views")
                left = config.views[derive.left]
                right = config.views[derive.right]
                left_space = left.coordinate_space_id
                right_space = right.coordinate_space_id
                if left_space != right_space or view.coordinate_space_id != left_space:
                    raise CoordinateSpaceError(
                        f"derived view {view_id} violates coordinate space legality: {left_space!r} vs {right_space!r}"
                    )
                if derive.alignment not in config.alignments:
                    raise WorkspaceValidationError(
                        f"derived view {view_id} references unknown alignment {derive.alignment!r}"
                    )
            elif derive.kind == "concatenate":
                missing = [input_view for input_view in derive.inputs if input_view not in config.views]
                if missing:
                    raise WorkspaceValidationError(f"derived view {view_id} references unknown input views {missing!r}")
                input_spaces = {
                    input_view: config.views[input_view].coordinate_space_id for input_view in derive.inputs
                }
                unique_spaces = set(input_spaces.values())
                if len(unique_spaces) > 1 and not all(
                    _view_declares_reduced_space(config, input_view) for input_view in derive.inputs
                ):
                    rendered = ", ".join(f"{input_view}={space}" for input_view, space in input_spaces.items())
                    raise CoordinateSpaceError(
                        "derived view "
                        f"{view_id} concatenate inputs must share one coordinate "
                        f"space or all be reduced; got {rendered}"
                    )
            elif derive.kind in {"aggregate_by_key", "apply_reducer", "normalize"}:
                if derive.view not in config.views:
                    raise WorkspaceValidationError(
                        f"derived view {view_id} references unknown input view {derive.view!r}"
                    )
                if derive.kind in {"aggregate_by_key", "normalize"}:
                    input_view = config.views[derive.view]
                    if view.coordinate_space_id != input_view.coordinate_space_id:
                        raise CoordinateSpaceError(
                            f"derived view {view_id} violates coordinate space legality: "
                            f"{view.coordinate_space_id!r} vs {input_view.coordinate_space_id!r}"
                        )

    for view_id, view in config.views.items():
        if isinstance(view, SourceBackedViewConfig):
            if view.source not in config.sources:
                raise WorkspaceValidationError(f"view {view_id} references unknown source {view.source!r}")
            source = config.sources[view.source]
            if view.vector.kind == "bundle_matrix":
                if not isinstance(source, MatrixBundleSourceConfig):
                    raise WorkspaceValidationError(
                        f"view {view_id} uses vector kind bundle_matrix but source {view.source!r} is not matrix_bundle"
                    )
            elif isinstance(source, MatrixBundleSourceConfig):
                raise WorkspaceValidationError(
                    f"view {view_id} uses vector kind column but source {view.source!r} is matrix_bundle"
                )

    for scalar_id, scalar in config.scalars.items():
        if scalar.derive.kind == "vector_norm" and scalar.derive.view not in config.views:
            raise WorkspaceValidationError(f"scalar {scalar_id} references unknown view {scalar.derive.view!r}")

    for landmark_id, landmark in config.landmarks.items():
        if landmark.source not in config.sources:
            raise WorkspaceValidationError(f"landmark {landmark_id} references unknown source {landmark.source!r}")

    for reference_set_id in config.reference_sets:
        validate_identifier(reference_set_id, label="reference_set id")

    for candidate_set_id, candidate_set in config.candidate_sets.items():
        validate_identifier(candidate_set_id, label="candidate_set id")
        _validate_representation_identity(candidate_set_id, owner="candidate_set id")
        for view_id in candidate_set.views:
            validate_identifier(view_id, label=f"candidate_set {candidate_set_id} view")
            if view_id not in config.views:
                raise WorkspaceValidationError(f"candidate_set {candidate_set_id} references unknown view {view_id!r}")
        for view_id in candidate_set.panel_titles:
            validate_identifier(view_id, label=f"candidate_set {candidate_set_id} panel title view")
            if view_id not in config.views:
                raise WorkspaceValidationError(
                    f"candidate_set {candidate_set_id} panel_titles references unknown view {view_id!r}"
                )

    for plot_id, plot in config.plots.items():
        _validate_plot(config, plot_id, plot)

    for cohort_id, cohort in config.cohorts.items():
        if cohort.source not in config.sources:
            raise WorkspaceValidationError(f"cohort {cohort_id} references unknown source {cohort.source!r}")

    for export_id, export in config.exports.items():
        seen_block_ids: set[str] = set()
        for block in export.blocks:
            validate_identifier(block.block_id, label=f"export {export_id} block id")
            if block.block_id in seen_block_ids:
                raise WorkspaceValidationError(f"export {export_id} reuses block id {block.block_id!r}")
            seen_block_ids.add(block.block_id)
            if block.alignment is not None and block.alignment not in config.alignments:
                raise WorkspaceValidationError(
                    f"export {export_id} block {block.block_id} references unknown alignment {block.alignment!r}"
                )

    for notebook_id, notebook in config.notebooks.items():
        _validate_notebook(config, notebook_id, notebook)

    for notebook_id, notebook in config.notebooks.items():
        if notebook.default_deliverable not in config.deliverables:
            raise WorkspaceValidationError(
                f"notebook {notebook_id} references unknown default deliverable {notebook.default_deliverable!r}"
            )

    for recipe_id, recipe in config.recipes.items():
        _validate_recipe(recipe_id, recipe)

    plot_owners: dict[str, str] = {}
    for deliverable_id, deliverable in config.deliverables.items():
        _validate_deliverable(config, deliverable_id, deliverable)
        for plot_id in deliverable.outputs.get("plots", []):
            owner = plot_owners.get(plot_id)
            if owner is not None and owner != deliverable_id:
                raise WorkspaceValidationError(
                    f"plot output {plot_id!r} is owned by multiple deliverables: {owner!r}, {deliverable_id!r}"
                )
            plot_owners[plot_id] = deliverable_id

    if config.study_binding is not None:
        record_root = resolve_repo_path(config.study_binding.record_root)
        missing_record_files = missing_required_files(record_root, REQUIRED_STUDY_RECORD_FILES)
        if not record_root.is_dir():
            raise WorkspaceValidationError(f"study record_root does not exist: {record_root}")
        if missing_record_files:
            raise WorkspaceValidationError(
                "study record_root is missing required checked-in record files: "
                f"{record_root} ({', '.join(sorted(missing_record_files))})"
            )
        deliverable_docs_root = resolve_repo_path(config.study_binding.deliverable_docs_root)
        missing_docs_files = missing_required_files(deliverable_docs_root, REQUIRED_STUDY_DELIVERABLE_DOC_FILES)
        if not deliverable_docs_root.is_dir():
            raise WorkspaceValidationError(f"study deliverable_docs_root does not exist: {deliverable_docs_root}")
        if missing_docs_files:
            raise WorkspaceValidationError(
                "study deliverable_docs_root is missing required files: "
                f"{deliverable_docs_root} ({', '.join(sorted(missing_docs_files))})"
            )


def _validate_recipe(recipe_id: str, recipe: RecipeConfig) -> None:
    step_ids: set[str] = set()
    for step in recipe.steps:
        validate_identifier(step.id, label=f"recipe {recipe_id} step id")
        if step.id in step_ids:
            raise WorkspaceValidationError(f"recipe {recipe_id} reuses step id {step.id!r}")
        step_ids.add(step.id)
        if step.op not in SUPPORTED_RECIPE_OPS:
            raise WorkspaceValidationError(f"recipe {recipe_id} uses unsupported op {step.op!r}")

    for step in recipe.steps:
        for dependency in step.depends_on:
            if dependency not in step_ids:
                raise WorkspaceValidationError(
                    f"recipe {recipe_id} step {step.id} depends on unknown step {dependency!r}"
                )

    try:
        topological_step_order(recipe.steps)
    except ValueError as exc:
        raise WorkspaceValidationError(f"recipe {recipe_id} contains a cycle") from exc


def _validate_representation_identity(value: str, *, owner: str) -> None:
    try:
        validate_representation_identity(value, owner=owner)
    except ValueError as exc:
        raise WorkspaceValidationError(str(exc)) from exc


def _validate_notebook(config: WorkspaceConfig, notebook_id: str, notebook: NotebookConfig) -> None:
    assert isinstance(notebook, WorkspaceNotebookConfig)
    validate_identifier(notebook.default_deliverable, label=f"notebook {notebook_id} default deliverable")
    seen_plot_ids: set[str] = set()
    for plot_id in notebook.ordered_plots:
        validate_identifier(plot_id, label=f"notebook {notebook_id} ordered plot")
        if plot_id in seen_plot_ids:
            raise WorkspaceValidationError(f"notebook {notebook_id} reuses ordered plot {plot_id!r}")
        seen_plot_ids.add(plot_id)
        if plot_id not in config.plots:
            raise WorkspaceValidationError(f"notebook {notebook_id} references unknown plot {plot_id!r}")
        visibility_tier = str(getattr(config.plots[plot_id], "visibility_tier", "primary") or "primary")
        if visibility_tier in {"debug", "hidden"}:
            raise WorkspaceValidationError(
                f"notebook {notebook_id} ordered plot {plot_id!r} must not use visibility_tier {visibility_tier!r}"
            )
        if not any(plot_id in deliverable.outputs.get("plots", []) for deliverable in config.deliverables.values()):
            raise WorkspaceValidationError(
                f"notebook {notebook_id} ordered plot {plot_id!r} is not owned by any deliverable output"
            )
    for label, view_ids in (
        ("geometry_order", notebook.geometry_order),
        ("candidate_grid_views", notebook.candidate_grid_views),
        ("default_compare_views", notebook.default_compare_views),
    ):
        for view_id in view_ids:
            validate_identifier(view_id, label=f"notebook {notebook_id} {label} view")
            if view_id not in config.views:
                raise WorkspaceValidationError(f"notebook {notebook_id} references unknown view {view_id!r} in {label}")
    for candidate_set_id in notebook.candidate_sets:
        validate_identifier(candidate_set_id, label=f"notebook {notebook_id} candidate_set")
        if candidate_set_id not in config.candidate_sets:
            raise WorkspaceValidationError(
                f"notebook {notebook_id} references unknown candidate_set {candidate_set_id!r}"
            )
    if notebook.default_candidate_set is not None:
        validate_identifier(notebook.default_candidate_set, label=f"notebook {notebook_id} default_candidate_set")
        if notebook.default_candidate_set not in config.candidate_sets:
            raise WorkspaceValidationError(
                f"notebook {notebook_id} references unknown default_candidate_set {notebook.default_candidate_set!r}"
            )
    if notebook.default_reference_set is not None:
        validate_identifier(notebook.default_reference_set, label=f"notebook {notebook_id} default_reference_set")
        if notebook.default_reference_set not in config.reference_sets:
            raise WorkspaceValidationError(
                f"notebook {notebook_id} references unknown default_reference_set {notebook.default_reference_set!r}"
            )


def _validate_plot(config: WorkspaceConfig, plot_id: str, plot: Any) -> None:
    if not getattr(plot, "semantics_ref", None):
        raise WorkspaceValidationError(f"plot {plot_id} must declare semantics_ref")
    if getattr(plot, "annotation", None) is not None:
        annotation = plot.annotation
        assert annotation is not None
        if annotation.reference_set not in config.reference_sets:
            raise WorkspaceValidationError(
                f"plot {plot_id} references unknown reference_set {annotation.reference_set!r}"
            )
    if plot.kind == "projection_scatter":
        validate_identifier(plot.projection, label=f"plot {plot_id} projection")
        return
    if plot.kind == "projection_grid":
        for projection_id in plot.projections:
            validate_identifier(projection_id, label=f"plot {plot_id} projection")
        return
    if plot.kind == "heatmap":
        if plot.enrichment is not None:
            validate_identifier(plot.enrichment, label=f"plot {plot_id} enrichment")
        if plot.scalar is not None:
            validate_identifier(plot.scalar, label=f"plot {plot_id} scalar")
        return
    if plot.kind == "heatmap_grid":
        for scalar_id in plot.scalars:
            validate_identifier(scalar_id, label=f"plot {plot_id} scalar")
        return
    if plot.kind == "distance_scatter":
        validate_identifier(plot.distance, label=f"plot {plot_id} distance")
        return
    if plot.kind == "xy_scatter":
        for label, value in (("scalar", plot.scalar), ("distance", plot.distance)):
            if value is not None:
                validate_identifier(value, label=f"plot {plot_id} {label}")
        return
    if plot.kind == "xy_scatter_grid":
        for scalar_id in plot.scalars:
            validate_identifier(scalar_id, label=f"plot {plot_id} scalar")
        return
    if plot.kind == "paired_xy_scatter_grid":
        for scalar_id in plot.scalars:
            validate_identifier(scalar_id, label=f"plot {plot_id} scalar")
        return
    if plot.kind == "categorical_count":
        validate_identifier(plot.scalar, label=f"plot {plot_id} scalar")
        return
    if plot.kind == "metric_panel_grid":
        validate_identifier(plot.scalar, label=f"plot {plot_id} scalar")
        return
    if plot.kind == "distribution":
        for label, value in (
            ("scalar", plot.scalar),
            ("distance", plot.distance),
            ("enrichment", plot.enrichment),
            ("agreement", plot.agreement),
        ):
            if value is not None:
                validate_identifier(value, label=f"plot {plot_id} {label}")
        return
    if plot.kind == "distribution_grid":
        for scalar_id in plot.scalars:
            validate_identifier(scalar_id, label=f"plot {plot_id} scalar")
        return
    if plot.kind == "curve":
        if plot.reducer is not None:
            validate_identifier(plot.reducer, label=f"plot {plot_id} reducer")
        return
    if plot.kind == "curve_grid":
        for reducer_id in plot.reducers:
            validate_identifier(reducer_id, label=f"plot {plot_id} reducer")
        return
    if plot.kind == "correspondence_heatmap":
        validate_identifier(plot.left_cluster, label=f"plot {plot_id} left_cluster")
        validate_identifier(plot.right_cluster, label=f"plot {plot_id} right_cluster")
        return
    if plot.kind == "agreement_summary_grid":
        for agreement_id in plot.agreements:
            validate_identifier(agreement_id, label=f"plot {plot_id} agreement")
        return
    validate_identifier(plot.agreement, label=f"plot {plot_id} agreement")


def _validate_deliverable(config: WorkspaceConfig, deliverable_id: str, deliverable: DeliverableConfig) -> None:
    if deliverable.recipe not in config.recipes:
        raise WorkspaceValidationError(f"deliverable {deliverable_id} references unknown recipe {deliverable.recipe!r}")
    recipe = config.recipes[deliverable.recipe]
    expected_outputs: set[tuple[str, str]] = set()
    for step in recipe.steps:
        try:
            expected_outputs.update(expected_step_artifacts(step.op, step.params))
        except ValueError as exc:  # pragma: no cover - recipe validation already covers this
            raise WorkspaceValidationError(str(exc)) from exc

    for category, ids in {**deliverable.requires, **deliverable.outputs}.items():
        if category not in SUPPORTED_DELIVERABLE_REFERENCE_CATEGORIES:
            raise WorkspaceValidationError(
                f"deliverable {deliverable_id} uses unsupported reference category {category!r}"
            )
        if not ids:
            raise WorkspaceValidationError(f"deliverable {deliverable_id} declares an empty {category!r} list")

    for docs_ref in deliverable.docs_refs:
        if config.study_binding is None:
            raise WorkspaceValidationError(f"deliverable {deliverable_id} uses docs_refs without study_binding")
        try:
            resolve_docs_ref_path(
                study_id=config.study_binding.study_id,
                deliverable_docs_root=config.study_binding.deliverable_docs_root,
                docs_ref=docs_ref,
                workspace_id=config.workspace.id,
            )
        except WorkspaceValidationError as exc:
            raise WorkspaceValidationError(
                f"deliverable {deliverable_id} has invalid docs_ref {docs_ref!r}: {exc}"
            ) from exc

    config_sections = {
        "sources": config.sources,
        "views": config.views,
        "alignments": config.alignments,
        "scalars": config.scalars,
        "landmarks": config.landmarks,
        "cohorts": config.cohorts,
        "exports": config.exports,
        "notebooks": config.notebooks,
        "recipes": config.recipes,
    }
    for section_name, references in [("requires", deliverable.requires), ("outputs", deliverable.outputs)]:
        for category, ids in references.items():
            section = config_sections.get(category)
            artifact_kind = ARTIFACT_REFERENCE_CATEGORIES.get(category)
            for item_id in ids:
                if section is not None and item_id not in section:
                    raise WorkspaceValidationError(
                        f"deliverable {deliverable_id} references unknown {category[:-1]} {item_id!r} in {section_name}"
                    )
                if (
                    section_name == "outputs"
                    and artifact_kind is not None
                    and (artifact_kind, item_id) not in expected_outputs
                ):
                    raise WorkspaceValidationError(
                        f"deliverable {deliverable_id} references output {item_id!r} in {category}, "
                        "but the linked recipe does not produce it"
                    )

    for acceptance_check in deliverable.acceptance_checks:
        _validate_acceptance_check(config, deliverable_id, deliverable, acceptance_check)


def _validate_acceptance_check(
    config: WorkspaceConfig,
    deliverable_id: str,
    deliverable: DeliverableConfig,
    acceptance_check: AcceptanceCheckConfig,
) -> None:
    if acceptance_check.kind == "required_plot_kind":
        plot_ids = deliverable.outputs.get("plots", [])
        if not plot_ids:
            raise WorkspaceValidationError(
                f"deliverable {deliverable_id} requires a plot output for acceptance check {acceptance_check.kind}"
            )
        value = str(acceptance_check.value)
        invalid = [plot_id for plot_id in plot_ids if config.plots[plot_id].kind != value]
        if invalid:
            raise WorkspaceValidationError(
                f"deliverable {deliverable_id} acceptance check required_plot_kind={value!r} failed for {invalid!r}"
            )
        return
    if acceptance_check.kind == "required_reference_set":
        plot_ids = deliverable.outputs.get("plots", [])
        if not plot_ids:
            raise WorkspaceValidationError(
                f"deliverable {deliverable_id} requires a plot output for acceptance check {acceptance_check.kind}"
            )
        value = str(acceptance_check.value)
        invalid = []
        for plot_id in plot_ids:
            annotation = getattr(config.plots[plot_id], "annotation", None)
            if annotation is None or annotation.reference_set != value:
                invalid.append(plot_id)
        if invalid:
            raise WorkspaceValidationError(
                f"deliverable {deliverable_id} acceptance check required_reference_set={value!r} failed for {invalid!r}"
            )
        return
    if acceptance_check.kind == "require_reference_set_in_every_panel":
        if not isinstance(acceptance_check.value, bool):
            raise WorkspaceValidationError(
                f"deliverable {deliverable_id} acceptance check {acceptance_check.kind} requires a boolean value"
            )
        return
    raise WorkspaceValidationError(
        f"deliverable {deliverable_id} uses unsupported acceptance check {acceptance_check.kind!r}"
    )
