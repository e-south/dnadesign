"""
Workspace loading and scaffolding for latentdna.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Any

import yaml

from ..contracts.deliverable import ARTIFACT_REFERENCE_CATEGORIES, SUPPORTED_DELIVERABLE_REFERENCE_CATEGORIES
from ..contracts.errors import CoordinateSpaceError, WorkspaceValidationError
from ..contracts.ids import validate_identifier
from ..contracts.notebook import (
    SUPPORTED_NOTEBOOK_ARTIFACT_KINDS,
    ArtifactReviewNotebookConfig,
    NotebookConfig,
    WorkspaceBrowserNotebookConfig,
)
from ..contracts.recipe import SUPPORTED_RECIPE_OPS, expected_step_artifacts, topological_step_order
from ..contracts.workspace import (
    AlignmentConfig,
    CohortConfig,
    DeliverableConfig,
    DerivedViewConfig,
    ExportConfig,
    LandmarkConfig,
    MatrixBundleSourceConfig,
    RecipeConfig,
    ScalarConfig,
    SourceBackedViewConfig,
    WorkspaceConfig,
)
from ..io.json_io import read_json


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def builtin_templates_dir() -> Path:
    return project_root() / "src" / "dnadesign" / "latentdna" / "workspaces" / "templates"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    repo_candidate = project_root() / candidate
    if repo_candidate.exists():
        return repo_candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def default_workspace_root() -> tuple[Path, str]:
    env_value = Path(Path.cwd().as_posix())
    source = "cwd"
    from os import environ

    if environ.get("LATENTDNA_WORKSPACE_ROOT"):
        env_value = Path(environ["LATENTDNA_WORKSPACE_ROOT"])
        source = "env"
    return env_value.resolve(), source


def resolve_workspace_path(workspace: str | Path) -> Path:
    candidate = Path(workspace)
    if candidate.is_file():
        return candidate.parent.resolve()
    if candidate.is_dir():
        return candidate.resolve()
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.is_dir():
        return cwd_candidate.resolve()
    raise WorkspaceValidationError(f"workspace not found: {workspace}")


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


def _validate_workspace_config(config: WorkspaceConfig) -> None:
    validate_identifier(config.workspace.id, label="workspace.id")
    for source_id in config.sources:
        validate_identifier(source_id, label="source id")
    for view_id in config.views:
        validate_identifier(view_id, label="view id")
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

    for alignment_id, alignment in config.alignments.items():
        if alignment.left not in config.views and alignment.left not in config.sources:
            raise WorkspaceValidationError(f"alignment {alignment_id} references unknown left input {alignment.left!r}")
        if alignment.right not in config.views and alignment.right not in config.sources:
            raise WorkspaceValidationError(
                f"alignment {alignment_id} references unknown right input {alignment.right!r}"
            )
        if isinstance(alignment.on, list) and not alignment.on:
            raise WorkspaceValidationError(f"alignment {alignment_id} must declare at least one key column in 'on'")

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

    for plot_id, plot in config.plots.items():
        _validate_plot(plot_id, plot)

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
        _validate_notebook(notebook_id, notebook)

    for notebook_id, notebook in config.notebooks.items():
        if (
            isinstance(notebook, WorkspaceBrowserNotebookConfig)
            and notebook.default_deliverable not in config.deliverables
        ):
            raise WorkspaceValidationError(
                f"notebook {notebook_id} references unknown default deliverable {notebook.default_deliverable!r}"
            )

    for recipe_id, recipe in config.recipes.items():
        _validate_recipe(recipe_id, recipe)

    for deliverable_id, deliverable in config.deliverables.items():
        _validate_deliverable(config, deliverable_id, deliverable)


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


def _validate_notebook(notebook_id: str, notebook: NotebookConfig) -> None:
    if isinstance(notebook, WorkspaceBrowserNotebookConfig):
        validate_identifier(notebook.default_deliverable, label=f"notebook {notebook_id} default deliverable")
        return

    assert isinstance(notebook, ArtifactReviewNotebookConfig)
    aliases: set[str] = set()
    for artifact in notebook.artifacts:
        validate_identifier(artifact.id, label=f"notebook {notebook_id} artifact id")
        if artifact.kind not in SUPPORTED_NOTEBOOK_ARTIFACT_KINDS:
            raise WorkspaceValidationError(
                f"notebook {notebook_id} uses unsupported notebook artifact kind {artifact.kind!r}"
            )
        alias = artifact.alias or artifact.id
        validate_identifier(alias, label=f"notebook {notebook_id} artifact alias")
        if alias in aliases:
            raise WorkspaceValidationError(f"notebook {notebook_id} reuses artifact alias {alias!r}")
        aliases.add(alias)


def _validate_plot(plot_id: str, plot) -> None:
    if plot.kind == "projection_scatter":
        validate_identifier(plot.projection, label=f"plot {plot_id} projection")
        return
    if plot.kind == "projection_grid":
        for projection_id in plot.projections:
            validate_identifier(projection_id, label=f"plot {plot_id} projection")
        return
    if plot.kind == "heatmap":
        validate_identifier(plot.enrichment, label=f"plot {plot_id} enrichment")
        return
    if plot.kind == "distance_scatter":
        validate_identifier(plot.distance, label=f"plot {plot_id} distance")
        return
    if plot.kind == "xy_scatter":
        for label, value in (("scalar", plot.scalar), ("distance", plot.distance)):
            if value is not None:
                validate_identifier(value, label=f"plot {plot_id} {label}")
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
    if plot.kind == "curve":
        if plot.reducer is not None:
            validate_identifier(plot.reducer, label=f"plot {plot_id} reducer")
        return
    if plot.kind == "correspondence_heatmap":
        validate_identifier(plot.left_cluster, label=f"plot {plot_id} left_cluster")
        validate_identifier(plot.right_cluster, label=f"plot {plot_id} right_cluster")
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


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    workspace_dir: Path
    config_path: Path
    config: WorkspaceConfig

    @property
    def workspace_id(self) -> str:
        return self.config.workspace.id

    @property
    def output_root(self) -> Path:
        candidate = Path(self.config.workspace.output_root)
        if not candidate.is_absolute():
            candidate = self.workspace_dir / candidate
        resolved = candidate.resolve()
        if self.workspace_dir.resolve() not in resolved.parents and resolved != self.workspace_dir.resolve():
            raise WorkspaceValidationError(f"workspace output root must stay inside the workspace: {resolved}")
        return resolved

    @property
    def analysis_dtype(self) -> str:
        return self.config.defaults.analysis_dtype

    def require_source(self, source_id: str):
        if source_id not in self.config.sources:
            raise WorkspaceValidationError(f"unknown source: {source_id}")
        return self.config.sources[source_id]

    def require_view(self, view_id: str):
        if view_id not in self.config.views:
            raise WorkspaceValidationError(f"unknown view: {view_id}")
        return self.config.views[view_id]

    def require_source_view(self, view_id: str) -> SourceBackedViewConfig:
        view = self.require_view(view_id)
        if not isinstance(view, SourceBackedViewConfig):
            raise WorkspaceValidationError(f"view {view_id} is not a source-backed view")
        return view

    def require_alignment(self, alignment_id: str) -> AlignmentConfig:
        if alignment_id not in self.config.alignments:
            raise WorkspaceValidationError(f"unknown alignment: {alignment_id}")
        return self.config.alignments[alignment_id]

    def require_scalar(self, scalar_id: str) -> ScalarConfig:
        if scalar_id not in self.config.scalars:
            raise WorkspaceValidationError(f"unknown scalar: {scalar_id}")
        return self.config.scalars[scalar_id]

    def require_landmark(self, landmark_id: str) -> LandmarkConfig:
        if landmark_id not in self.config.landmarks:
            raise WorkspaceValidationError(f"unknown landmark: {landmark_id}")
        return self.config.landmarks[landmark_id]

    def require_plot(self, plot_id: str):
        if plot_id not in self.config.plots:
            raise WorkspaceValidationError(f"unknown plot recipe: {plot_id}")
        return self.config.plots[plot_id]

    def require_cohort(self, cohort_id: str) -> CohortConfig:
        if cohort_id not in self.config.cohorts:
            raise WorkspaceValidationError(f"unknown cohort: {cohort_id}")
        return self.config.cohorts[cohort_id]

    def require_export(self, export_id: str) -> ExportConfig:
        if export_id not in self.config.exports:
            raise WorkspaceValidationError(f"unknown export: {export_id}")
        return self.config.exports[export_id]

    def require_notebook(self, notebook_id: str) -> NotebookConfig:
        if notebook_id not in self.config.notebooks:
            raise WorkspaceValidationError(f"unknown notebook: {notebook_id}")
        return self.config.notebooks[notebook_id]

    def require_recipe(self, recipe_id: str) -> RecipeConfig:
        if recipe_id not in self.config.recipes:
            raise WorkspaceValidationError(f"unknown recipe: {recipe_id}")
        return self.config.recipes[recipe_id]

    def require_deliverable(self, deliverable_id: str) -> DeliverableConfig:
        if deliverable_id not in self.config.deliverables:
            raise WorkspaceValidationError(f"unknown deliverable: {deliverable_id}")
        return self.config.deliverables[deliverable_id]

    def require_source_backed_reference_source(self, ref_id: str):
        if ref_id in self.config.sources:
            return self.require_source(ref_id)
        view = self.require_source_view(ref_id)
        return self.require_source(view.source)

    def read_manifest(self, path: Path) -> dict[str, Any]:
        return read_json(path)


def load_workspace_config(workspace: str | Path) -> WorkspaceContext:
    workspace_dir = resolve_workspace_path(workspace)
    config_path = workspace_dir / "config.yaml"
    if not config_path.exists():
        raise WorkspaceValidationError(f"workspace config.yaml not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = WorkspaceConfig.model_validate(payload)
    _validate_workspace_config(config)
    return WorkspaceContext(workspace_dir=workspace_dir, config_path=config_path, config=config)


def _read_study_datasets(study_dir: Path) -> dict[str, dict[str, Any]]:
    required_files = ["campaign.yaml", "datasets.yaml", "status.md", "ops.study.yaml"]
    missing = [name for name in required_files if not (study_dir / name).exists()]
    if missing:
        raise WorkspaceValidationError(
            f"study record is missing required files in {study_dir}: {', '.join(sorted(missing))}"
        )
    payload = yaml.safe_load((study_dir / "datasets.yaml").read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise WorkspaceValidationError(f"study datasets registry is empty: {study_dir / 'datasets.yaml'}")
    by_role: dict[str, dict[str, Any]] = {}
    for entry in datasets:
        if not isinstance(entry, dict):
            raise WorkspaceValidationError(f"study dataset entries must be mappings: {study_dir / 'datasets.yaml'}")
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise WorkspaceValidationError(f"study dataset entry is missing role: {entry!r}")
        by_role[role] = entry
    return by_role


def _hydrate_template_from_study(payload: dict[str, Any], *, study_dir: Path, workspace_dir: Path) -> None:
    datasets = _read_study_datasets(study_dir)
    source_role_map = {
        "anchor60": "merged_anchor_source",
        "ctx1k": "construct_context",
    }
    for source_id, role in source_role_map.items():
        source_payload = payload.get("sources", {}).get(source_id)
        if not isinstance(source_payload, dict):
            continue
        study_entry = datasets.get(role)
        if study_entry is None:
            raise WorkspaceValidationError(f"study record is missing dataset role {role!r} for source {source_id!r}")
        usr_root = study_entry.get("usr_root")
        dataset_id = study_entry.get("dataset")
        if not isinstance(usr_root, str) or not usr_root:
            raise WorkspaceValidationError(f"study dataset role {role!r} is missing usr_root")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise WorkspaceValidationError(f"study dataset role {role!r} is missing dataset")
        resolved_usr_root = resolve_repo_path(usr_root)
        source_payload["kind"] = "usr"
        source_payload["root"] = Path(relpath(resolved_usr_root, workspace_dir)).as_posix()
        source_payload["dataset"] = dataset_id

    study_dir_text: str
    try:
        study_dir_text = study_dir.relative_to(project_root()).as_posix()
    except ValueError:
        study_dir_text = study_dir.as_posix()
    payload["study_binding"] = {
        "kind": "dnadesign_study",
        "study_dir": study_dir_text,
        "readiness_vocabulary": ["missing", "attention", "ok"],
    }


def scaffold_workspace(*, workspace_dir: Path, template: str, from_study_dir: str | Path | None = None) -> Path:
    template_dir = builtin_templates_dir() / template
    if not template_dir.is_dir():
        raise WorkspaceValidationError(f"unknown workspace template: {template}")
    if workspace_dir.exists():
        raise WorkspaceValidationError(f"workspace already exists: {workspace_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=False)
    try:
        for source in template_dir.rglob("*"):
            relative = source.relative_to(template_dir)
            target = workspace_dir / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        config_path = workspace_dir / "config.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["workspace"]["id"] = workspace_dir.name
        if from_study_dir is not None:
            _hydrate_template_from_study(
                payload,
                study_dir=resolve_repo_path(from_study_dir),
                workspace_dir=workspace_dir,
            )
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        (workspace_dir / "outputs" / "latentdna" / "logs" / "audit").mkdir(parents=True, exist_ok=True)
        return workspace_dir
    except Exception:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise
