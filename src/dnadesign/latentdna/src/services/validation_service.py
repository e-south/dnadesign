"""
Workspace validation services for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.errors import WorkspaceValidationError
from ..contracts.workspace import ColumnCohortConfig, PromoterMetadataCohortConfig, SourceBackedViewConfig
from ..sources.resolver import inspect_source_schema, resolve_source
from ..workspaces.loader import load_workspace_config, resolve_repo_path

_PROMOTER_METADATA_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "design_family": {"densegen__plan", "usr_label__primary", "template_id"},
    "design_regulator_composition": {
        "densegen__plan",
        "densegen__required_regulators",
        "usr_label__primary",
        "template_id",
    },
    "sigma70_variant": {"densegen__plan", "usr_label__primary", "template_id"},
    "campaign_prior": {"densegen__plan", "usr_label__primary", "template_id"},
    "is_control": {"densegen__plan", "usr_label__primary", "template_id"},
    "source_class": {"densegen__plan", "usr_label__primary", "template_id"},
}


def _deep_validate_workspace(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    source_columns: dict[str, set[str]] = {}
    source_details: list[dict[str, object]] = []
    for source_id in sorted(context.config.sources):
        source = context.require_source(source_id)
        resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
        schema_info = inspect_source_schema(resolved)
        columns = set(schema_info["columns"])
        required_columns = [source.record_key, source.subject_key]
        if source.context_key is not None:
            required_columns.append(source.context_key)
        missing_columns = [name for name in required_columns if name not in columns]
        if missing_columns:
            missing_rendered = ", ".join(missing_columns)
            raise WorkspaceValidationError(f"source {source_id} is missing required columns: {missing_rendered}")
        source_columns[source_id] = columns
        source_details.append(
            {
                "source_id": source_id,
                "kind": source.kind,
                "path": schema_info["path"],
                "row_count": schema_info["row_count"],
                "required_columns": required_columns,
                "vector_columns": schema_info["vector_columns"],
            }
        )

    view_details: list[dict[str, object]] = []
    for view_id in sorted(context.config.views):
        view = context.require_view(view_id)
        if isinstance(view, SourceBackedViewConfig):
            columns = source_columns[view.source]
            view_detail = {
                "view_id": view_id,
                "declaration_kind": "source_backed",
                "source": view.source,
                "vector_kind": view.vector.kind,
                "coordinate_space_id": view.coordinate_space_id,
            }
            if view.vector.kind == "column" and view.vector.name not in columns:
                raise WorkspaceValidationError(
                    f"view {view_id} vector column is missing from source {view.source}: {view.vector.name}"
                )
            if view.vector.kind == "column":
                view_detail["vector_column"] = view.vector.name
            view_details.append(view_detail)
            continue
        view_details.append(
            {
                "view_id": view_id,
                "declaration_kind": "derived",
                "derive_kind": view.derive.kind,
                "coordinate_space_id": view.coordinate_space_id,
            }
        )

    landmark_details: list[dict[str, object]] = []
    for landmark_id in sorted(context.config.landmarks):
        landmark = context.require_landmark(landmark_id)
        column = str(landmark.where["column"])
        if column not in source_columns[landmark.source]:
            raise WorkspaceValidationError(
                f"landmark {landmark_id} selector column is missing from source {landmark.source}: {column}"
            )
        landmark_details.append(
            {
                "landmark_id": landmark_id,
                "source": landmark.source,
                "selector_column": column,
                "representation_mode": landmark.representation.mode,
            }
        )

    cohort_details: list[dict[str, object]] = []
    for cohort_id in sorted(context.config.cohorts):
        cohort = context.require_cohort(cohort_id)
        if isinstance(cohort, ColumnCohortConfig):
            if cohort.column not in source_columns[cohort.source]:
                raise WorkspaceValidationError(
                    f"cohort {cohort_id} column is missing from source {cohort.source}: {cohort.column}"
                )
            cohort_details.append(
                {
                    "cohort_id": cohort_id,
                    "source": cohort.source,
                    "kind": cohort.kind,
                    "column": cohort.column,
                }
            )
            continue
        assert isinstance(cohort, PromoterMetadataCohortConfig)
        missing = sorted(_PROMOTER_METADATA_REQUIRED_COLUMNS[cohort.derive] - source_columns[cohort.source])
        if missing:
            raise WorkspaceValidationError(
                f"cohort {cohort_id} promoter metadata inputs are missing from source {cohort.source}: {missing}"
            )
        cohort_details.append(
            {
                "cohort_id": cohort_id,
                "source": cohort.source,
                "kind": cohort.kind,
                "derive": cohort.derive,
            }
        )

    study_binding = None
    if context.config.study_binding is not None:
        study_dir = resolve_repo_path(context.config.study_binding.study_dir)
        required_files = ["campaign.yaml", "datasets.yaml", "status.md", "ops.study.yaml"]
        missing = [name for name in required_files if not (study_dir / name).exists()]
        if missing:
            raise WorkspaceValidationError(
                f"study binding directory is missing required files: {study_dir} ({', '.join(sorted(missing))})"
            )
        study_binding = {
            "kind": context.config.study_binding.kind,
            "study_dir": study_dir.as_posix(),
            "required_files": required_files,
        }

    return {
        "schema_version": "latentdna.validation_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "deep": True,
        "study_binding": study_binding,
        "source_details": source_details,
        "view_details": view_details,
        "landmark_details": landmark_details,
        "cohort_details": cohort_details,
        "sources": sorted(context.config.sources),
        "views": sorted(context.config.views),
        "notebooks": sorted(context.config.notebooks),
        "recipes": sorted(context.config.recipes),
        "deliverables": sorted(context.config.deliverables),
    }


def validate_workspace(workspace: str | Path, *, deep: bool = False) -> dict[str, object]:
    if deep:
        return _deep_validate_workspace(workspace)
    context = load_workspace_config(workspace)
    return {
        "schema_version": "latentdna.validation_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "deep": False,
        "sources": sorted(context.config.sources),
        "views": sorted(context.config.views),
        "notebooks": sorted(context.config.notebooks),
        "recipes": sorted(context.config.recipes),
        "deliverables": sorted(context.config.deliverables),
    }
