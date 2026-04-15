"""
Workspace validation services for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.errors import WorkspaceValidationError
from ..contracts.notebook import WorkspaceNotebookControls
from ..contracts.workspace import ColumnCohortConfig, PromoterMetadataCohortConfig, SourceBackedViewConfig
from ..io.json_io import read_json
from ..io.parquet_io import read_schema
from ..sources.resolver import inspect_source_schema, resolve_source
from ..workspaces.loader import load_workspace_config
from ..workspaces.paths import resolve_repo_path

_PROMOTER_METADATA_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "design_family": {"densegen__plan", "usr_label__primary"},
    "design_regulator_composition": {
        "densegen__plan",
        "densegen__required_regulators",
        "usr_label__primary",
    },
    "sigma70_variant": {"densegen__plan", "usr_label__primary"},
    "campaign_prior": {"densegen__plan", "usr_label__primary"},
    "is_control": {"densegen__plan", "usr_label__primary"},
    "source_class": {"densegen__plan", "usr_label__primary"},
}

_MATERIALIZED_VIEW_METADATA_REQUIRED_COLUMNS = {
    "densegen__plan",
    "densegen__required_regulators",
    "usr_label__primary",
    "template_id",
    "construct__template_id",
}


def _deep_validate_notebook_artifacts(context) -> list[dict[str, object]]:
    notebook_details: list[dict[str, object]] = []
    for notebook_id in sorted(context.config.notebooks):
        notebook_dir = context.output_root / "notebooks" / notebook_id
        notebook_path = notebook_dir / "notebook.py"
        controls_path = notebook_dir / "controls.json"
        artifact_present = notebook_dir.exists()
        detail: dict[str, object] = {
            "notebook_id": notebook_id,
            "artifact_present": artifact_present,
        }
        if not artifact_present:
            notebook_details.append(detail)
            continue
        if not notebook_path.is_file():
            raise WorkspaceValidationError(f"workspace notebook artifact is missing notebook.py: {notebook_path}")
        if not controls_path.is_file():
            raise WorkspaceValidationError(f"workspace notebook artifact is missing controls.json: {controls_path}")
        try:
            controls = WorkspaceNotebookControls.model_validate(read_json(controls_path))
        except Exception as exc:
            raise WorkspaceValidationError(f"workspace notebook controls are invalid for {notebook_id}: {exc}") from exc
        if controls.workspace_id != context.workspace_id:
            raise WorkspaceValidationError(
                "workspace notebook controls declare a different workspace_id: "
                f"{controls.workspace_id} != {context.workspace_id}"
            )
        if controls.notebook_id != notebook_id:
            raise WorkspaceValidationError(
                f"workspace notebook controls declare notebook_id {controls.notebook_id!r}; expected {notebook_id!r}"
            )
        detail.update(
            {
                "notebook_path": notebook_path.as_posix(),
                "controls_path": controls_path.as_posix(),
                "schema_version": controls.schema_version,
                "default_deliverable": context.require_notebook(notebook_id).default_deliverable,
                "geometries": len(controls.geometry_switchboard.geometries),
            }
        )
        notebook_details.append(detail)
    return notebook_details


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
            view_dir = context.output_root / "views" / view_id
            rows_path = view_dir / "rows.parquet"
            matrix_path = view_dir / "matrix.npy"
            if rows_path.is_file() and matrix_path.is_file():
                materialized_columns = {field.name for field in read_schema(rows_path)}
                required_materialized_columns = {
                    source.record_key,
                    source.subject_key,
                    *(context.config.metadata.include or []),
                    *(
                        _MATERIALIZED_VIEW_METADATA_REQUIRED_COLUMNS
                        if any(
                            isinstance(cohort, PromoterMetadataCohortConfig) and cohort.source == view.source
                            for cohort in context.config.cohorts.values()
                        )
                        else set()
                    ),
                }
                if source.context_key is not None:
                    required_materialized_columns.add(source.context_key)
                missing_materialized_columns = sorted(
                    column
                    for column in required_materialized_columns
                    if column in columns and column not in materialized_columns
                )
                if missing_materialized_columns:
                    raise WorkspaceValidationError(
                        "materialized view rows are missing configured metadata columns: "
                        f"{view_id} ({missing_materialized_columns})"
                    )
                view_detail["materialized"] = True
                view_detail["materialized_row_columns"] = sorted(materialized_columns)
            else:
                view_detail["materialized"] = False
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
        docs_root = resolve_repo_path(context.config.study_binding.docs_root)
        required_files = ["study.yaml"]
        missing = [name for name in required_files if not (docs_root / name).exists()]
        if missing:
            raise WorkspaceValidationError(
                f"study docs_root is missing required files: {docs_root} ({', '.join(sorted(missing))})"
            )
        study_binding = {
            "study_id": context.config.study_binding.study_id,
            "docs_root": docs_root.as_posix(),
            "required_files": required_files,
        }

    notebook_details = _deep_validate_notebook_artifacts(context)

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
        "notebook_details": notebook_details,
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
