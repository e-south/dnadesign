"""
Workspace validation services for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dnadesign.usr import SequencesError

from ..contracts.errors import SourceResolutionError, WorkspaceValidationError
from ..contracts.notebook import WorkspaceNotebookControls
from ..contracts.workspace import ColumnCohortConfig, SourceBackedViewConfig
from ..io.json_io import read_json
from ..io.parquet_io import read_schema
from ..sources.resolver import inspect_source_schema, resolve_source
from ..studies.study_binding import (
    REQUIRED_STUDY_DELIVERABLE_DOC_FILES,
    REQUIRED_STUDY_RECORD_FILES,
    missing_required_files,
)
from ..views.row_contracts import source_backed_view_row_contract
from ..workspaces.loader import load_workspace_config
from ..workspaces.paths import resolve_repo_path
from ..workspaces.plot_semantics import validate_plot_semantics_sidecars
from .semantic_validation_service import validate_workspace_sequence_semantics

_NON_MATERIALIZABLE_VIEW_ROLES = {"planned", "retired"}
_OPTIONAL_SOURCE_ROLES = {"planned", "retired"}
_MISSING_SOURCE_MARKERS = ("not found", "not initialized")


def _normalized_role(value: object) -> str:
    return str(value or "").strip().lower()


def _is_optional_missing_source_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if not isinstance(exc, SourceResolutionError | SequencesError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _MISSING_SOURCE_MARKERS)


def _materialized_view_source_contract_state(
    view_dir: Path,
    *,
    source_id: str,
    vector_column: str | None,
) -> str:
    manifest_path = view_dir / "manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    manifest = read_json(manifest_path)
    params = manifest.get("params", {}) if isinstance(manifest, dict) else {}
    if not isinstance(params, dict):
        return "unknown"
    if params.get("source") != source_id:
        return "stale"
    if vector_column is not None and params.get("vector_column") != vector_column:
        return "stale"
    return "current"


def _materialized_view_row_count(rows_path: Path) -> int:
    try:
        return int(pq.read_metadata(rows_path).num_rows)
    except Exception as exc:
        raise WorkspaceValidationError(f"materialized view rows are unreadable: {rows_path}") from exc


def _materialized_view_matrix_shape(matrix_path: Path) -> tuple[int, ...]:
    try:
        matrix = np.load(matrix_path, mmap_mode="r")
        return tuple(int(value) for value in matrix.shape)
    except Exception as exc:
        raise WorkspaceValidationError(f"materialized view matrix is unreadable: {matrix_path}") from exc


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
                "geometries": len(controls.geometry_controls.geometries),
            }
        )
        notebook_details.append(detail)
    return notebook_details


def _source_required_columns(source) -> list[str]:
    required_columns: list[str] = []
    for column in [source.record_key, source.subject_key, source.context_key]:
        if column is not None and column not in required_columns:
            required_columns.append(column)
    return required_columns


def _deep_validate_workspace(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    validate_plot_semantics_sidecars(context)
    source_columns: dict[str, set[str]] = {}
    source_schemas: dict[str, dict[str, object]] = {}
    source_details: list[dict[str, object]] = []
    for source_id in sorted(context.config.sources):
        source = context.require_source(source_id)
        resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
        required_columns = _source_required_columns(source)
        try:
            schema_info = inspect_source_schema(resolved)
        except Exception as exc:
            source_role = _normalized_role(getattr(source, "role", None))
            if source_role not in _OPTIONAL_SOURCE_ROLES or not _is_optional_missing_source_error(exc):
                raise
            source_columns[source_id] = set()
            source_details.append(
                {
                    "source_id": source_id,
                    "kind": source.kind,
                    "path": resolved.records_path.as_posix() if resolved.records_path is not None else None,
                    "row_count": 0,
                    "required_columns": required_columns,
                    "vector_columns": [],
                    "validation_status": f"skipped_{source_role}",
                    "missing_reason": str(exc),
                }
            )
            continue
        columns = set(schema_info["columns"])
        missing_columns = [name for name in required_columns if name not in columns]
        if missing_columns:
            missing_rendered = ", ".join(missing_columns)
            raise WorkspaceValidationError(f"source {source_id} is missing required columns: {missing_rendered}")
        source_columns[source_id] = columns
        source_schemas[source_id] = schema_info
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
        role = str(getattr(view, "role", "") or "").strip().lower()
        if isinstance(view, SourceBackedViewConfig):
            source = context.require_source(view.source)
            columns = source_columns[view.source]
            view_detail = {
                "view_id": view_id,
                "declaration_kind": "source_backed",
                "source": view.source,
                "vector_kind": view.vector.kind,
                "coordinate_space_id": view.coordinate_space_id,
            }
            if role in _NON_MATERIALIZABLE_VIEW_ROLES:
                view_detail["materialized"] = False
                view_detail["validation_status"] = f"skipped_{role}"
                view_details.append(view_detail)
                continue
            if view.vector.kind == "column" and view.vector.name not in columns:
                raise WorkspaceValidationError(
                    f"view {view_id} vector column is missing from source {view.source}: {view.vector.name}"
                )
            if view.vector.kind == "column":
                view_detail["vector_column"] = view.vector.name
            try:
                row_contract = source_backed_view_row_contract(
                    context,
                    source_id=view.source,
                    source=source,
                    available_columns=columns,
                )
            except Exception as exc:
                raise WorkspaceValidationError(f"view {view_id} row-column contract is invalid: {exc}") from exc
            view_dir = context.output_root / "views" / view_id
            rows_path = view_dir / "rows.parquet"
            matrix_path = view_dir / "matrix.npy"
            if rows_path.is_file() and matrix_path.is_file():
                expected_row_count = int(source_schemas[view.source]["row_count"])
                materialized_row_count = _materialized_view_row_count(rows_path)
                materialized_matrix_shape = _materialized_view_matrix_shape(matrix_path)
                view_detail["materialized_row_count"] = materialized_row_count
                view_detail["materialized_matrix_shape"] = list(materialized_matrix_shape)
                if not materialized_matrix_shape:
                    raise WorkspaceValidationError(f"materialized view matrix has no shape: {view_id}")
                if int(materialized_matrix_shape[0]) != materialized_row_count:
                    raise WorkspaceValidationError(
                        "materialized view row table and matrix row counts disagree: "
                        f"{view_id} ({materialized_row_count} rows vs matrix shape {materialized_matrix_shape})"
                    )
                if materialized_row_count != expected_row_count:
                    raise WorkspaceValidationError(
                        "materialized view row count no longer matches source schema: "
                        f"{view_id} ({materialized_row_count} materialized vs {expected_row_count} source rows)"
                    )
                materialized_columns = {field.name for field in read_schema(rows_path)}
                required_materialized_columns = set(row_contract.materialized_row_columns)
                missing_materialized_columns = sorted(
                    column for column in required_materialized_columns if column not in materialized_columns
                )
                if missing_materialized_columns:
                    source_contract_state = _materialized_view_source_contract_state(
                        view_dir,
                        source_id=view.source,
                        vector_column=getattr(view.vector, "name", None),
                    )
                    if source_contract_state == "stale":
                        view_detail["materialized_contract_status"] = "stale_source_contract"
                        view_detail["missing_materialized_row_columns"] = missing_materialized_columns
                        view_detail["materialized_source"] = "stale"
                    elif str(getattr(view, "role", "") or "").strip().lower() != "hidden":
                        raise WorkspaceValidationError(
                            "materialized view rows are missing configured metadata columns: "
                            f"{view_id} ({missing_materialized_columns})"
                        )
                    else:
                        view_detail["materialized_contract_status"] = "skipped_hidden"
                        view_detail["missing_materialized_row_columns"] = missing_materialized_columns
                else:
                    view_detail["materialized_contract_status"] = "ok"
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

    sequence_semantic_details, sequence_semantic_warnings = validate_workspace_sequence_semantics(
        context,
        source_columns=source_columns,
        source_schemas=source_schemas,
    )

    metadata_derivation_details: list[dict[str, object]] = []
    for column_name, derivation in sorted(context.config.metadata.derivations.items()):
        detail: dict[str, object] = {"column": column_name, "kind": derivation.kind}
        if derivation.kind == "lookup":
            source_columns_for_lookup = source_columns.get(derivation.source, set())
            missing = sorted(
                column
                for column in (derivation.right_key, derivation.value_column)
                if column not in source_columns_for_lookup
            )
            if missing:
                raise WorkspaceValidationError(
                    f"metadata derivation {column_name!r} lookup source {derivation.source!r} "
                    f"is missing columns: {missing}"
                )
            detail.update(
                {
                    "source": derivation.source,
                    "left_key": derivation.left_key,
                    "right_key": derivation.right_key,
                    "value_column": derivation.value_column,
                    "missing_policy": derivation.missing_policy,
                }
            )
        elif derivation.kind == "annotation":
            detail.update(
                {
                    "source": derivation.source,
                    "derive": derivation.derive,
                    "handler": derivation.handler,
                    "required_columns": list(derivation.required_columns),
                    "any_required_column_groups": [list(group) for group in derivation.any_required_column_groups],
                    "missing_policy": derivation.missing_policy,
                    "value_type": derivation.value_type,
                }
            )
        metadata_derivation_details.append(detail)

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
        if not isinstance(cohort, ColumnCohortConfig):
            raise WorkspaceValidationError(f"cohort {cohort_id} uses unsupported cohort kind {cohort.kind!r}")
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

    study_binding = None
    if context.config.study_binding is not None:
        record_root = resolve_repo_path(context.config.study_binding.record_root)
        missing_record_files = missing_required_files(record_root, REQUIRED_STUDY_RECORD_FILES)
        if missing_record_files:
            raise WorkspaceValidationError(
                "study record_root is missing required checked-in record files: "
                f"{record_root} ({', '.join(sorted(missing_record_files))})"
            )
        deliverable_docs_root = resolve_repo_path(context.config.study_binding.deliverable_docs_root)
        missing_docs_files = missing_required_files(deliverable_docs_root, REQUIRED_STUDY_DELIVERABLE_DOC_FILES)
        if missing_docs_files:
            raise WorkspaceValidationError(
                "study deliverable_docs_root is missing required files: "
                f"{deliverable_docs_root} ({', '.join(sorted(missing_docs_files))})"
            )
        study_binding = {
            "study_id": context.config.study_binding.study_id,
            "record_root": record_root.as_posix(),
            "deliverable_docs_root": deliverable_docs_root.as_posix(),
            "record_required_files": list(REQUIRED_STUDY_RECORD_FILES),
            "deliverable_docs_required_files": list(REQUIRED_STUDY_DELIVERABLE_DOC_FILES),
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
        "metadata_derivation_details": metadata_derivation_details,
        "sequence_semantic_details": sequence_semantic_details,
        "warnings": sequence_semantic_warnings,
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
    validate_plot_semantics_sidecars(context)
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
