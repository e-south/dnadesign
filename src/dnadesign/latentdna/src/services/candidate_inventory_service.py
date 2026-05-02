"""Machine-readable candidate X inventory for workspace status surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from dnadesign.usr import SequencesError

from ..contracts.errors import SourceResolutionError
from ..contracts.workspace import (
    DerivedViewConfig,
    InferFeatureScalarSidecarSourceConfig,
    InferFeatureSidecarSourceConfig,
    MatrixBundleSourceConfig,
    ParquetSourceConfig,
    SourceBackedViewConfig,
    USRSourceConfig,
)
from ..sources.resolver import inspect_source_schema, resolve_source
from ..workspaces.loader import WorkspaceContext
from ._artifacts import artifact_exists
from .candidate_set_service import candidate_set_view_ids
from .freshness_service import FreshnessCache, evaluate_artifact_freshness

_MISSING_SOURCE_MARKERS = ("not found", "not initialized")
_NON_MATERIALIZED_VIEW_ROLES = {"hidden", "planned", "retired"}


def _normalized_role(value: object) -> str:
    return str(value or "").strip().lower()


def _is_missing_source_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if not isinstance(exc, SourceResolutionError | SequencesError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _MISSING_SOURCE_MARKERS)


def _dataset_id(source: object) -> str | None:
    if isinstance(source, USRSourceConfig | InferFeatureSidecarSourceConfig | InferFeatureScalarSidecarSourceConfig):
        return source.dataset
    if isinstance(source, ParquetSourceConfig | MatrixBundleSourceConfig):
        return source.path
    return None


def _source_row_count(context: WorkspaceContext, source_id: str, source: object) -> int | None:
    try:
        resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
        return int(inspect_source_schema(resolved)["row_count"])
    except Exception as exc:
        if _is_missing_source_error(exc):
            return None
        raise


def _view_shape(context: WorkspaceContext, view_id: str) -> tuple[int | None, int | None]:
    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    if not matrix_path.is_file():
        return None, None
    try:
        matrix = np.load(matrix_path, mmap_mode="r")
    except Exception:
        return None, None
    if len(matrix.shape) < 2:
        return int(matrix.shape[0]) if matrix.shape else None, None
    return int(matrix.shape[0]), int(matrix.shape[1])


def _materialization_status(context: WorkspaceContext, view_id: str, *, role: str) -> str:
    if role in _NON_MATERIALIZED_VIEW_ROLES:
        return role
    rows, dims = _view_shape(context, view_id)
    if rows is not None and dims is not None:
        return "materialized"
    return "missing"


def _freshness_status(
    context: WorkspaceContext,
    view_id: str,
    *,
    materialization_status: str,
    cache: FreshnessCache,
) -> str:
    if materialization_status in _NON_MATERIALIZED_VIEW_ROLES:
        return materialization_status
    if materialization_status == "missing" or not artifact_exists(context, artifact_kind="view", artifact_id=view_id):
        return "missing"
    freshness = evaluate_artifact_freshness(
        context,
        artifact_kind="view",
        artifact_id=view_id,
        cache=cache,
    )
    return str(freshness.get("status") or "attention")


def _configured_views(context: WorkspaceContext) -> dict[str, object]:
    return dict(getattr(getattr(context, "config", None), "views", {}) or {})


def _configured_sources(context: WorkspaceContext) -> dict[str, object]:
    return dict(getattr(getattr(context, "config", None), "sources", {}) or {})


def _configured_candidate_sets(context: WorkspaceContext) -> dict[str, object]:
    return dict(getattr(getattr(context, "config", None), "candidate_sets", {}) or {})


def _candidate_set_explicit_view_ids(candidate_set: object) -> list[str]:
    if isinstance(candidate_set, dict):
        raw_view_ids = candidate_set.get("views", [])
    else:
        raw_view_ids = getattr(candidate_set, "views", [])
    return [str(view_id) for view_id in raw_view_ids]


def _candidate_set_view_ids_safe(
    context: WorkspaceContext,
    candidate_set_id: str,
    *,
    views: dict[str, object],
    candidate_sets: dict[str, object],
) -> list[str]:
    if candidate_set_id not in candidate_sets:
        return []
    try:
        return candidate_set_view_ids(context, candidate_set_id)
    except (AttributeError, KeyError, TypeError):
        return [
            view_id
            for view_id in _candidate_set_explicit_view_ids(candidate_sets[candidate_set_id])
            if view_id in views
        ]


def _candidate_set_ids_by_view(
    context: WorkspaceContext,
    *,
    views: dict[str, object],
    candidate_sets: dict[str, object],
) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {view_id: [] for view_id in views}
    for candidate_set_id in candidate_sets:
        for view_id in _candidate_set_view_ids_safe(
            context,
            candidate_set_id,
            views=views,
            candidate_sets=candidate_sets,
        ):
            memberships.setdefault(view_id, []).append(candidate_set_id)
    return memberships


def _ordered_inventory_view_ids(
    context: WorkspaceContext,
    memberships: dict[str, list[str]],
    *,
    views: dict[str, object],
    candidate_sets: dict[str, object],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate_set_id in candidate_sets:
        for view_id in _candidate_set_view_ids_safe(
            context,
            candidate_set_id,
            views=views,
            candidate_sets=candidate_sets,
        ):
            if view_id in seen:
                continue
            seen.add(view_id)
            ordered.append(view_id)
    for view_id in views:
        if view_id in seen:
            continue
        if memberships.get(view_id):
            continue
        seen.add(view_id)
        ordered.append(view_id)
    return ordered


def _model_name(*, tags: dict[str, Any], where: dict[str, Any]) -> str | None:
    raw_model = where.get("model_name") or tags.get("model_name")
    if raw_model:
        return str(raw_model)
    model = str(tags.get("model") or "").strip()
    encoder = str(tags.get("encoder") or "").strip()
    if not model:
        return None
    if model.startswith("evo2_") or not encoder:
        return model
    return f"{encoder}_{model}"


def _feature_family(*, tags: dict[str, Any], where: dict[str, Any], view: object) -> str | None:
    raw_family = where.get("representation_kind") or where.get("scalar_kind") or tags.get("family")
    if raw_family:
        return str(raw_family)
    if isinstance(view, DerivedViewConfig):
        return view.derive.kind
    return None


def _row_basis(source: object | None, *, view: object) -> str | None:
    raw_basis = str(dict(getattr(view, "tags", {}) or {}).get("row_basis") or "").strip()
    if raw_basis:
        return raw_basis
    if source is None:
        return None
    subject_key = str(getattr(source, "subject_key", "") or "").strip()
    context_key = str(getattr(source, "context_key", "") or "").strip()
    if subject_key and context_key:
        return f"{subject_key}+{context_key}"
    return subject_key or None


def _modality(source: object | None, view: object) -> str:
    if isinstance(source, InferFeatureScalarSidecarSourceConfig):
        return "scalar"
    if isinstance(view, SourceBackedViewConfig):
        return "vector"
    return "vector"


def build_candidate_inventory(
    context: WorkspaceContext,
    *,
    freshness_cache: FreshnessCache | None = None,
) -> list[dict[str, object]]:
    """Build one status row per configured candidate representation view."""

    views = _configured_views(context)
    sources = _configured_sources(context)
    candidate_sets = _configured_candidate_sets(context)
    memberships = _candidate_set_ids_by_view(context, views=views, candidate_sets=candidate_sets)
    ordered_view_ids = _ordered_inventory_view_ids(
        context,
        memberships,
        views=views,
        candidate_sets=candidate_sets,
    )
    cache = freshness_cache or FreshnessCache()
    study_binding = getattr(context.config, "study_binding", None)
    study_id = str(getattr(study_binding, "study_id", None) or context.workspace_id)
    rows: list[dict[str, object]] = []
    for view_id in ordered_view_ids:
        view = views[view_id]
        tags = dict(getattr(view, "tags", {}) or {})
        role = _normalized_role(getattr(view, "role", None))
        source_id: str | None = None
        source: object | None = None
        where: dict[str, Any] = {}
        dataset: str | None = None
        source_rows: int | None = None
        n_rows, n_dims = _view_shape(context, view_id)
        if isinstance(view, SourceBackedViewConfig):
            source_id = view.source
            source = sources.get(source_id)
            if source is not None:
                where = dict(getattr(source, "where", None) or {})
                dataset = _dataset_id(source)
            if source is not None and n_rows is None and role not in _NON_MATERIALIZED_VIEW_ROLES:
                source_rows = _source_row_count(context, source_id, source)
        if n_rows is None:
            n_rows = source_rows if role not in _NON_MATERIALIZED_VIEW_ROLES else None
        materialization_status = _materialization_status(context, view_id, role=role)
        rows.append(
            {
                "study_id": study_id,
                "candidate_set_ids": memberships.get(view_id, []),
                "view_id": view_id,
                "source_id": source_id,
                "dataset": dataset,
                "row_basis": _row_basis(source, view=view),
                "model_name": _model_name(tags=tags, where=where),
                "feature_family": _feature_family(tags=tags, where=where, view=view),
                "modality": _modality(source, view),
                "sequence_scope": str(tags.get("scope") or "") or None,
                "pooling_operation": str(where.get("pooling_operation") or tags.get("pooling") or "") or None,
                "orientation": str(where.get("orientation") or tags.get("orientation") or "") or None,
                "coordinate_space_id": str(getattr(view, "coordinate_space_id", "") or "") or None,
                "role": role or None,
                "n_rows": n_rows,
                "n_dims": n_dims,
                "materialization_status": materialization_status,
                "freshness_status": _freshness_status(
                    context,
                    view_id,
                    materialization_status=materialization_status,
                    cache=cache,
                ),
            }
        )
    return rows


__all__ = ["build_candidate_inventory"]
