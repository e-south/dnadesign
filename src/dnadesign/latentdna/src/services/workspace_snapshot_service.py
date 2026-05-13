"""
Study-facing workspace snapshot service for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dnadesign.usr import SequencesError

from ..contracts.deliverable import DeliverableEntryStatus
from ..contracts.errors import SourceResolutionError
from ..contracts.workspace import (
    InferFeatureScalarSidecarSourceConfig,
    InferFeatureSidecarSourceConfig,
    MatrixBundleSourceConfig,
    ParquetSourceConfig,
    USRSourceConfig,
)
from ..contracts.workspace_snapshot import (
    WorkspaceSnapshot,
    WorkspaceSnapshotBrowser,
    WorkspaceSnapshotDeliverable,
    WorkspaceSnapshotExport,
    WorkspaceSnapshotSource,
)
from ..io.json_io import write_json
from ..services.candidate_inventory_service import build_candidate_inventory
from ..services.deliverable_service import deliverable_status_from_context
from ..services.freshness_service import FreshnessCache
from ..services.notebook_geometry_controls import build_workspace_geometry_controls
from ..sources.resolver import inspect_source_schema, resolve_source
from ..workspaces.loader import load_workspace_config

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


def _source_snapshot(context) -> dict[str, WorkspaceSnapshotSource]:
    snapshots: dict[str, WorkspaceSnapshotSource] = {}
    for source_id, source in context.config.sources.items():
        resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
        try:
            schema = inspect_source_schema(resolved)
        except Exception as exc:
            if _normalized_role(getattr(source, "role", None)) in _OPTIONAL_SOURCE_ROLES and (
                _is_optional_missing_source_error(exc)
            ):
                continue
            raise
        dataset_id: str | None = None
        if isinstance(source, USRSourceConfig):
            dataset_id = source.dataset
        elif isinstance(source, ParquetSourceConfig):
            dataset_id = source.path
        elif isinstance(source, MatrixBundleSourceConfig):
            dataset_id = source.path
        elif isinstance(source, InferFeatureSidecarSourceConfig | InferFeatureScalarSidecarSourceConfig):
            dataset_id = source.dataset
        snapshots[source_id] = WorkspaceSnapshotSource(
            kind=source.kind,
            path=str(schema["path"]),
            dataset_id=dataset_id,
            row_count=int(schema["row_count"]),
            columns=[str(name) for name in schema.get("columns", [])],
            vector_columns=[str(name) for name in schema.get("vector_columns", [])],
        )
    return snapshots


def _model_families(context) -> list[str]:
    families: set[str] = set()
    for view in context.config.views.values():
        tags = dict(getattr(view, "tags", {}) or {})
        encoder = str(tags.get("encoder") or "").strip().lower()
        model = str(tags.get("model") or "").strip().lower()
        if encoder and model:
            families.add(model if model.startswith(f"{encoder}_") else f"{encoder}_{model}")
        elif model:
            families.add(model)
    return sorted(families)


def _browser_snapshot(context) -> WorkspaceSnapshotBrowser:
    geometry_controls = build_workspace_geometry_controls(context)
    geometry_ids = [row.view_id for row in geometry_controls.geometries]
    candidate_sets = {
        row.candidate_set_id: {
            "label": row.label,
            "view_ids": list(row.view_ids),
            "available_view_ids": list(row.available_view_ids),
        }
        for row in geometry_controls.candidate_sets
    }
    return WorkspaceSnapshotBrowser(
        default_geometry_ids=geometry_ids,
        preferred_hues=list(geometry_controls.preferred_hues),
        candidate_sets=candidate_sets,
    )


def _freshness_from_outputs(outputs: list[DeliverableEntryStatus]) -> str:
    statuses = {entry.status for entry in outputs}
    if "error" in statuses:
        return "error"
    if "attention" in statuses:
        return "attention"
    if "missing" in statuses:
        return "missing"
    return "ok"


def _deliverable_snapshots(context) -> dict[str, WorkspaceSnapshotDeliverable]:
    snapshots: dict[str, WorkspaceSnapshotDeliverable] = {}
    freshness_cache = FreshnessCache()
    for deliverable_id in context.config.deliverables:
        status = deliverable_status_from_context(context, deliverable_id, freshness_cache=freshness_cache)
        snapshots[deliverable_id] = WorkspaceSnapshotDeliverable(
            title=status.title,
            status=status.status,
            freshness=_freshness_from_outputs(status.outputs),
            acceptance_checks=list(status.acceptance_checks),
            artifact_paths=[entry.path for entry in status.outputs if entry.path is not None],
            docs_refs=list(status.docs_refs),
            warnings=list(status.warnings),
        )
    return snapshots


def _decision_ladder(context) -> list[str]:
    ladder: list[str] = []
    for deliverable_id, deliverable in context.config.deliverables.items():
        section = str(getattr(deliverable, "section", "") or "").strip().lower()
        if section == "gate":
            continue
        plot_ids = [str(plot_id) for plot_id in deliverable.outputs.get("plots", [])]
        if plot_ids:
            visibility_tiers = {
                str(getattr(context.config.plots[plot_id], "visibility_tier", "primary") or "primary")
                for plot_id in plot_ids
            }
            if visibility_tiers.isdisjoint({"primary"}):
                continue
        ladder.append(deliverable_id)
    return ladder


def _export_snapshots(context) -> dict[str, WorkspaceSnapshotExport]:
    snapshots: dict[str, WorkspaceSnapshotExport] = {}
    for export_id in context.config.exports:
        export_dir = context.output_root / "exports" / export_id
        manifest_path = export_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = context.read_manifest(manifest_path)
            snapshots[export_id] = WorkspaceSnapshotExport(
                status=str(manifest.get("status", "ok")),
                artifact_path=export_dir.as_posix(),
                manifest_path=manifest_path.as_posix(),
                warnings=[str(item) for item in manifest.get("warnings", [])],
                params=dict(manifest.get("params", {})),
            )
            continue
        snapshots[export_id] = WorkspaceSnapshotExport(
            status="missing",
            artifact_path=export_dir.as_posix(),
            manifest_path=manifest_path.as_posix(),
        )
    return snapshots


def workspace_snapshot(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace, validate_plot_semantics=False)
    payload = WorkspaceSnapshot(
        schema_version="latentdna.workspace_snapshot.v1",
        workspace_id=context.workspace_id,
        output_root=context.output_root.as_posix(),
        sources=_source_snapshot(context),
        model_families=_model_families(context),
        canonical_views=list(context.config.views),
        candidate_inventory=build_candidate_inventory(context),
        deliverables=_deliverable_snapshots(context),
        exports=_export_snapshots(context),
        browser=_browser_snapshot(context),
        decision_ladder=_decision_ladder(context),
        last_updated_at=datetime.now(UTC).isoformat(),
    ).model_dump(mode="json")
    write_json(context.output_root / "status" / "workspace_snapshot.json", payload)
    return payload
