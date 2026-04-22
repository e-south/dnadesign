"""
Geometry control assembly for workspace notebook surfaces.
"""

from __future__ import annotations

import numpy as np
import pyarrow.types as pa_types

from ..contracts.notebook import (
    WorkspaceNotebookCompareMetrics,
    WorkspaceNotebookComparisonBasis,
    WorkspaceNotebookGeometry,
    WorkspaceNotebookGeometryControls,
    WorkspaceNotebookLayoutPreset,
    WorkspaceNotebookTableRef,
)
from ..io.json_io import read_json
from ..io.parquet_io import read_schema
from ..labels import humanize_candidate

_CANONICAL_GEOMETRY_ORDER = [
    "intermediate_embedding_7b_anchor_60bp",
    "pooled_logits_7b_anchor_60bp",
    "intermediate_embedding_7b_full_context_1kb",
    "pooled_logits_7b_full_context_1kb",
    "intermediate_embedding_7b_full_context_anchor_mean",
    "intermediate_embedding_7b_anchor_plus_full_context_concat",
    "intermediate_embedding_7b_anchor_plus_anchor_mean_concat",
]

_PREFERRED_HUES = [
    "design_family",
    "design_regulator_composition",
    "sig35_variant",
    "spacer_length",
    "source_class",
    "is_control",
    "synthetic_margin_ethanol_vs_background",
    "synthetic_margin_cipro_vs_background",
    "log_likelihood_per_token_7b",
    "context_self_cosine",
    "context_shift_l2",
]

_PREFERRED_HUE_KIND_DEFAULTS = {
    "design_family": "categorical",
    "design_regulator_composition": "categorical",
    "sig35_variant": "categorical",
    "spacer_length": "ordinal",
    "source_class": "categorical",
    "is_control": "binary",
    "synthetic_margin_ethanol_vs_background": "continuous",
    "synthetic_margin_cipro_vs_background": "continuous",
    "log_likelihood_per_token_7b": "continuous",
    "context_self_cosine": "continuous",
    "context_shift_l2": "continuous",
}

_JOINABLE_KEY_COLUMNS = {"construct__anchor_id", "id", "subject_id", "context_id"}
_JOINABLE_VALUE_COLUMNS = set(_PREFERRED_HUES) | {"cluster_label"}

_FAMILY_LABELS = {
    "intermediate_embedding": "Intermediate block mean",
    "pooled_logits": "Pooled logits",
}

_SCOPE_LABELS = {
    "anchor_60bp": "60 bp anchor",
    "full_context_1kb": "1 kb construct context",
}


def _format_view_label(*, model: str | None, family: str | None, scope_name: str | None) -> str:
    return humanize_candidate(
        {
            "candidate_model": f"evo2_{model}" if model else "",
            "candidate_scope": scope_name or "",
            "candidate_family": family or "",
        }
    )


def _projection_inventory(context) -> dict[str, list[str]]:
    projection_root = context.output_root / "projections"
    inventory: dict[str, list[tuple[tuple[int, int, int, str], str]]] = {}
    if not projection_root.is_dir():
        return inventory
    for projection_dir in sorted(path for path in projection_root.iterdir() if path.is_dir()):
        manifest_path = projection_dir / "manifest.json"
        coords_path = projection_dir / "coords.parquet"
        if not manifest_path.is_file() or not coords_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if not _manifest_is_current(
            manifest,
            artifact_id=projection_dir.name,
            artifact_kind="projection",
            allowed_statuses={"ok", "attention"},
        ):
            continue
        view_id = next(
            (
                str(item.get("id"))
                for item in manifest.get("inputs", [])
                if isinstance(item, dict) and item.get("kind") == "view_matrix" and item.get("id")
            ),
            None,
        )
        if view_id is None:
            continue
        stats = manifest.get("stats", {}) if isinstance(manifest.get("stats"), dict) else {}
        params = manifest.get("params", {}) if isinstance(manifest.get("params"), dict) else {}
        projected_rows = int(stats.get("projected_rows", stats.get("rows", 0)) or 0)
        population_rows = int(stats.get("population_rows", projected_rows) or projected_rows)
        is_full_population = bool(stats.get("is_full_population", projected_rows == population_rows))
        projection_role = str(params.get("projection_role") or "").strip().lower()
        if projection_role not in {"primary", "appendix", "audit", "experimental"}:
            if projection_dir.name.startswith("audit_"):
                projection_role = "audit"
            elif projection_dir.name.startswith("appendix_") or projection_dir.name.startswith("umap_"):
                projection_role = "appendix"
            else:
                projection_role = "primary" if is_full_population else "appendix"
        default_rank = int(params.get("default_rank", 0 if is_full_population else 100))
        sort_key = (
            {"primary": 0, "appendix": 1, "audit": 2, "experimental": 3}[projection_role],
            0 if is_full_population else 1,
            default_rank,
            projection_dir.name,
        )
        inventory.setdefault(view_id, []).append((sort_key, projection_dir.name))
    return {
        view_id: [projection_id for _, projection_id in sorted(entries, key=lambda item: item[0])]
        for view_id, entries in sorted(inventory.items())
    }


def _view_shape(context, view_id: str) -> tuple[int, int] | None:
    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    if not matrix_path.is_file():
        return None
    matrix = np.load(matrix_path, mmap_mode="r")
    return int(matrix.shape[0]), int(matrix.shape[1])


def _geometry_inventory(context, *, projection_ids_by_view: dict[str, list[str]]) -> list[WorkspaceNotebookGeometry]:
    geometries: list[WorkspaceNotebookGeometry] = []
    order = {view_id: index for index, view_id in enumerate(_CANONICAL_GEOMETRY_ORDER)}
    for view_id, view in sorted(context.config.views.items(), key=lambda item: (order.get(item[0], 999), item[0])):
        tags = dict(getattr(view, "tags", {}) or {})
        role = str(getattr(view, "role", "") or "").strip().lower()
        model = str(tags.get("model") or "")
        family = str(tags.get("family") or "")
        scope_name = str(tags.get("scope") or "")
        if role == "hidden":
            continue
        shape = _view_shape(context, view_id)
        view_dir = context.output_root / "views" / view_id
        geometries.append(
            WorkspaceNotebookGeometry(
                view_id=view_id,
                label=_format_view_label(model=model, family=family, scope_name=scope_name),
                model=model.lower(),
                family=family,
                context=scope_name,
                role=getattr(view, "role", None),
                materialized=(view_dir / "matrix.npy").is_file() and (view_dir / "rows.parquet").is_file(),
                projection_ids=projection_ids_by_view.get(view_id, []),
                coordinate_space_id=getattr(view, "coordinate_space_id", None),
                rows=None if shape is None else shape[0],
                dims=None if shape is None else shape[1],
            )
        )
    return geometries


def _manifest_view_ids(
    manifest: dict[str, object],
    *,
    input_kinds: set[str] | None = None,
) -> set[str]:
    return {
        str(item.get("id"))
        for item in manifest.get("inputs", [])
        if isinstance(item, dict)
        and item.get("kind") in (input_kinds or {"view_matrix", "view_rows"})
        and str(item.get("id") or "").strip()
    }


def _manifest_is_current(
    manifest: dict[str, object],
    *,
    artifact_id: str,
    artifact_kind: str | None = None,
    allowed_statuses: set[str] | None = None,
) -> bool:
    if not manifest:
        return False
    manifest_artifact_id = str(manifest.get("artifact_id") or "").strip()
    if not manifest_artifact_id or manifest_artifact_id != artifact_id:
        return False
    manifest_kind = str(manifest.get("artifact_kind") or "").strip()
    if artifact_kind and (not manifest_kind or manifest_kind != artifact_kind):
        return False
    status = str(manifest.get("status") or "").strip().lower()
    if not status:
        return False
    valid_statuses = {item.strip().lower() for item in (allowed_statuses or {"ok"}) if str(item).strip()}
    if status not in valid_statuses:
        return False
    return not bool(manifest.get("stale"))


def _table_targets_visible_views(
    *,
    artifact_id: str,
    manifest: dict[str, object],
    visible_view_ids: set[str],
) -> bool:
    manifest_view_ids = _manifest_view_ids(manifest)
    del artifact_id
    return bool(manifest_view_ids.intersection(visible_view_ids))


def _table_inventory(
    context,
    *,
    visible_view_ids: set[str],
) -> tuple[list[WorkspaceNotebookTableRef], dict[str, object]]:
    inventory: list[WorkspaceNotebookTableRef] = []
    schemas: dict[str, object] = {}
    table_roots = [
        ("scalar", "scalars", "table.parquet", "scalar_table"),
        ("distance", "distances", "table.parquet", "distance_set"),
        ("cluster", "clusters", "assignments.parquet", "cluster_set"),
    ]
    for kind, root_name, filename, artifact_kind in table_roots:
        root = context.output_root / root_name
        if not root.is_dir():
            continue
        for artifact_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            table_path = artifact_dir / filename
            manifest_path = artifact_dir / "manifest.json"
            if not table_path.is_file():
                continue
            manifest: dict[str, object] = {}
            if manifest_path.is_file():
                try:
                    manifest = read_json(manifest_path)
                except Exception:
                    manifest = {}
            if not _manifest_is_current(manifest, artifact_id=artifact_dir.name, artifact_kind=artifact_kind):
                continue
            if not _table_targets_visible_views(
                artifact_id=artifact_dir.name,
                manifest=manifest,
                visible_view_ids=visible_view_ids,
            ):
                continue
            relative_path = table_path.relative_to(context.output_root).as_posix()
            schema = read_schema(table_path)
            field_names = [field.name for field in schema]
            if not _JOINABLE_KEY_COLUMNS.intersection(field_names):
                continue
            if not _JOINABLE_VALUE_COLUMNS.intersection(field_names):
                continue
            schemas[relative_path] = schema
            row_scope_view_ids = _manifest_view_ids(manifest, input_kinds={"view_rows"})
            table_view_ids = row_scope_view_ids or _manifest_view_ids(manifest)
            inventory.append(
                WorkspaceNotebookTableRef(
                    kind=kind,
                    artifact_id=artifact_dir.name,
                    relative_path=relative_path,
                    columns=field_names,
                    manifest_path=(
                        manifest_path.relative_to(context.output_root).as_posix() if manifest_path.is_file() else None
                    ),
                    view_ids=sorted(table_view_ids),
                )
            )
    return inventory, schemas


def _infer_hue_kind_from_type(field_type) -> str | None:
    if pa_types.is_boolean(field_type):
        return "binary"
    if pa_types.is_string(field_type) or pa_types.is_large_string(field_type) or pa_types.is_dictionary(field_type):
        return "categorical"
    if (
        pa_types.is_integer(field_type)
        or pa_types.is_unsigned_integer(field_type)
        or pa_types.is_floating(field_type)
        or pa_types.is_decimal(field_type)
    ):
        return "continuous"
    return None


def _preferred_hue_kinds(
    joinable_tables: list[WorkspaceNotebookTableRef],
    *,
    schemas_by_path: dict[str, object],
) -> dict[str, str]:
    kinds: dict[str, str] = {}
    preferred = set(_PREFERRED_HUES)
    discovered: set[str] = set()
    for table_ref in joinable_tables:
        schema = schemas_by_path[table_ref.relative_path]
        for field in schema:
            if field.name not in preferred:
                continue
            discovered.add(field.name)
            if field.name in kinds:
                continue
            kind = _infer_hue_kind_from_type(field.type)
            if kind is not None:
                kinds[field.name] = kind
    for column, kind in _PREFERRED_HUE_KIND_DEFAULTS.items():
        if column in discovered:
            kinds[column] = kind
    return {column: kinds[column] for column in _PREFERRED_HUES if column in kinds}


def _layout_presets(geometry_rows: list[WorkspaceNotebookGeometry]) -> list[WorkspaceNotebookLayoutPreset]:
    available = {row.view_id for row in geometry_rows}
    presets: list[WorkspaceNotebookLayoutPreset] = [
        WorkspaceNotebookLayoutPreset(
            id="single_view",
            label="Single view",
            mode="single_view",
            description="Render one persisted projection with the selected hue.",
        ),
    ]
    if set(_CANONICAL_GEOMETRY_ORDER).issubset(available):
        presets.append(
            WorkspaceNotebookLayoutPreset(
                id="candidate_grid",
                label="Candidate grid",
                mode="fixed_grid",
                description="Projection grid across the surfaced 7B intermediate candidates and concat experiments.",
                view_ids=_CANONICAL_GEOMETRY_ORDER,
                panel_titles=[
                    "Evo 2 7B · 60 bp anchor · Intermediate block mean",
                    "Evo 2 7B · 60 bp anchor · Pooled logits",
                    "Evo 2 7B · 1 kb construct context · Intermediate block mean",
                    "Evo 2 7B · 1 kb construct context · Pooled logits",
                    "Evo 2 7B · 1 kb context anchor mean · Intermediate block mean",
                    "Evo 2 7B · Anchor + 1 kb context concat · Intermediate block mean",
                    "Evo 2 7B · Anchor + anchor-mean concat · Intermediate block mean",
                ],
            )
        )
    return presets


def _reference_labels(context) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    for reference_set in context.config.reference_sets.values():
        if getattr(reference_set, "label_mode", None) != "label_and_highlight":
            continue
        display_labels = dict(getattr(reference_set, "display_labels", {}) or {})
        for raw_label in getattr(reference_set, "ids", []):
            raw_text = str(raw_label).strip()
            label = str(display_labels.get(raw_text, raw_text)).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)

    if labels:
        return labels

    for landmark_id, landmark in context.config.landmarks.items():
        where = getattr(landmark, "where", {}) or {}
        if str(where.get("column") or "") != "usr_label__primary":
            continue
        label = str(landmark_id).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _comparison_bases(
    context, geometry_rows: list[WorkspaceNotebookGeometry]
) -> list[WorkspaceNotebookComparisonBasis]:
    geometry_view_ids = {row.view_id for row in geometry_rows}
    bases: list[WorkspaceNotebookComparisonBasis] = []
    for alignment_id, alignment in sorted(context.config.alignments.items()):
        if alignment.left not in geometry_view_ids or alignment.right not in geometry_view_ids:
            continue
        bases.append(
            WorkspaceNotebookComparisonBasis(
                id=alignment_id,
                kind="alignment",
                alignment_id=alignment_id,
                left_view=alignment.left,
                right_view=alignment.right,
                support=alignment.support,
                left_aggregation=alignment.left_aggregation,
                right_aggregation=alignment.right_aggregation,
                label=f"{alignment.left} ↔ {alignment.right} ({alignment_id})",
            )
        )
    return bases


def _default_compare_views(
    geometry_rows: list[WorkspaceNotebookGeometry],
) -> tuple[str | None, str | None]:
    view_ids = {row.view_id for row in geometry_rows}
    for left_view, right_view in [
        ("intermediate_embedding_7b_anchor_60bp", "intermediate_embedding_7b_full_context_anchor_mean"),
        ("intermediate_embedding_7b_anchor_60bp", "intermediate_embedding_7b_full_context_1kb"),
    ]:
        if left_view in view_ids and right_view in view_ids:
            return left_view, right_view
    if len(geometry_rows) >= 2:
        return geometry_rows[0].view_id, geometry_rows[1].view_id
    if geometry_rows:
        only = geometry_rows[0].view_id
        return only, only
    return None, None


def build_workspace_geometry_controls(context) -> WorkspaceNotebookGeometryControls:
    projection_ids_by_view = _projection_inventory(context)
    geometries = _geometry_inventory(context, projection_ids_by_view=projection_ids_by_view)
    joinable_tables, schemas_by_path = _table_inventory(
        context,
        visible_view_ids={row.view_id for row in geometries},
    )
    hue_kinds = _preferred_hue_kinds(joinable_tables, schemas_by_path=schemas_by_path)
    preferred_hues = [column for column in _PREFERRED_HUES if column in hue_kinds]
    comparison_bases = _comparison_bases(context, geometries)
    default_compare_left, default_compare_right = _default_compare_views(geometries)
    return WorkspaceNotebookGeometryControls(
        default_model="7b",
        default_family="intermediate_embedding",
        default_context="anchor_60bp",
        default_layout="single_view",
        default_compare_left=default_compare_left,
        default_compare_right=default_compare_right,
        geometries=geometries,
        preferred_hues=preferred_hues,
        hue_kinds=hue_kinds,
        joinable_tables=joinable_tables,
        layout_presets=_layout_presets(geometries),
        comparison_bases=comparison_bases,
        reference_labels=_reference_labels(context),
        compare_metrics=WorkspaceNotebookCompareMetrics(
            sample_rows=192,
            distance_pair_limit=4096,
            knn_k=10,
        ),
    )


__all__ = ["build_workspace_geometry_controls"]
