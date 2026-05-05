"""
Geometry control assembly for workspace notebook surfaces.
"""

from __future__ import annotations

import pyarrow.types as pa_types

from ..contracts.notebook import (
    WorkspaceNotebookCompareMetrics,
    WorkspaceNotebookComparisonBasis,
    WorkspaceNotebookGeometry,
    WorkspaceNotebookGeometryControls,
    WorkspaceNotebookLayoutPreset,
    WorkspaceNotebookReferenceSet,
    WorkspaceNotebookTableRef,
)
from ..io.json_io import read_json
from ..io.parquet_io import read_schema
from ..labels import humanize_candidate
from ..visual_style import reference_annotation_label
from .candidate_set_service import build_workspace_candidate_sets, candidate_set_view_ids
from .view_shape_cache import ViewShapeCache

_PREFERRED_HUES = [
    "design_family",
    "design_regulator_composition",
    "sig35_variant",
    "spacer_length",
    "source_class",
    "emitted_length_bp",
    "is_control",
    "synthetic_margin_ethanol_vs_background",
    "synthetic_margin_cipro_vs_background",
    "context_self_cosine",
    "context_shift_l2",
]

_PREFERRED_HUE_KIND_DEFAULTS = {
    "design_family": "categorical",
    "design_regulator_composition": "categorical",
    "sig35_variant": "categorical",
    "spacer_length": "ordinal",
    "source_class": "categorical",
    "emitted_length_bp": "continuous",
    "is_control": "binary",
    "synthetic_margin_ethanol_vs_background": "continuous",
    "synthetic_margin_cipro_vs_background": "continuous",
    "context_self_cosine": "continuous",
    "context_shift_l2": "continuous",
}

_JOINABLE_KEY_COLUMNS = {"construct__anchor_id", "id", "subject_id", "context_id"}
_DEFAULT_JOINABLE_VALUE_COLUMNS = set(_PREFERRED_HUES) | {"cluster_label"}

_FAMILY_LABELS = {
    "intermediate_embedding": "Intermediate block mean",
    "output_layer_mean": "Output-layer mean",
}

_SCOPE_LABELS = {
    "merged_anchor_insert_seq_mean": "Mixed-length anchor-source insert",
    "full_context_1kb": "1 kb construct context",
    "full_context_anchor_mean": "1 kb context anchor mean",
    "reverse_complement_context_anchor_mean": "1 kb reverse-complement context anchor mean",
    "reverse_complement_context_1kb": "1 kb reverse-complement construct context",
    "reference_core60": "reference core60",
    "reference_context_forward_1kb": "reference 1 kb forward context",
    "reference_context_forward_anchor_mean": "reference 1 kb forward anchor mean",
    "reference_context_reverse_complement_1kb": "reference 1 kb reverse-complement context",
    "reference_context_reverse_complement_anchor_mean": "reference 1 kb reverse-complement anchor mean",
    "anchor_plus_anchor_mean_concat": "anchor + anchor-mean concat",
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


def _resolve_notebook(context, notebook_id: str | None):
    if notebook_id is not None:
        return context.require_notebook(notebook_id)
    if "latent_geometry_browser" in context.config.notebooks:
        return context.config.notebooks["latent_geometry_browser"]
    if context.config.notebooks:
        return next(iter(context.config.notebooks.values()))
    return None


def _geometry_order(context, *, notebook_id: str | None) -> list[str]:
    notebook = _resolve_notebook(context, notebook_id)
    configured = list(getattr(notebook, "geometry_order", []) or []) if notebook is not None else []
    if configured:
        return configured
    candidate_set_ids = list(getattr(notebook, "candidate_sets", []) or []) if notebook is not None else []
    default_candidate_set = str(getattr(notebook, "default_candidate_set", "") or "") if notebook is not None else ""
    ordered_candidate_sets = [
        *([default_candidate_set] if default_candidate_set else []),
        *[candidate_set_id for candidate_set_id in candidate_set_ids if candidate_set_id != default_candidate_set],
    ]
    resolved: list[str] = []
    seen: set[str] = set()
    for candidate_set_id in ordered_candidate_sets:
        if candidate_set_id not in context.config.candidate_sets:
            continue
        for view_id in candidate_set_view_ids(context, candidate_set_id):
            if view_id in seen:
                continue
            seen.add(view_id)
            resolved.append(view_id)
    return resolved or list(context.config.views)


def _unique_in_order(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _preferred_hue_order(context, *, notebook_id: str | None) -> list[str]:
    notebook = _resolve_notebook(context, notebook_id)
    configured = list(getattr(notebook, "preferred_hues", []) or []) if notebook is not None else []
    return _unique_in_order([*_PREFERRED_HUES, *configured])


def _preferred_hue_kind_defaults(context, *, notebook_id: str | None) -> dict[str, str]:
    notebook = _resolve_notebook(context, notebook_id)
    configured = dict(getattr(notebook, "preferred_hue_kinds", {}) or {}) if notebook is not None else {}
    return {
        **_PREFERRED_HUE_KIND_DEFAULTS,
        **{str(column): str(kind) for column, kind in configured.items() if str(column).strip()},
    }


def _view_shape(view_id: str, *, shape_cache: ViewShapeCache) -> tuple[int, int] | None:
    rows, dims = shape_cache.get(view_id)
    if rows is None or dims is None:
        return None
    return rows, dims


def _geometry_inventory(
    context,
    *,
    projection_ids_by_view: dict[str, list[str]],
    geometry_order: list[str],
    shape_cache: ViewShapeCache,
) -> list[WorkspaceNotebookGeometry]:
    geometries: list[WorkspaceNotebookGeometry] = []
    order = {view_id: index for index, view_id in enumerate(geometry_order)}
    for view_id, view in sorted(context.config.views.items(), key=lambda item: (order.get(item[0], 999), item[0])):
        tags = dict(getattr(view, "tags", {}) or {})
        role = str(getattr(view, "role", "") or "").strip().lower()
        model = str(tags.get("model") or "")
        family = str(tags.get("family") or "")
        scope_name = str(tags.get("scope") or "")
        if role in {"hidden", "planned", "retired"}:
            continue
        shape = _view_shape(view_id, shape_cache=shape_cache)
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
    joinable_value_columns: set[str],
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
            if not joinable_value_columns.intersection(field_names):
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


def _view_row_schema_inventory(context, *, visible_view_ids: set[str]) -> dict[str, object]:
    schemas: dict[str, object] = {}
    for view_id in sorted(visible_view_ids):
        rows_path = context.output_root / "views" / view_id / "rows.parquet"
        if not rows_path.is_file():
            continue
        try:
            schemas[view_id] = read_schema(rows_path)
        except Exception:
            continue
    return schemas


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
    view_row_schemas_by_id: dict[str, object],
    preferred_hues: list[str],
    default_hue_kinds: dict[str, str],
) -> dict[str, str]:
    kinds: dict[str, str] = {}
    preferred = set(preferred_hues)
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
    for schema in view_row_schemas_by_id.values():
        for field in schema:
            if field.name not in preferred:
                continue
            discovered.add(field.name)
            if field.name in kinds:
                continue
            kind = _infer_hue_kind_from_type(field.type)
            if kind is not None:
                kinds[field.name] = kind
    for column, kind in default_hue_kinds.items():
        if column in discovered:
            kinds[column] = kind
    return {column: kinds[column] for column in preferred_hues if column in kinds}


def _layout_presets(
    context,
    geometry_rows: list[WorkspaceNotebookGeometry],
    *,
    notebook_id: str | None,
    candidate_sets: list | None = None,
) -> list[WorkspaceNotebookLayoutPreset]:
    available = {row.view_id for row in geometry_rows}
    notebook = _resolve_notebook(context, notebook_id)
    candidate_grid_views = list(getattr(notebook, "candidate_grid_views", []) or []) if notebook is not None else []
    candidate_grid_titles = (
        list(getattr(notebook, "candidate_grid_panel_titles", []) or []) if notebook is not None else []
    )
    presets: list[WorkspaceNotebookLayoutPreset] = [
        WorkspaceNotebookLayoutPreset(
            id="single_view",
            label="Single view",
            mode="single_view",
            description="Render one persisted projection with the selected hue.",
        ),
    ]
    if candidate_grid_views and set(candidate_grid_views).issubset(available):
        presets.append(
            WorkspaceNotebookLayoutPreset(
                id="candidate_grid",
                label="Candidate grid",
                mode="fixed_grid",
                description="Projection grid across the surfaced 7B intermediate candidates.",
                view_ids=candidate_grid_views,
                panel_titles=candidate_grid_titles,
            )
        )
    seen_preset_ids = {preset.id for preset in presets}
    for candidate_set in candidate_sets or []:
        candidate_set_id = str(getattr(candidate_set, "candidate_set_id", "") or "")
        preset_id = f"candidate_set__{candidate_set_id}"
        if not candidate_set_id or preset_id in seen_preset_ids:
            continue
        candidate_view_ids = [
            view_id for view_id in getattr(candidate_set, "available_view_ids", []) if str(view_id) in available
        ]
        if len(candidate_view_ids) < 2:
            continue
        presets.append(
            WorkspaceNotebookLayoutPreset(
                id=preset_id,
                label=str(getattr(candidate_set, "label", "") or candidate_set_id),
                mode="fixed_grid",
                description=str(
                    getattr(candidate_set, "description", None)
                    or "Projection grid across a configured candidate representation set."
                ),
                view_ids=[str(view_id) for view_id in candidate_view_ids],
                panel_titles=[str(title) for title in getattr(candidate_set, "panel_titles", [])],
            )
        )
        seen_preset_ids.add(preset_id)
    return presets


def _reference_labels(context) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    for reference_set in context.config.reference_sets.values():
        if getattr(reference_set, "label_mode", None) != "label_and_highlight":
            continue
        display_labels = dict(getattr(reference_set, "display_labels", {}) or {})
        explicit_ids = list(getattr(reference_set, "ids", []) or [])
        selector_ids = [
            str(value)
            for selector in [
                *(getattr(reference_set, "where", []) or []),
                *(getattr(reference_set, "where_all", []) or []),
            ]
            for value in (getattr(selector, "in_values", []) or [])
        ]
        seed_ids = explicit_ids or selector_ids
        if len(seed_ids) > 5:
            continue
        for raw_label in seed_ids:
            raw_text = str(raw_label).strip()
            label = reference_annotation_label(display_labels.get(raw_text, raw_text))
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


def _reference_set_controls(context) -> list[WorkspaceNotebookReferenceSet]:
    controls: list[WorkspaceNotebookReferenceSet] = []
    for reference_set_id, reference_set in context.config.reference_sets.items():
        if not bool(getattr(reference_set, "notebook_exposed", True)):
            continue
        selector_columns = [
            str(getattr(selector, "column"))
            for selector in [
                *(getattr(reference_set, "where", []) or []),
                *(getattr(reference_set, "where_all", []) or []),
            ]
            if str(getattr(selector, "column", "")).strip()
        ]
        controls.append(
            WorkspaceNotebookReferenceSet(
                reference_set_id=reference_set_id,
                label=getattr(reference_set, "label", None),
                match_column=reference_set.match_column,
                label_column=reference_set.label_column,
                label_mode=reference_set.label_mode,
                explicit_ids=[str(value) for value in reference_set.ids],
                selector_columns=list(dict.fromkeys(selector_columns)),
            )
        )
    return controls


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
    context,
    geometry_rows: list[WorkspaceNotebookGeometry],
    *,
    notebook_id: str | None,
) -> tuple[str | None, str | None]:
    view_ids = {row.view_id for row in geometry_rows}
    notebook = _resolve_notebook(context, notebook_id)
    configured_pairs: list[tuple[str, str]] = []
    configured_compare = list(getattr(notebook, "default_compare_views", []) or []) if notebook is not None else []
    if len(configured_compare) == 2:
        configured_pairs.append((configured_compare[0], configured_compare[1]))
    for left_view, right_view in configured_pairs:
        if left_view in view_ids and right_view in view_ids:
            return left_view, right_view
    if len(geometry_rows) >= 2:
        return geometry_rows[0].view_id, geometry_rows[1].view_id
    if geometry_rows:
        only = geometry_rows[0].view_id
        return only, only
    return None, None


def _default_layout_id(
    context,
    presets: list[WorkspaceNotebookLayoutPreset],
    *,
    notebook_id: str | None,
) -> str:
    available = {preset.id for preset in presets}
    notebook = _resolve_notebook(context, notebook_id)
    configured = str(getattr(notebook, "default_layout", "") or "").strip() if notebook is not None else ""
    if configured and configured in available:
        return configured
    default_candidate_set = (
        str(getattr(notebook, "default_candidate_set", "") or "").strip() if notebook is not None else ""
    )
    candidate_preset_id = f"candidate_set__{default_candidate_set}" if default_candidate_set else ""
    if candidate_preset_id in available:
        return candidate_preset_id
    for preset in presets:
        if preset.mode == "fixed_grid" and len(preset.view_ids) > 1:
            return preset.id
    return "single_view"


def build_workspace_geometry_controls(
    context,
    *,
    notebook_id: str | None = None,
    shape_cache: ViewShapeCache | None = None,
) -> WorkspaceNotebookGeometryControls:
    shapes = shape_cache or ViewShapeCache(output_root=context.output_root)
    projection_ids_by_view = _projection_inventory(context)
    geometry_order = _geometry_order(context, notebook_id=notebook_id)
    preferred_hue_order = _preferred_hue_order(context, notebook_id=notebook_id)
    default_hue_kinds = _preferred_hue_kind_defaults(context, notebook_id=notebook_id)
    geometries = _geometry_inventory(
        context,
        projection_ids_by_view=projection_ids_by_view,
        geometry_order=geometry_order,
        shape_cache=shapes,
    )
    visible_view_ids = {row.view_id for row in geometries}
    candidate_sets = build_workspace_candidate_sets(
        context,
        notebook_id=notebook_id,
        visible_view_ids=visible_view_ids,
        shape_cache=shapes,
    )
    joinable_tables, schemas_by_path = _table_inventory(
        context,
        visible_view_ids=visible_view_ids,
        joinable_value_columns=_DEFAULT_JOINABLE_VALUE_COLUMNS | set(preferred_hue_order),
    )
    view_row_schemas_by_id = _view_row_schema_inventory(
        context,
        visible_view_ids=visible_view_ids,
    )
    hue_kinds = _preferred_hue_kinds(
        joinable_tables,
        schemas_by_path=schemas_by_path,
        view_row_schemas_by_id=view_row_schemas_by_id,
        preferred_hues=preferred_hue_order,
        default_hue_kinds=default_hue_kinds,
    )
    preferred_hues = [column for column in preferred_hue_order if column in hue_kinds]
    view_row_columns = {field.name for schema in view_row_schemas_by_id.values() for field in schema}
    row_metadata_hues = [column for column in preferred_hue_order if column in hue_kinds and column in view_row_columns]
    comparison_bases = _comparison_bases(context, geometries)
    default_compare_left, default_compare_right = _default_compare_views(
        context,
        geometries,
        notebook_id=notebook_id,
    )
    layout_presets = _layout_presets(
        context,
        geometries,
        notebook_id=notebook_id,
        candidate_sets=candidate_sets,
    )
    default_geometry = geometries[0] if geometries else None
    notebook = _resolve_notebook(context, notebook_id)
    default_reference_set = str(getattr(notebook, "default_reference_set", "") or "") if notebook is not None else ""
    return WorkspaceNotebookGeometryControls(
        default_model=default_geometry.model if default_geometry is not None else "7b",
        default_family=default_geometry.family if default_geometry is not None else "intermediate_embedding",
        default_context=default_geometry.context if default_geometry is not None else "merged_anchor_insert_seq_mean",
        default_layout=_default_layout_id(context, layout_presets, notebook_id=notebook_id),
        default_reference_set=default_reference_set,
        default_compare_left=default_compare_left,
        default_compare_right=default_compare_right,
        geometries=geometries,
        preferred_hues=preferred_hues,
        row_metadata_hues=row_metadata_hues,
        hue_kinds=hue_kinds,
        joinable_tables=joinable_tables,
        layout_presets=layout_presets,
        comparison_bases=comparison_bases,
        reference_labels=_reference_labels(context),
        reference_sets=_reference_set_controls(context),
        candidate_sets=candidate_sets,
        compare_metrics=WorkspaceNotebookCompareMetrics(
            sample_rows=192,
            distance_pair_limit=4096,
            knn_k=10,
        ),
    )


__all__ = ["build_workspace_geometry_controls"]
