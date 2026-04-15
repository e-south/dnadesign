"""
Workspace notebook control-plane assembly for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..contracts.notebook import (
    WorkspaceNotebookCompareMetrics,
    WorkspaceNotebookComparisonBasis,
    WorkspaceNotebookContextAudit,
    WorkspaceNotebookControls,
    WorkspaceNotebookGeometry,
    WorkspaceNotebookLayoutPreset,
    WorkspaceNotebookSwitchboardControls,
    WorkspaceNotebookTableRef,
)
from ..io.json_io import read_json
from ..io.parquet_io import read_schema, read_table

_MODEL_PAIR_VIEW_ORDER = [
    "z7_60",
    "z20_60",
    "z7_1k_anchor",
    "z20_1k_anchor",
    "z7_1k_seq",
    "z20_1k_seq",
    "logits7_60",
    "logits20_60",
    "logits7_1k_anchor",
    "logits20_1k_anchor",
]

_PREFERRED_HUES = [
    "design_family",
    "design_regulator_composition",
    "sigma70_variant",
    "campaign_prior",
    "is_control",
    "source_class",
    "densegen__plan",
    "usr_label__primary",
    "template_id",
    "construct__template_id",
    "delta20_norm",
    "drag20_norm",
    "d_spyp",
    "d_sulap",
    "d_soxsp",
    "d_j23105",
    "ethanol_vs_cipro",
    "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token",
    "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token",
    "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token",
    "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token",
]

_REFERENCE_LABELS = ["spyP", "sulAp", "soxSp", "J23105"]


def _format_view_label(*, model: str | None, family: str | None, context_name: str | None) -> str:
    model_text = (model or "unknown").upper()
    family_text = {
        "intermediate": "intermediate",
        "pooled_logits": "pooled logits",
    }.get(str(family or ""), str(family or "unknown").replace("_", " "))
    context_text = {
        "60bp": "60 bp",
        "1kb_anchor": "1 kb anchor",
        "1kb_seq": "1 kb seq",
        "1kb_drag": "1 kb drag",
    }.get(str(context_name or ""), str(context_name or "unknown").replace("_", " "))
    return f"{model_text} {family_text} {context_text}"


def _projection_inventory(context) -> dict[str, list[str]]:
    projection_root = context.output_root / "projections"
    inventory: dict[str, list[str]] = {}
    if not projection_root.is_dir():
        return inventory
    for projection_dir in sorted(path for path in projection_root.iterdir() if path.is_dir()):
        manifest_path = projection_dir / "manifest.json"
        coords_path = projection_dir / "coords.parquet"
        if not manifest_path.is_file() or not coords_path.is_file():
            continue
        manifest = read_json(manifest_path)
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
        inventory.setdefault(view_id, []).append(projection_dir.name)
    return {view_id: projection_ids for view_id, projection_ids in sorted(inventory.items())}


def _view_shape(context, view_id: str) -> tuple[int, int] | None:
    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    if not matrix_path.is_file():
        return None
    matrix = np.load(matrix_path, mmap_mode="r")
    return int(matrix.shape[0]), int(matrix.shape[1])


def _geometry_inventory(context, *, projection_ids_by_view: dict[str, list[str]]) -> list[WorkspaceNotebookGeometry]:
    geometries: list[WorkspaceNotebookGeometry] = []
    for view_id, view in sorted(context.config.views.items()):
        tags = dict(getattr(view, "tags", {}) or {})
        model = str(tags.get("model") or "")
        family = str(tags.get("family") or "")
        context_name = str(tags.get("context") or "")
        if model.lower() not in {"20b", "7b"}:
            continue
        if family not in {"intermediate", "pooled_logits"}:
            continue
        shape = _view_shape(context, view_id)
        view_dir = context.output_root / "views" / view_id
        geometries.append(
            WorkspaceNotebookGeometry(
                view_id=view_id,
                label=_format_view_label(model=model, family=family, context_name=context_name),
                model=model.lower(),
                family=family,
                context=context_name,
                role=getattr(view, "role", None),
                materialized=(view_dir / "matrix.npy").is_file() and (view_dir / "rows.parquet").is_file(),
                projection_ids=projection_ids_by_view.get(view_id, []),
                coordinate_space_id=getattr(view, "coordinate_space_id", None),
                rows=None if shape is None else shape[0],
                dims=None if shape is None else shape[1],
            )
        )
    return geometries


def _table_inventory(context) -> list[WorkspaceNotebookTableRef]:
    inventory: list[WorkspaceNotebookTableRef] = []
    table_roots = [
        ("scalar", "scalars", "table.parquet"),
        ("distance", "distances", "table.parquet"),
        ("cluster", "clusters", "assignments.parquet"),
    ]
    for kind, root_name, filename in table_roots:
        root = context.output_root / root_name
        if not root.is_dir():
            continue
        for artifact_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            table_path = artifact_dir / filename
            manifest_path = artifact_dir / "manifest.json"
            if not table_path.is_file():
                continue
            schema = read_schema(table_path)
            inventory.append(
                WorkspaceNotebookTableRef(
                    kind=kind,
                    artifact_id=artifact_dir.name,
                    relative_path=table_path.relative_to(context.output_root).as_posix(),
                    columns=[field.name for field in schema],
                    manifest_path=(
                        manifest_path.relative_to(context.output_root).as_posix() if manifest_path.is_file() else None
                    ),
                )
            )
    return inventory


def _layout_presets(geometry_rows: list[WorkspaceNotebookGeometry]) -> list[WorkspaceNotebookLayoutPreset]:
    available = {row.view_id for row in geometry_rows}
    presets: list[WorkspaceNotebookLayoutPreset] = [
        WorkspaceNotebookLayoutPreset(
            id="single_view",
            label="Single view",
            mode="single_view",
            description="Render one persisted projection with the selected hue.",
        ),
        WorkspaceNotebookLayoutPreset(
            id="model_pair",
            label="Side-by-side pair",
            mode="model_pair",
            description="Render the selected context/family as a 7B versus 20B pair when both views exist.",
            view_order=_MODEL_PAIR_VIEW_ORDER,
        ),
    ]
    atlas_2x2_views = ["z7_60", "z20_60", "z7_1k_anchor", "z20_1k_anchor"]
    if set(atlas_2x2_views).issubset(available):
        presets.append(
            WorkspaceNotebookLayoutPreset(
                id="atlas_2x2_intermediate",
                label="2 x 2 intermediate atlas",
                mode="fixed_grid",
                description="Intermediate anchor-only and anchor-aware projections for 7B and 20B.",
                view_ids=atlas_2x2_views,
                panel_titles=[
                    "7B anchor-only (60 bp)",
                    "20B anchor-only (60 bp)",
                    "7B context-aware (1 kb anchor)",
                    "20B context-aware (1 kb anchor)",
                ],
            )
        )
    atlas_2x3_views = [
        "z7_60",
        "z20_60",
        "z7_1k_anchor",
        "z20_1k_anchor",
        "logits7_1k_anchor",
        "logits20_1k_anchor",
    ]
    if set(atlas_2x3_views).issubset(available):
        presets.append(
            WorkspaceNotebookLayoutPreset(
                id="atlas_2x3_model_family",
                label="2 x 3 model-by-family atlas",
                mode="fixed_grid",
                description=(
                    "Columns are 7B and 20B; rows are anchor-only intermediate, "
                    "context-aware intermediate, and pooled logits."
                ),
                view_ids=atlas_2x3_views,
                panel_titles=[
                    "7B anchor-only intermediate",
                    "20B anchor-only intermediate",
                    "7B anchor-aware intermediate",
                    "20B anchor-aware intermediate",
                    "7B anchor-aware pooled logits",
                    "20B anchor-aware pooled logits",
                ],
            )
        )
    return presets


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
        ("z20_60", "z20_1k_anchor"),
        ("z20_1k_anchor", "z20_1k_seq"),
        ("z20_60", "logits20_60"),
        ("z20_1k_anchor", "logits20_1k_anchor"),
    ]:
        if left_view in view_ids and right_view in view_ids:
            return left_view, right_view
    if len(geometry_rows) >= 2:
        return geometry_rows[0].view_id, geometry_rows[1].view_id
    if geometry_rows:
        only = geometry_rows[0].view_id
        return only, only
    return None, None


def _context_audit_summary(context) -> WorkspaceNotebookContextAudit:
    table_path = context.output_root / "scalars" / "context_audit_20b" / "table.parquet"
    min_signal_median = 1e-8
    payload = WorkspaceNotebookContextAudit(
        artifact_id="context_audit_20b",
        status="missing",
        decision="not_evaluated",
        rule={
            "strategy": "median_ratio",
            "demote_threshold_ratio": 0.1,
            "min_signal_median": min_signal_median,
            "description": (
                "If both medians are below 1e-8, treat the context lane as numerically null; "
                "otherwise demote delta20 when median(delta20_norm) / median(drag20_norm) < 0.1 "
                "and drag20 is present."
            ),
        },
    )
    if not table_path.is_file():
        return payload
    table = read_table(table_path)
    required_columns = {"delta20_norm", "drag20_norm"}
    missing = required_columns.difference(table.column_names)
    if missing:
        payload.status = "error"
        payload.error = f"context audit table is missing columns: {sorted(missing)}"
        return payload
    delta = np.asarray(table["delta20_norm"].to_pylist(), dtype=np.float64)
    drag = np.asarray(table["drag20_norm"].to_pylist(), dtype=np.float64)
    if delta.size == 0 or drag.size == 0:
        payload.status = "error"
        payload.error = "context audit table is empty"
        return payload
    delta_median = float(np.median(delta))
    drag_median = float(np.median(drag))
    ratio = None if drag_median == 0.0 else float(delta_median / drag_median)
    if max(delta_median, drag_median) < min_signal_median:
        decision = "no_context_signal"
    elif drag_median > 0.0 and ratio is not None and ratio < 0.1:
        decision = "demote_delta_in_x2"
    else:
        decision = "retain_delta_candidate"
    payload.status = "ok"
    payload.decision = decision
    payload.rows = int(table.num_rows)
    payload.table_path = (Path("scalars") / "context_audit_20b" / "table.parquet").as_posix()
    payload.metrics = {
        "delta20_median": delta_median,
        "delta20_p95": float(np.percentile(delta, 95.0)),
        "drag20_median": drag_median,
        "drag20_p95": float(np.percentile(drag, 95.0)),
        "delta20_to_drag20_median_ratio": ratio,
    }
    return payload


def build_workspace_notebook_controls_payload(context, *, notebook_id: str) -> WorkspaceNotebookControls:
    projection_ids_by_view = _projection_inventory(context)
    geometries = _geometry_inventory(context, projection_ids_by_view=projection_ids_by_view)
    joinable_tables = _table_inventory(context)
    comparison_bases = _comparison_bases(context, geometries)
    default_compare_left, default_compare_right = _default_compare_views(geometries)
    return WorkspaceNotebookControls(
        schema_version="latentdna.workspace_notebook_controls.v1",
        workspace_id=context.workspace_id,
        notebook_id=notebook_id,
        generated_at=datetime.now(UTC).isoformat(),
        geometry_switchboard=WorkspaceNotebookSwitchboardControls(
            default_model="20b",
            default_family="intermediate",
            default_context="60bp",
            default_layout="single_view",
            default_compare_left=default_compare_left,
            default_compare_right=default_compare_right,
            geometries=geometries,
            preferred_hues=_PREFERRED_HUES,
            joinable_tables=joinable_tables,
            layout_presets=_layout_presets(geometries),
            comparison_bases=comparison_bases,
            reference_labels=_REFERENCE_LABELS,
            compare_metrics=WorkspaceNotebookCompareMetrics(
                sample_rows=192,
                distance_pair_limit=4096,
                knn_k=10,
            ),
        ),
        context_audit=_context_audit_summary(context),
    )
