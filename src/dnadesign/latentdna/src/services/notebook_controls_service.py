"""
Workspace notebook control-plane assembly for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from os.path import relpath
from pathlib import Path

import numpy as np

from ..contracts.notebook import (
    WorkspaceNotebookCompareMetrics,
    WorkspaceNotebookComparisonBasis,
    WorkspaceNotebookContextAudit,
    WorkspaceNotebookControls,
    WorkspaceNotebookGeometry,
    WorkspaceNotebookLayoutPreset,
    WorkspaceNotebookRuntimePaths,
    WorkspaceNotebookSwitchboardControls,
    WorkspaceNotebookTableRef,
)
from ..io.json_io import read_json
from ..io.parquet_io import read_schema, read_table

_MODEL_PAIR_VIEW_ORDER = [
    "z7_60",
    "z20_60",
    "z7_1k_seq",
    "z20_1k_seq",
    "logits7_60",
    "logits20_60",
    "logits7_1k_seq",
    "logits20_1k_seq",
    "z7_1k_anchor",
    "z20_1k_anchor",
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
    "construct_shift20_norm",
    "construct_self_cosine20",
    "d_spyp_centered",
    "d_sulap_centered",
    "d_soxsp_centered",
    "d_spyp",
    "d_sulap",
    "d_soxsp",
    "d_j23105",
    "ethanol_vs_cipro",
]

_CONTEXT_LABELS = {
    "60bp": "60 bp anchor-only",
    "1kb_anchor": "1 kb anchor-aligned context",
    "1kb_seq": "1 kb expanded-context",
    "1kb_drag": "1 kb context shift",
}


def _format_view_label(*, model: str | None, family: str | None, context_name: str | None) -> str:
    model_text = (model or "unknown").upper()
    family_text = {
        "intermediate": "intermediate",
        "pooled_logits": "pooled logits",
    }.get(str(family or ""), str(family or "unknown").replace("_", " "))
    context_text = _CONTEXT_LABELS.get(str(context_name or ""), str(context_name or "unknown").replace("_", " "))
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
    atlas_2x2_views = ["z7_60", "z20_60", "z7_1k_seq", "z20_1k_seq"]
    if set(atlas_2x2_views).issubset(available):
        presets.append(
            WorkspaceNotebookLayoutPreset(
                id="atlas_2x2_intermediate",
                label="2 x 2 intermediate comparison",
                mode="fixed_grid",
                description="Intermediate anchor-only and 1 kb expanded-context projections for 7B and 20B.",
                view_ids=atlas_2x2_views,
                panel_titles=[
                    "7B anchor-only (60 bp)",
                    "20B anchor-only (60 bp)",
                    "7B expanded-context (1 kb)",
                    "20B expanded-context (1 kb)",
                ],
            )
        )
    atlas_2x3_views = [
        "z7_60",
        "z20_60",
        "z7_1k_seq",
        "z20_1k_seq",
        "logits7_1k_seq",
        "logits20_1k_seq",
    ]
    if set(atlas_2x3_views).issubset(available):
        presets.append(
            WorkspaceNotebookLayoutPreset(
                id="atlas_2x3_model_family",
                label="2 x 3 model comparison",
                mode="fixed_grid",
                description=(
                    "Columns are 7B and 20B; rows are anchor-only intermediate, "
                    "expanded-context intermediate, and expanded-context pooled logits."
                ),
                view_ids=atlas_2x3_views,
                panel_titles=[
                    "7B anchor-only intermediate",
                    "20B anchor-only intermediate",
                    "7B expanded-context intermediate",
                    "20B expanded-context intermediate",
                    "7B expanded-context pooled logits",
                    "20B expanded-context pooled logits",
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
        for raw_label in getattr(reference_set, "ids", []):
            label = str(raw_label).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)

    if labels:
        return labels

    for landmark in context.config.landmarks.values():
        where = getattr(landmark, "where", {}) or {}
        if str(where.get("column") or "") != "usr_label__primary":
            continue
        label = str(where.get("equals") or "").strip()
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
        ("z20_60", "z20_1k_seq"),
        ("z20_1k_anchor", "z20_1k_seq"),
        ("z20_60", "logits20_60"),
        ("z20_1k_seq", "logits20_1k_seq"),
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
    geometry_summary_path = context.output_root / "agreements" / "context_geometry_primary_20b" / "summary.json"
    min_signal_median = 1e-8
    anchor_ll_column = "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token"
    expanded_context_ll_column = "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token"
    payload = WorkspaceNotebookContextAudit(
        artifact_id="context_audit_20b",
        status="missing",
        decision="not_evaluated",
        rule={
            "strategy": "whole_sequence_shift",
            "min_signal_median": min_signal_median,
            "description": (
                "If median construct_shift20_norm is below 1e-8, treat the 1 kb expanded-context lane "
                "as numerically null; otherwise keep the anchor-versus-expanded-context comparison active."
            ),
        },
    )
    if not table_path.is_file():
        return payload
    table = read_table(table_path)
    required_columns = {"construct_shift20_norm", "construct_self_cosine20"}
    missing = required_columns.difference(table.column_names)
    if missing:
        payload.status = "error"
        payload.error = f"context audit table is missing columns: {sorted(missing)}"
        return payload
    shift = np.asarray(table["construct_shift20_norm"].to_pylist(), dtype=np.float64)
    self_cosine = np.asarray(table["construct_self_cosine20"].to_pylist(), dtype=np.float64)
    if shift.size == 0 or self_cosine.size == 0:
        payload.status = "error"
        payload.error = "context audit table is empty"
        return payload
    shift_median = float(np.median(shift))
    self_cosine_median = float(np.median(self_cosine))
    if shift_median < min_signal_median:
        decision = "no_context_signal"
    else:
        decision = "whole_sequence_primary"
    payload.status = "ok"
    payload.decision = decision
    payload.rows = int(table.num_rows)
    payload.table_path = (Path("scalars") / "context_audit_20b" / "table.parquet").as_posix()
    metrics: dict[str, object] = {
        "construct_shift20_norm_median": shift_median,
        "construct_shift20_norm_p95": float(np.percentile(shift, 95.0)),
        "construct_self_cosine20_median": self_cosine_median,
        "construct_self_cosine20_p05": float(np.percentile(self_cosine, 5.0)),
    }
    if anchor_ll_column in table.column_names:
        anchor_ll = np.asarray(table[anchor_ll_column].to_pylist(), dtype=np.float64)
        metrics["anchor20_log_likelihood_per_token_median"] = float(np.median(anchor_ll))
    if expanded_context_ll_column in table.column_names:
        expanded_context_ll = np.asarray(table[expanded_context_ll_column].to_pylist(), dtype=np.float64)
        metrics["expanded_context20_log_likelihood_per_token_median"] = float(np.median(expanded_context_ll))
    if geometry_summary_path.is_file():
        summary = read_json(geometry_summary_path)
        mean_knn_overlap = summary.get("mean_overlap_fraction")
        if isinstance(mean_knn_overlap, int | float):
            metrics["mean_knn_overlap"] = float(mean_knn_overlap)
        landmark_summary = summary.get("landmark_neighbor_overlap")
        if isinstance(landmark_summary, dict):
            mean_jaccard = landmark_summary.get("mean_jaccard_overlap")
            if isinstance(mean_jaccard, int | float):
                metrics["mean_jaccard_overlap"] = float(mean_jaccard)
    payload.metrics = metrics
    return payload


def _runtime_paths(context, *, notebook_id: str) -> WorkspaceNotebookRuntimePaths:
    notebook_dir = context.output_root / "notebooks" / notebook_id

    def relative_to_notebook(target: Path) -> str:
        return Path(relpath(target, start=notebook_dir)).as_posix()

    return WorkspaceNotebookRuntimePaths(
        workspace_relative_path=relative_to_notebook(context.workspace_dir),
        output_relative_path=relative_to_notebook(context.output_root),
        catalog_relative_path=relative_to_notebook(context.output_root / "catalog.json"),
        health_relative_path=relative_to_notebook(context.output_root / "notebooks" / "health.json"),
    )


def build_workspace_notebook_controls_payload(context, *, notebook_id: str) -> WorkspaceNotebookControls:
    projection_ids_by_view = _projection_inventory(context)
    geometries = _geometry_inventory(context, projection_ids_by_view=projection_ids_by_view)
    joinable_tables = _table_inventory(context)
    comparison_bases = _comparison_bases(context, geometries)
    default_compare_left, default_compare_right = _default_compare_views(geometries)
    return WorkspaceNotebookControls(
        schema_version="latentdna.workspace_notebook_controls.v2",
        workspace_id=context.workspace_id,
        notebook_id=notebook_id,
        generated_at=datetime.now(UTC).isoformat(),
        runtime_paths=_runtime_paths(context, notebook_id=notebook_id),
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
            reference_labels=_reference_labels(context),
            compare_metrics=WorkspaceNotebookCompareMetrics(
                sample_rows=192,
                distance_pair_limit=4096,
                knn_k=10,
            ),
        ),
        context_audit=_context_audit_summary(context),
    )
