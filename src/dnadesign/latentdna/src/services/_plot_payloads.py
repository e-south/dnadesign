"""
Payload and manifest helpers for plot services.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.errors import ContractViolationError
from ..contracts.manifest import ArtifactInput
from ..contracts.plot import ResolvedPlotSpec
from ..workspaces.loader import WorkspaceContext
from ._artifact_inputs import dependency_artifact_input


def _artifact_input(context: WorkspaceContext, kind: str, artifact_id: str, path: Path) -> ArtifactInput:
    return dependency_artifact_input(context, kind=kind, artifact_id=artifact_id, path=path)


def plot_artifact_inputs(context: WorkspaceContext, spec: ResolvedPlotSpec) -> list[ArtifactInput]:
    if spec.kind in {"projection_scatter", "projection_grid"}:
        return [
            _artifact_input(
                context,
                "projection",
                projection_id,
                context.output_root / "projections" / projection_id / "coords.parquet",
            )
            for projection_id in spec.projection_ids
        ]
    if spec.kind == "heatmap":
        if spec.enrichment_id is not None:
            return [
                _artifact_input(
                    context,
                    "enrichment_set",
                    spec.enrichment_id,
                    context.output_root / "enrichments" / spec.enrichment_id / "table.parquet",
                )
            ]
    if spec.kind == "heatmap_grid":
        return [
            _artifact_input(
                context,
                "scalar_table",
                scalar_id,
                context.output_root / "scalars" / scalar_id / "table.parquet",
            )
            for scalar_id in spec.scalar_ids
        ]
        assert spec.scalar_id is not None
        return [
            _artifact_input(
                context,
                "scalar_table",
                spec.scalar_id,
                context.output_root / "scalars" / spec.scalar_id / "table.parquet",
            )
        ]
    if spec.kind == "distance_scatter":
        assert spec.distance_id is not None
        return [
            _artifact_input(
                context,
                "distance_set",
                spec.distance_id,
                context.output_root / "distances" / spec.distance_id / "table.parquet",
            )
        ]
    if spec.kind == "xy_scatter":
        if spec.distance_id is not None:
            return [
                _artifact_input(
                    context,
                    "distance_set",
                    spec.distance_id,
                    context.output_root / "distances" / spec.distance_id / "table.parquet",
                )
            ]
        assert spec.scalar_id is not None
        return [
            _artifact_input(
                context,
                "scalar_table",
                spec.scalar_id,
                context.output_root / "scalars" / spec.scalar_id / "table.parquet",
            )
        ]
    if spec.kind in {"xy_scatter_grid", "paired_xy_scatter_grid"}:
        return [
            _artifact_input(
                context,
                "scalar_table",
                scalar_id,
                context.output_root / "scalars" / scalar_id / "table.parquet",
            )
            for scalar_id in spec.scalar_ids
        ]
    if spec.kind in {"categorical_count", "metric_panel_grid"}:
        assert spec.scalar_id is not None
        return [
            _artifact_input(
                context,
                "scalar_table",
                spec.scalar_id,
                context.output_root / "scalars" / spec.scalar_id / "table.parquet",
            )
        ]
    if spec.kind == "distribution":
        table_inputs = [
            (
                "scalar_table",
                spec.scalar_id,
                context.output_root / "scalars" / spec.scalar_id / "table.parquet"
                if spec.scalar_id is not None
                else None,
            ),
            (
                "distance_set",
                spec.distance_id,
                context.output_root / "distances" / spec.distance_id / "table.parquet"
                if spec.distance_id is not None
                else None,
            ),
            (
                "enrichment_set",
                spec.enrichment_id,
                context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
                if spec.enrichment_id is not None
                else None,
            ),
            (
                "agreement_set",
                spec.agreement_id,
                context.output_root / "agreements" / spec.agreement_id / "table.parquet"
                if spec.agreement_id is not None
                else None,
            ),
        ]
        selected = [
            _artifact_input(context, input_kind, str(artifact_id), input_path)
            for input_kind, artifact_id, input_path in table_inputs
            if artifact_id is not None and input_path is not None
        ]
        if len(selected) != 1:
            raise ContractViolationError("distribution rendering requires exactly one table-backed artifact input")
        return selected
    if spec.kind == "distribution_grid":
        return [
            _artifact_input(
                context,
                "scalar_table",
                scalar_id,
                context.output_root / "scalars" / scalar_id / "table.parquet",
            )
            for scalar_id in spec.scalar_ids
        ]
    if spec.kind == "curve":
        assert spec.reducer_id is not None
        return [
            _artifact_input(
                context,
                "reducer",
                spec.reducer_id,
                context.output_root / "reducers" / spec.reducer_id / "summary.json",
            )
        ]
    if spec.kind == "curve_grid":
        return [
            _artifact_input(
                context,
                "reducer",
                reducer_id,
                context.output_root / "reducers" / reducer_id / "summary.json",
            )
            for reducer_id in spec.reducer_ids
        ]
    if spec.kind == "correspondence_heatmap":
        assert spec.left_cluster_id is not None and spec.right_cluster_id is not None
        return [
            _artifact_input(
                context,
                "cluster_set",
                spec.left_cluster_id,
                context.output_root / "clusters" / spec.left_cluster_id / "assignments.parquet",
            ),
            _artifact_input(
                context,
                "cluster_set",
                spec.right_cluster_id,
                context.output_root / "clusters" / spec.right_cluster_id / "assignments.parquet",
            ),
        ]
    if spec.kind == "agreement_summary_grid":
        return [
            _artifact_input(
                context,
                "agreement_set",
                agreement_id,
                context.output_root / "agreements" / agreement_id / "summary.json",
            )
            for agreement_id in spec.agreement_ids
        ]
    assert spec.agreement_id is not None
    return [
        _artifact_input(
            context,
            "agreement_set",
            spec.agreement_id,
            context.output_root / "agreements" / spec.agreement_id / "summary.json",
        )
    ]


def plot_input_payload(spec: ResolvedPlotSpec) -> dict[str, object]:
    payload: dict[str, object] = {"kind": spec.kind}
    if spec.projection_ids:
        payload["projections"] = spec.projection_ids
    if spec.panel_titles:
        payload["panel_titles"] = spec.panel_titles
    if spec.enrichment_id is not None:
        payload["enrichment"] = spec.enrichment_id
    if spec.distance_id is not None:
        payload["distance"] = spec.distance_id
    if spec.scalar_id is not None:
        payload["scalar"] = spec.scalar_id
    if spec.scalar_ids:
        payload["scalars"] = spec.scalar_ids
    if spec.reducer_ids:
        payload["reducers"] = spec.reducer_ids
    if spec.panel_column is not None:
        payload["panel_column"] = spec.panel_column
    if spec.agreement_id is not None:
        payload["agreement"] = spec.agreement_id
    if spec.agreement_ids:
        payload["agreements"] = spec.agreement_ids
    if spec.reducer_id is not None:
        payload["reducer"] = spec.reducer_id
    if spec.left_cluster_id is not None:
        payload["left_cluster"] = spec.left_cluster_id
    if spec.right_cluster_id is not None:
        payload["right_cluster"] = spec.right_cluster_id
    if spec.render_mode is not None:
        payload["render_mode"] = spec.render_mode
    if spec.pair_id_column is not None:
        payload["pair_id_column"] = spec.pair_id_column
    if spec.shape_column is not None:
        payload["shape_column"] = spec.shape_column
    if spec.size_column is not None:
        payload["size_column"] = spec.size_column
    if spec.size_range is not None:
        payload["size_range"] = list(spec.size_range)
    if spec.default_hue is not None:
        payload["default_hue"] = spec.default_hue
    if spec.hue_options:
        payload["hue_options"] = [option.model_dump(mode="json") for option in spec.hue_options]
    if spec.x_axis_label is not None:
        payload["x_axis_label"] = spec.x_axis_label
    if spec.y_axis_label is not None:
        payload["y_axis_label"] = spec.y_axis_label
    if spec.colorbar_label is not None:
        payload["colorbar_label"] = spec.colorbar_label
    if spec.direction_column is not None:
        payload["direction_column"] = spec.direction_column
    if spec.unit_column is not None:
        payload["unit_column"] = spec.unit_column
    if spec.reference_line is not None:
        payload["reference_line"] = spec.reference_line
    if spec.row_column is not None:
        payload["row_column"] = spec.row_column
    if spec.column_column is not None:
        payload["column_column"] = spec.column_column
    if spec.row_order:
        payload["row_order"] = spec.row_order
    if spec.column_order:
        payload["column_order"] = spec.column_order
    if spec.label_column is not None:
        payload["label_column"] = spec.label_column
    if spec.label_values:
        payload["label_values"] = spec.label_values
    if spec.config_id is not None:
        payload["plot_recipe"] = spec.config_id
    return payload


def manifest_params_for_plot(spec: ResolvedPlotSpec) -> dict[str, object]:
    params: dict[str, object] = {"plot_kind": spec.kind}
    if spec.projection_ids:
        params["projection_ids"] = spec.projection_ids
    if spec.enrichment_id is not None:
        params["enrichment_id"] = spec.enrichment_id
    if spec.distance_id is not None:
        params["distance_id"] = spec.distance_id
    if spec.scalar_id is not None:
        params["scalar_id"] = spec.scalar_id
    if spec.scalar_ids:
        params["scalar_ids"] = spec.scalar_ids
    if spec.reducer_ids:
        params["reducer_ids"] = spec.reducer_ids
    if spec.agreement_id is not None:
        params["agreement_id"] = spec.agreement_id
    if spec.agreement_ids:
        params["agreement_ids"] = spec.agreement_ids
    if spec.reducer_id is not None:
        params["reducer_id"] = spec.reducer_id
    if spec.left_cluster_id is not None:
        params["left_cluster_id"] = spec.left_cluster_id
    if spec.right_cluster_id is not None:
        params["right_cluster_id"] = spec.right_cluster_id
    if spec.value_column is not None:
        params["value_column"] = spec.value_column
    if spec.value_columns:
        params["value_columns"] = spec.value_columns
    if spec.metric_columns:
        params["metric_columns"] = spec.metric_columns
    if spec.row_column is not None:
        params["row_column"] = spec.row_column
    if spec.column_column is not None:
        params["column_column"] = spec.column_column
    if spec.row_order:
        params["row_order"] = spec.row_order
    if spec.column_order:
        params["column_order"] = spec.column_order
    if spec.panel_column is not None:
        params["panel_column"] = spec.panel_column
    if spec.x_column is not None:
        params["x_column"] = spec.x_column
    if spec.y_column is not None:
        params["y_column"] = spec.y_column
    if spec.color_column is not None:
        params["color_column"] = spec.color_column
    if spec.color_scale is not None:
        params["color_scale"] = spec.color_scale
    if spec.shape_column is not None:
        params["shape_column"] = spec.shape_column
    if spec.size_column is not None:
        params["size_column"] = spec.size_column
    if spec.size_range is not None:
        params["size_range"] = list(spec.size_range)
    if spec.default_hue is not None:
        params["default_hue"] = spec.default_hue
    if spec.hue_options:
        params["hue_options"] = [option.model_dump(mode="json") for option in spec.hue_options]
    if spec.x_axis_label is not None:
        params["x_axis_label"] = spec.x_axis_label
    if spec.y_axis_label is not None:
        params["y_axis_label"] = spec.y_axis_label
    if spec.colorbar_label is not None:
        params["colorbar_label"] = spec.colorbar_label
    if spec.direction_column is not None:
        params["direction_column"] = spec.direction_column
    if spec.unit_column is not None:
        params["unit_column"] = spec.unit_column
    if spec.reference_line is not None:
        params["reference_line"] = spec.reference_line
    if spec.pair_id_column is not None:
        params["pair_id_column"] = spec.pair_id_column
    if spec.render_mode is not None:
        params["render_mode"] = spec.render_mode
    if spec.label_column is not None:
        params["label_column"] = spec.label_column
    if spec.label_values:
        params["label_values"] = spec.label_values
    if spec.panel_titles:
        params["panel_titles"] = spec.panel_titles
    if spec.measure_kind is not None:
        params["measure_kind"] = spec.measure_kind
    if spec.value_kind is not None:
        params["value_kind"] = spec.value_kind
    if spec.value_label is not None:
        params["value_label"] = spec.value_label
    if spec.sort_rule is not None:
        params["sort_rule"] = spec.sort_rule
    if spec.annotation is not None:
        params["annotation"] = spec.annotation.model_dump(mode="json")
    if spec.single_row_panels:
        params["single_row_panels"] = True
    if spec.square_panels:
        params["square_panels"] = True
    if spec.hide_repeated_y_axis:
        params["hide_repeated_y_axis"] = True
    if spec.kind in {"categorical_count", "metric_panel_grid", "distribution"}:
        if spec.scalar_id is not None:
            params["input_kind"] = "scalar_table"
            params["input_id"] = spec.scalar_id
        elif spec.distance_id is not None:
            params["input_kind"] = "distance_set"
            params["input_id"] = spec.distance_id
        elif spec.enrichment_id is not None:
            params["input_kind"] = "enrichment_set"
            params["input_id"] = spec.enrichment_id
        elif spec.agreement_id is not None:
            params["input_kind"] = "agreement_set"
            params["input_id"] = spec.agreement_id
    if spec.kind == "heatmap":
        if spec.enrichment_id is not None:
            params["input_kind"] = "enrichment_set"
            params["input_id"] = spec.enrichment_id
        elif spec.scalar_id is not None:
            params["input_kind"] = "scalar_table"
            params["input_id"] = spec.scalar_id
    if spec.kind == "heatmap_grid":
        params["input_kind"] = "scalar_table"
        params["input_ids"] = spec.scalar_ids
    if spec.config_id is not None:
        params["plot_config_id"] = spec.config_id
    return params


__all__ = ["manifest_params_for_plot", "plot_artifact_inputs", "plot_input_payload"]
