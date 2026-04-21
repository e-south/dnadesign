"""
Plot recipe resolution for config-backed and inline plot renders.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import TypeAdapter

from ..contracts.errors import ContractViolationError, WorkspaceValidationError
from ..contracts.plot import PlotConfig, ResolvedPlotSpec

_PLOT_CONFIG_ADAPTER = TypeAdapter(PlotConfig)


def _resolved_color_column(config: PlotConfig) -> str | None:
    return getattr(config, "color_column", None) or getattr(config, "default_hue", None)


def _resolved_hue_fields(config: PlotConfig) -> dict[str, object]:
    return {
        "default_hue": getattr(config, "default_hue", None),
        "hue_options": list(getattr(config, "hue_options", []) or []),
    }


def _resolved_from_config(plot_id: str, config: PlotConfig, *, config_id: str | None) -> ResolvedPlotSpec:
    if config.kind == "projection_scatter":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            projection_ids=[config.projection],
            color_column=_resolved_color_column(config),
            shape_column=config.shape_column,
            label_column=config.label_column,
            label_values=list(config.label_values),
            annotation=config.annotation,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "projection_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            projection_ids=list(config.projections),
            color_column=_resolved_color_column(config),
            shape_column=config.shape_column,
            label_column=config.label_column,
            label_values=list(config.label_values),
            panel_titles=list(config.panel_titles or []),
            annotation=config.annotation,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "heatmap":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            enrichment_id=config.enrichment,
            scalar_id=config.scalar,
            row_column=config.row_column,
            column_column=config.column_column,
            value_column=config.value_column,
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "distance_scatter":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            distance_id=config.distance,
            x_column=config.x_column,
            y_column=config.y_column,
            color_column=_resolved_color_column(config),
            shape_column=config.shape_column,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "xy_scatter":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_id=config.scalar,
            distance_id=config.distance,
            x_column=config.x_column,
            y_column=config.y_column,
            color_column=_resolved_color_column(config),
            shape_column=config.shape_column,
            render_mode=config.render_mode,
            annotation=config.annotation,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "xy_scatter_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_ids=list(config.scalars),
            x_column=config.x_column,
            y_column=config.y_column,
            color_column=_resolved_color_column(config),
            shape_column=config.shape_column,
            render_mode=config.render_mode,
            panel_titles=list(config.panel_titles or []),
            annotation=config.annotation,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "paired_xy_scatter_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_ids=list(config.scalars),
            x_column=config.x_column,
            y_column=config.y_column,
            color_column=_resolved_color_column(config),
            shape_column=config.shape_column,
            pair_id_column=config.pair_id_column,
            render_mode=config.render_mode,
            panel_titles=list(config.panel_titles or []),
            annotation=config.annotation,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "categorical_count":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_id=config.scalar,
            row_column=config.category_column,
            column_column=config.label_column,
            value_column=config.value_column,
            panel_column=config.panel_column,
            color_column=_resolved_color_column(config),
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "metric_panel_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_id=config.scalar,
            row_column=config.facet_column,
            panel_column=config.panel_title_column,
            column_column=config.category_column,
            label_column=config.label_column,
            value_column=config.value_column,
            ci_lower_column=config.ci_lower_column,
            ci_upper_column=config.ci_upper_column,
            color_column=_resolved_color_column(config),
            measure_kind=config.measure_kind,
            value_kind=config.value_kind,
            value_label=config.value_label,
            sort_rule=config.sort_rule,
            direction_column=config.direction_column,
            unit_column=config.unit_column,
            reference_line=config.reference_line,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "distribution":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_id=config.scalar,
            distance_id=config.distance,
            enrichment_id=config.enrichment,
            agreement_id=config.agreement,
            value_column=config.value_column,
            color_column=_resolved_color_column(config),
            render_mode=config.render_mode,
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "distribution_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_ids=list(config.scalars),
            metric_columns=list(config.metric_columns or []),
            value_columns=list(config.value_columns or []),
            color_column=_resolved_color_column(config),
            render_mode=config.render_mode,
            panel_titles=list(config.panel_titles or []),
            **_resolved_hue_fields(config),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "curve":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            reducer_id=config.reducer,
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "curve_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            reducer_ids=list(config.reducers),
            panel_titles=list(config.panel_titles or []),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "correspondence_heatmap":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            left_cluster_id=config.left_cluster,
            right_cluster_id=config.right_cluster,
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    if config.kind == "agreement_summary_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            agreement_ids=list(config.agreements),
            panel_titles=list(config.panel_titles or []),
            config_id=config_id,
            semantics_ref=config.semantics_ref,
        )
    return ResolvedPlotSpec(
        plot_id=plot_id,
        kind=config.kind,
        agreement_id=config.agreement,
        config_id=config_id,
        semantics_ref=config.semantics_ref,
    )


def _inline_payload(
    *,
    kind: str,
    projection_ids: list[str],
    panel_titles: list[str],
    enrichment_id: str | None,
    distance_id: str | None,
    scalar_id: str | None,
    scalar_ids: list[str],
    agreement_id: str | None,
    agreement_ids: list[str],
    reducer_id: str | None,
    left_cluster_id: str | None,
    right_cluster_id: str | None,
    value_column: str | None,
    x_column: str | None,
    y_column: str | None,
    color_column: str | None,
    shape_column: str | None,
    render_mode: str | None,
    label_column: str | None,
    label_values: list[str],
) -> dict[str, object]:
    if kind == "projection_scatter":
        if len(projection_ids) != 1:
            raise ContractViolationError("projection_scatter rendering requires exactly one --projection")
        return {
            "kind": kind,
            "projection": projection_ids[0],
            "color_column": color_column,
            "shape_column": shape_column,
            "label_column": label_column,
            "label_values": label_values,
            "annotation": None,
        }
    if kind == "projection_grid":
        return {
            "kind": kind,
            "projections": projection_ids,
            "panel_titles": panel_titles,
            "color_column": color_column,
            "shape_column": shape_column,
            "label_column": label_column,
            "label_values": label_values,
            "annotation": None,
        }
    if kind == "heatmap":
        return {
            "kind": kind,
            "enrichment": enrichment_id,
            "scalar": scalar_id,
            "row_column": None,
            "column_column": None,
            "value_column": value_column,
        }
    if kind == "distance_scatter":
        return {
            "kind": kind,
            "distance": distance_id,
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
            "shape_column": shape_column,
        }
    if kind == "xy_scatter":
        payload = {
            "kind": kind,
            "scalar": scalar_id,
            "distance": distance_id,
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
            "shape_column": shape_column,
        }
        if render_mode is not None:
            payload["render_mode"] = render_mode
        return payload
    if kind == "xy_scatter_grid":
        payload = {
            "kind": kind,
            "scalars": scalar_ids,
            "panel_titles": panel_titles,
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
            "shape_column": shape_column,
        }
        if render_mode is not None:
            payload["render_mode"] = render_mode
        return payload
    if kind == "paired_xy_scatter_grid":
        payload = {
            "kind": kind,
            "scalars": scalar_ids,
            "panel_titles": panel_titles,
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
            "shape_column": shape_column,
        }
        if render_mode is not None:
            payload["render_mode"] = render_mode
        return payload
    if kind == "distribution":
        payload = {
            "kind": kind,
            "scalar": scalar_id,
            "distance": distance_id,
            "enrichment": enrichment_id,
            "agreement": agreement_id,
            "value_column": value_column,
            "color_column": color_column,
        }
        if render_mode is not None:
            payload["render_mode"] = render_mode
        return payload
    if kind == "distribution_grid":
        payload = {
            "kind": kind,
            "scalars": scalar_ids,
            "panel_titles": panel_titles,
            "color_column": color_column,
        }
        if x_column is not None:
            payload["metric_columns"] = [x_column]
        if value_column is not None:
            payload["value_columns"] = [value_column] * len(scalar_ids)
        if render_mode is not None:
            payload["render_mode"] = render_mode
        return payload
    if kind == "metric_panel_grid":
        if shape_column is not None:
            raise ContractViolationError("metric_panel_grid does not support shape_column")
        payload = {
            "kind": kind,
            "scalar": scalar_id,
            "facet_column": x_column,
            "panel_title_column": y_column,
            "category_column": color_column,
            "label_column": label_column,
            "value_column": value_column,
            "value_kind": "score",
            "value_label": value_column or "value",
        }
        return payload
    if kind == "curve":
        return {"kind": kind, "reducer": reducer_id}
    if kind == "curve_grid":
        raise ContractViolationError("curve_grid inline rendering is not supported; declare it under plots:")
    if kind == "correspondence_heatmap":
        return {
            "kind": kind,
            "left_cluster": left_cluster_id,
            "right_cluster": right_cluster_id,
        }
    if kind == "agreement_summary":
        return {
            "kind": kind,
            "agreement": agreement_id,
        }
    if kind == "agreement_summary_grid":
        return {
            "kind": kind,
            "agreements": agreement_ids,
            "panel_titles": panel_titles,
        }
    raise ContractViolationError(f"unsupported plot kind: {kind}")


def resolve_plot_spec(
    *,
    plots: Mapping[str, PlotConfig],
    plot_id: str,
    kind: str | None,
    projection_ids: list[str],
    panel_titles: list[str],
    enrichment_id: str | None,
    distance_id: str | None,
    scalar_id: str | None,
    scalar_ids: list[str],
    agreement_id: str | None,
    agreement_ids: list[str],
    reducer_id: str | None,
    left_cluster_id: str | None,
    right_cluster_id: str | None,
    value_column: str | None,
    x_column: str | None,
    y_column: str | None,
    color_column: str | None,
    render_mode: str | None,
    label_column: str | None,
    label_values: list[str],
    shape_column: str | None = None,
) -> ResolvedPlotSpec:
    has_inline_spec = (
        kind is not None
        or bool(projection_ids)
        or bool(panel_titles)
        or enrichment_id is not None
        or distance_id is not None
        or scalar_id is not None
        or bool(scalar_ids)
        or agreement_id is not None
        or bool(agreement_ids)
        or reducer_id is not None
        or left_cluster_id is not None
        or right_cluster_id is not None
        or value_column is not None
        or x_column is not None
        or y_column is not None
        or color_column is not None
        or shape_column is not None
        or render_mode is not None
        or label_column is not None
        or bool(label_values)
    )
    has_config_spec = plot_id in plots
    if has_inline_spec and has_config_spec:
        raise ContractViolationError(
            "plot render accepts either a named workspace plot recipe or inline plot flags, not both"
        )
    if has_config_spec:
        return _resolved_from_config(plot_id, plots[plot_id], config_id=plot_id)
    if kind is None:
        raise WorkspaceValidationError(
            f"unknown plot recipe: {plot_id}. Declare it under plots: or provide inline --kind and artifact flags"
        )

    payload = _inline_payload(
        kind=kind,
        projection_ids=projection_ids,
        panel_titles=panel_titles,
        enrichment_id=enrichment_id,
        distance_id=distance_id,
        scalar_id=scalar_id,
        scalar_ids=scalar_ids,
        agreement_id=agreement_id,
        agreement_ids=agreement_ids,
        reducer_id=reducer_id,
        left_cluster_id=left_cluster_id,
        right_cluster_id=right_cluster_id,
        value_column=value_column,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
        shape_column=shape_column,
        render_mode=render_mode,
        label_column=label_column,
        label_values=label_values,
    )
    return _resolved_from_config(plot_id, _PLOT_CONFIG_ADAPTER.validate_python(payload), config_id=None)
