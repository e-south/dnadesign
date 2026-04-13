"""
Plot recipe resolution for config-backed and inline plot renders.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import TypeAdapter

from ..contracts.errors import ContractViolationError, WorkspaceValidationError
from ..contracts.plot import PlotConfig, ResolvedPlotSpec

_PLOT_CONFIG_ADAPTER = TypeAdapter(PlotConfig)


def _resolved_from_config(plot_id: str, config: PlotConfig) -> ResolvedPlotSpec:
    if config.kind == "projection_scatter":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            projection_ids=[config.projection],
            color_column=config.color_column,
            label_column=config.label_column,
            label_values=list(config.label_values),
            config_id=plot_id,
        )
    if config.kind == "projection_grid":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            projection_ids=list(config.projections),
            color_column=config.color_column,
            label_column=config.label_column,
            label_values=list(config.label_values),
            panel_titles=list(config.panel_titles or []),
            config_id=plot_id,
        )
    if config.kind == "heatmap":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            enrichment_id=config.enrichment,
            value_column=config.value_column,
            config_id=plot_id,
        )
    if config.kind == "distance_scatter":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            distance_id=config.distance,
            x_column=config.x_column,
            y_column=config.y_column,
            color_column=config.color_column,
            config_id=plot_id,
        )
    if config.kind == "xy_scatter":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            scalar_id=config.scalar,
            distance_id=config.distance,
            x_column=config.x_column,
            y_column=config.y_column,
            color_column=config.color_column,
            render_mode=config.render_mode,
            config_id=plot_id,
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
            color_column=config.color_column,
            render_mode=config.render_mode,
            config_id=plot_id,
        )
    if config.kind == "curve":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            reducer_id=config.reducer,
            config_id=plot_id,
        )
    if config.kind == "correspondence_heatmap":
        return ResolvedPlotSpec(
            plot_id=plot_id,
            kind=config.kind,
            left_cluster_id=config.left_cluster,
            right_cluster_id=config.right_cluster,
            config_id=plot_id,
        )
    return ResolvedPlotSpec(
        plot_id=plot_id,
        kind=config.kind,
        agreement_id=config.agreement,
        config_id=plot_id,
    )


def _inline_payload(
    *,
    kind: str,
    projection_ids: list[str],
    panel_titles: list[str],
    enrichment_id: str | None,
    distance_id: str | None,
    scalar_id: str | None,
    agreement_id: str | None,
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
) -> dict[str, object]:
    if kind == "projection_scatter":
        if len(projection_ids) != 1:
            raise ContractViolationError("projection_scatter rendering requires exactly one --projection")
        return {
            "kind": kind,
            "projection": projection_ids[0],
            "color_column": color_column,
            "label_column": label_column,
            "label_values": label_values,
        }
    if kind == "projection_grid":
        return {
            "kind": kind,
            "projections": projection_ids,
            "panel_titles": panel_titles,
            "color_column": color_column,
            "label_column": label_column,
            "label_values": label_values,
        }
    if kind == "heatmap":
        return {
            "kind": kind,
            "enrichment": enrichment_id,
            "value_column": value_column,
        }
    if kind == "distance_scatter":
        return {
            "kind": kind,
            "distance": distance_id,
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
        }
    if kind == "xy_scatter":
        payload = {
            "kind": kind,
            "scalar": scalar_id,
            "distance": distance_id,
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
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
    if kind == "curve":
        return {"kind": kind, "reducer": reducer_id}
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
    agreement_id: str | None,
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
) -> ResolvedPlotSpec:
    has_inline_spec = (
        kind is not None
        or bool(projection_ids)
        or bool(panel_titles)
        or enrichment_id is not None
        or distance_id is not None
        or scalar_id is not None
        or agreement_id is not None
        or reducer_id is not None
        or left_cluster_id is not None
        or right_cluster_id is not None
        or value_column is not None
        or x_column is not None
        or y_column is not None
        or color_column is not None
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
        return _resolved_from_config(plot_id, plots[plot_id])
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
        agreement_id=agreement_id,
        reducer_id=reducer_id,
        left_cluster_id=left_cluster_id,
        right_cluster_id=right_cluster_id,
        value_column=value_column,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
        render_mode=render_mode,
        label_column=label_column,
        label_values=label_values,
    )
    return _resolved_from_config(plot_id, _PLOT_CONFIG_ADAPTER.validate_python(payload))
