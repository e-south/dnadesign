"""
Plot recipe contracts for latentdna.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1)]

SUPPORTED_PLOT_KINDS: frozenset[str] = frozenset(
    {
        "projection_scatter",
        "projection_grid",
        "heatmap",
        "distance_scatter",
        "xy_scatter",
        "xy_scatter_grid",
        "paired_xy_scatter_grid",
        "categorical_count",
        "metric_panel_grid",
        "distribution",
        "distribution_grid",
        "curve",
        "curve_grid",
        "correspondence_heatmap",
        "agreement_summary",
        "agreement_summary_grid",
    }
)

SQUARE_METRIC_PANEL_PLOT_IDS: frozenset[str] = frozenset(
    {
        "representation_health_summary",
        "design_structure_summary",
        "sigma35_ordinal_audit",
        "context_robustness_summary",
        "reference_alignment_summary",
    }
)


def metric_panel_uses_square_axes(plot_id: str | None) -> bool:
    return plot_id in SQUARE_METRIC_PANEL_PLOT_IDS


class StrictPlotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlotBaseConfig(StrictPlotModel):
    semantics_ref: str | None = None
    visibility_tier: Literal["primary", "appendix", "debug", "hidden"] = "primary"
    default_hue: str | None = None
    hue_options: list["PlotHueOptionConfig"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_hue_defaults(self) -> "PlotBaseConfig":
        if self.default_hue is None:
            return self
        allowed = {option.column for option in self.hue_options}
        if self.hue_options and self.default_hue not in allowed:
            raise ValueError("plot default_hue must be declared in hue_options")
        return self


class PlotHueOptionConfig(StrictPlotModel):
    column: Identifier
    label: str
    type: Literal["categorical", "binary", "continuous", "ordinal"]


class PlotAnnotationConfig(StrictPlotModel):
    reference_set: Identifier
    require_in_every_panel: bool = False
    missing_policy: Literal["fail"] = "fail"
    collision_policy: Literal["repel_then_callout", "direct_label"] = "repel_then_callout"


class ProjectionScatterPlotConfig(PlotBaseConfig):
    kind: Literal["projection_scatter"]
    projection: Identifier
    color_column: str | None = None
    shape_column: str | None = None
    label_column: str | None = None
    label_values: list[str] = Field(default_factory=list)
    annotation: PlotAnnotationConfig | None = None

    @model_validator(mode="after")
    def _validate_label_selection(self) -> "ProjectionScatterPlotConfig":
        if self.label_values and self.label_column is None:
            raise ValueError("projection_scatter label_values require label_column")
        return self


class ProjectionGridPlotConfig(PlotBaseConfig):
    kind: Literal["projection_grid"]
    projections: list[Identifier] = Field(min_length=1)
    color_column: str | None = None
    shape_column: str | None = None
    panel_titles: list[str] | None = None
    label_column: str | None = None
    label_values: list[str] = Field(default_factory=list)
    annotation: PlotAnnotationConfig | None = None

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "ProjectionGridPlotConfig":
        if self.panel_titles is not None and len(self.panel_titles) != len(self.projections):
            raise ValueError("projection_grid panel_titles must match projections length")
        if self.label_values and self.label_column is None:
            raise ValueError("projection_grid label_values require label_column")
        return self


class HeatmapPlotConfig(PlotBaseConfig):
    kind: Literal["heatmap"]
    enrichment: Identifier | None = None
    scalar: Identifier | None = None
    row_column: str | None = None
    column_column: str | None = None
    value_column: str | None = None

    @model_validator(mode="after")
    def _validate_single_input(self) -> "HeatmapPlotConfig":
        selected = [
            name for name, value in (("enrichment", self.enrichment), ("scalar", self.scalar)) if value is not None
        ]
        if len(selected) != 1:
            joined = ", ".join(selected) if selected else "none"
            raise ValueError(f"heatmap plots require exactly one input from enrichment or scalar; got {joined}")
        if self.scalar is not None and (self.row_column is None or self.column_column is None):
            raise ValueError("scalar-backed heatmap plots require row_column and column_column")
        return self


class DistanceScatterPlotConfig(PlotBaseConfig):
    kind: Literal["distance_scatter"]
    distance: Identifier
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    shape_column: str | None = None


class XYScatterPlotConfig(PlotBaseConfig):
    kind: Literal["xy_scatter"]
    scalar: Identifier | None = None
    distance: Identifier | None = None
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    shape_column: str | None = None
    render_mode: Literal["points", "hexbin", "density_contour"] = "points"
    annotation: PlotAnnotationConfig | None = None

    @model_validator(mode="after")
    def _validate_single_input(self) -> "XYScatterPlotConfig":
        selected = [name for name, value in (("scalar", self.scalar), ("distance", self.distance)) if value is not None]
        if len(selected) != 1:
            joined = ", ".join(selected) if selected else "none"
            raise ValueError(f"xy_scatter plots require exactly one input from scalar or distance; got {joined}")
        return self


class XYScatterGridPlotConfig(PlotBaseConfig):
    kind: Literal["xy_scatter_grid"]
    scalars: list[Identifier] = Field(min_length=1)
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    shape_column: str | None = None
    render_mode: Literal["points", "hexbin", "density_contour"] = "points"
    panel_titles: list[str] | None = None
    annotation: PlotAnnotationConfig | None = None

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "XYScatterGridPlotConfig":
        if self.panel_titles is not None and len(self.panel_titles) != len(self.scalars):
            raise ValueError("xy_scatter_grid panel_titles must match scalars length")
        return self


class PairedXYScatterGridPlotConfig(PlotBaseConfig):
    kind: Literal["paired_xy_scatter_grid"]
    scalars: list[Identifier] = Field(min_length=1)
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    shape_column: str | None = None
    pair_id_column: str | None = None
    render_mode: Literal["points", "hexbin", "density_contour"] = "points"
    panel_titles: list[str] | None = None
    annotation: PlotAnnotationConfig | None = None

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "PairedXYScatterGridPlotConfig":
        if self.panel_titles is not None and len(self.panel_titles) != len(self.scalars):
            raise ValueError("paired_xy_scatter_grid panel_titles must match scalars length")
        return self


class CategoricalCountPlotConfig(PlotBaseConfig):
    kind: Literal["categorical_count"]
    scalar: Identifier
    category_column: str
    label_column: str
    value_column: str = "row_count"
    panel_column: str | None = None
    color_column: str | None = None


class MetricPanelGridPlotConfig(PlotBaseConfig):
    kind: Literal["metric_panel_grid"]
    scalar: Identifier
    facet_column: str
    panel_title_column: str
    category_column: str
    label_column: str
    value_column: str
    ci_lower_column: str | None = None
    ci_upper_column: str | None = None
    color_column: str | None = None
    direction_column: str | None = None
    unit_column: str | None = None
    sort_rule: Literal["panel_direction", "value_desc", "value_asc", "label_asc"] = "panel_direction"
    measure_kind: Literal["metric"] = "metric"
    value_kind: str
    value_label: str
    reference_line: float | None = None


class DistributionPlotConfig(PlotBaseConfig):
    kind: Literal["distribution"]
    scalar: Identifier | None = None
    distance: Identifier | None = None
    enrichment: Identifier | None = None
    agreement: Identifier | None = None
    value_column: str | None = None
    color_column: str | None = None
    render_mode: Literal["histogram", "ecdf", "violin_box"] = "histogram"

    @model_validator(mode="after")
    def _validate_single_input(self) -> "DistributionPlotConfig":
        selected = [
            name
            for name, value in (
                ("scalar", self.scalar),
                ("distance", self.distance),
                ("enrichment", self.enrichment),
                ("agreement", self.agreement),
            )
            if value is not None
        ]
        if len(selected) != 1:
            joined = ", ".join(selected) if selected else "none"
            raise ValueError(
                "distribution plots require exactly one artifact input from scalar, "
                f"distance, enrichment, or agreement; got {joined}"
            )
        return self


class DistributionGridPlotConfig(PlotBaseConfig):
    kind: Literal["distribution_grid"]
    scalars: list[Identifier] = Field(min_length=1)
    metric_columns: list[str] | None = None
    value_columns: list[str] | None = None
    color_column: str | None = None
    render_mode: Literal["histogram", "ecdf", "violin_box"] = "histogram"
    panel_titles: list[str] | None = None

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "DistributionGridPlotConfig":
        if self.metric_columns is not None and not self.metric_columns:
            raise ValueError("distribution_grid metric_columns must not be empty when provided")
        expected_panels = len(self.scalars) * len(self.metric_columns or [None])
        if self.panel_titles is not None and len(self.panel_titles) != expected_panels:
            raise ValueError("distribution_grid panel_titles must match rendered panel count")
        if self.value_columns is not None and len(self.value_columns) != len(self.scalars):
            raise ValueError("distribution_grid value_columns must match scalars length")
        if self.metric_columns is not None and self.value_columns is not None:
            raise ValueError("distribution_grid cannot declare both metric_columns and value_columns")
        return self


class AgreementSummaryPlotConfig(PlotBaseConfig):
    kind: Literal["agreement_summary"]
    agreement: Identifier


class AgreementSummaryGridPlotConfig(PlotBaseConfig):
    kind: Literal["agreement_summary_grid"]
    agreements: list[Identifier] = Field(min_length=1)
    panel_titles: list[str] | None = None

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "AgreementSummaryGridPlotConfig":
        if self.panel_titles is not None and len(self.panel_titles) != len(self.agreements):
            raise ValueError("agreement_summary_grid panel_titles must match agreements length")
        return self


class CurvePlotConfig(PlotBaseConfig):
    kind: Literal["curve"]
    reducer: Identifier | None = None

    @model_validator(mode="after")
    def _validate_input(self) -> "CurvePlotConfig":
        if self.reducer is None:
            raise ValueError("curve plots currently require a reducer input")
        return self


class CurveGridPlotConfig(PlotBaseConfig):
    kind: Literal["curve_grid"]
    reducers: list[Identifier] = Field(min_length=1)
    panel_titles: list[str] | None = None

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "CurveGridPlotConfig":
        if self.panel_titles is not None and len(self.panel_titles) != len(self.reducers):
            raise ValueError("curve_grid panel_titles must match reducers length")
        return self


class CorrespondenceHeatmapPlotConfig(PlotBaseConfig):
    kind: Literal["correspondence_heatmap"]
    left_cluster: Identifier
    right_cluster: Identifier


PlotConfig = Annotated[
    ProjectionScatterPlotConfig
    | ProjectionGridPlotConfig
    | HeatmapPlotConfig
    | DistanceScatterPlotConfig
    | XYScatterPlotConfig
    | XYScatterGridPlotConfig
    | PairedXYScatterGridPlotConfig
    | CategoricalCountPlotConfig
    | MetricPanelGridPlotConfig
    | DistributionPlotConfig
    | DistributionGridPlotConfig
    | CurvePlotConfig
    | CurveGridPlotConfig
    | CorrespondenceHeatmapPlotConfig
    | AgreementSummaryPlotConfig
    | AgreementSummaryGridPlotConfig,
    Field(discriminator="kind"),
]


class ResolvedPlotSpec(StrictPlotModel):
    plot_id: Identifier
    kind: Literal[
        "projection_scatter",
        "projection_grid",
        "heatmap",
        "distance_scatter",
        "xy_scatter",
        "xy_scatter_grid",
        "paired_xy_scatter_grid",
        "categorical_count",
        "metric_panel_grid",
        "distribution",
        "distribution_grid",
        "curve",
        "curve_grid",
        "correspondence_heatmap",
        "agreement_summary",
        "agreement_summary_grid",
    ]
    projection_ids: list[Identifier] = Field(default_factory=list)
    enrichment_id: Identifier | None = None
    distance_id: Identifier | None = None
    scalar_id: Identifier | None = None
    scalar_ids: list[Identifier] = Field(default_factory=list)
    agreement_id: Identifier | None = None
    agreement_ids: list[Identifier] = Field(default_factory=list)
    reducer_id: Identifier | None = None
    reducer_ids: list[Identifier] = Field(default_factory=list)
    left_cluster_id: Identifier | None = None
    right_cluster_id: Identifier | None = None
    value_column: str | None = None
    ci_lower_column: str | None = None
    ci_upper_column: str | None = None
    value_columns: list[str] = Field(default_factory=list)
    metric_columns: list[str] = Field(default_factory=list)
    row_column: str | None = None
    column_column: str | None = None
    panel_column: str | None = None
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    shape_column: str | None = None
    direction_column: str | None = None
    unit_column: str | None = None
    reference_line: float | None = None
    pair_id_column: str | None = None
    render_mode: str | None = None
    label_column: str | None = None
    label_values: list[str] = Field(default_factory=list)
    panel_titles: list[str] = Field(default_factory=list)
    annotation: PlotAnnotationConfig | None = None
    measure_kind: str | None = None
    value_kind: str | None = None
    value_label: str | None = None
    sort_rule: str | None = None
    default_hue: str | None = None
    hue_options: list[PlotHueOptionConfig] = Field(default_factory=list)
    config_id: Identifier | None = None
    semantics_ref: str | None = None
