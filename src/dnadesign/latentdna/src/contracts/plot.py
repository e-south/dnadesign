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
        "distribution",
        "curve",
        "correspondence_heatmap",
        "agreement_summary",
    }
)


class StrictPlotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectionScatterPlotConfig(StrictPlotModel):
    kind: Literal["projection_scatter"]
    projection: Identifier
    color_column: str | None = None
    label_column: str | None = None
    label_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_label_selection(self) -> "ProjectionScatterPlotConfig":
        if self.label_values and self.label_column is None:
            raise ValueError("projection_scatter label_values require label_column")
        return self


class ProjectionGridPlotConfig(StrictPlotModel):
    kind: Literal["projection_grid"]
    projections: list[Identifier] = Field(min_length=1)
    color_column: str | None = None
    panel_titles: list[str] | None = None
    label_column: str | None = None
    label_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_grid_shape(self) -> "ProjectionGridPlotConfig":
        if self.panel_titles is not None and len(self.panel_titles) != len(self.projections):
            raise ValueError("projection_grid panel_titles must match projections length")
        if self.label_values and self.label_column is None:
            raise ValueError("projection_grid label_values require label_column")
        return self


class HeatmapPlotConfig(StrictPlotModel):
    kind: Literal["heatmap"]
    enrichment: Identifier
    value_column: str | None = None


class DistanceScatterPlotConfig(StrictPlotModel):
    kind: Literal["distance_scatter"]
    distance: Identifier
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None


class XYScatterPlotConfig(StrictPlotModel):
    kind: Literal["xy_scatter"]
    scalar: Identifier | None = None
    distance: Identifier | None = None
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    render_mode: Literal["points", "hexbin", "density_contour"] = "points"

    @model_validator(mode="after")
    def _validate_single_input(self) -> "XYScatterPlotConfig":
        selected = [name for name, value in (("scalar", self.scalar), ("distance", self.distance)) if value is not None]
        if len(selected) != 1:
            joined = ", ".join(selected) if selected else "none"
            raise ValueError(f"xy_scatter plots require exactly one input from scalar or distance; got {joined}")
        return self


class DistributionPlotConfig(StrictPlotModel):
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


class AgreementSummaryPlotConfig(StrictPlotModel):
    kind: Literal["agreement_summary"]
    agreement: Identifier


class CurvePlotConfig(StrictPlotModel):
    kind: Literal["curve"]
    reducer: Identifier | None = None

    @model_validator(mode="after")
    def _validate_input(self) -> "CurvePlotConfig":
        if self.reducer is None:
            raise ValueError("curve plots currently require a reducer input")
        return self


class CorrespondenceHeatmapPlotConfig(StrictPlotModel):
    kind: Literal["correspondence_heatmap"]
    left_cluster: Identifier
    right_cluster: Identifier


PlotConfig = Annotated[
    ProjectionScatterPlotConfig
    | ProjectionGridPlotConfig
    | HeatmapPlotConfig
    | DistanceScatterPlotConfig
    | XYScatterPlotConfig
    | DistributionPlotConfig
    | CurvePlotConfig
    | CorrespondenceHeatmapPlotConfig
    | AgreementSummaryPlotConfig,
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
        "distribution",
        "curve",
        "correspondence_heatmap",
        "agreement_summary",
    ]
    projection_ids: list[Identifier] = Field(default_factory=list)
    enrichment_id: Identifier | None = None
    distance_id: Identifier | None = None
    scalar_id: Identifier | None = None
    agreement_id: Identifier | None = None
    reducer_id: Identifier | None = None
    left_cluster_id: Identifier | None = None
    right_cluster_id: Identifier | None = None
    value_column: str | None = None
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    render_mode: str | None = None
    label_column: str | None = None
    label_values: list[str] = Field(default_factory=list)
    panel_titles: list[str] = Field(default_factory=list)
    config_id: Identifier | None = None
