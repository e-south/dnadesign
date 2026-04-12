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
        "distribution",
        "agreement_summary",
    }
)


class StrictPlotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectionScatterPlotConfig(StrictPlotModel):
    kind: Literal["projection_scatter"]
    projection: Identifier
    color_column: str | None = None


class ProjectionGridPlotConfig(StrictPlotModel):
    kind: Literal["projection_grid"]
    projections: list[Identifier] = Field(min_length=1)
    color_column: str | None = None


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


class DistributionPlotConfig(StrictPlotModel):
    kind: Literal["distribution"]
    scalar: Identifier | None = None
    distance: Identifier | None = None
    enrichment: Identifier | None = None
    agreement: Identifier | None = None
    value_column: str | None = None
    color_column: str | None = None

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


PlotConfig = Annotated[
    ProjectionScatterPlotConfig
    | ProjectionGridPlotConfig
    | HeatmapPlotConfig
    | DistanceScatterPlotConfig
    | DistributionPlotConfig
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
        "distribution",
        "agreement_summary",
    ]
    projection_ids: list[Identifier] = Field(default_factory=list)
    enrichment_id: Identifier | None = None
    distance_id: Identifier | None = None
    scalar_id: Identifier | None = None
    agreement_id: Identifier | None = None
    value_column: str | None = None
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    config_id: Identifier | None = None
