"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/registry.py

Supported plot identifiers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlotMeta:
    id: str
    summary: str
    requires: tuple[str, ...]
    output_formats: tuple[str, ...]
    data_shape: str
    failure_modes: tuple[str, ...]
    capability: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PLOT_REGISTRY: dict[str, PlotMeta] = {
    "position_scatter_and_heatmap": PlotMeta(
        id="position_scatter_and_heatmap",
        summary="Position-level mutation effect scatter and heatmap for one observed metric.",
        requires=("records.parquet", "permuter__observed__<metric_id>", "permuter__modifications"),
        output_formats=("pdf",),
        data_shape="single-reference mutation table",
        failure_modes=(
            "missing observed metric column",
            "missing mutation position metadata",
            "no round-1 variants to plot",
        ),
        capability={
            "data_layer": "permuter_records",
            "round_scope": "single_workspace_dataset",
            "requires_metric": True,
            "tidy_available": False,
        },
    ),
    "ranked_variants": PlotMeta(
        id="ranked_variants",
        summary="Ranked variant score distribution for one observed metric.",
        requires=("records.parquet", "permuter__observed__<metric_id>"),
        output_formats=("png",),
        data_shape="variant score table",
        failure_modes=("missing observed metric column", "non-numeric metric values"),
        capability={
            "data_layer": "permuter_records",
            "round_scope": "single_workspace_dataset",
            "requires_metric": True,
            "tidy_available": False,
        },
    ),
    "synergy_scatter": PlotMeta(
        id="synergy_scatter",
        summary="Observed-vs-expected interaction scatter for combination protocols.",
        requires=(
            "records.parquet",
            "permuter__observed__<metric_id>",
            "permuter__expected__<metric_id>",
        ),
        output_formats=("png",),
        data_shape="combination score table",
        failure_modes=("missing expected metric column", "ambiguous interaction metric"),
        capability={
            "data_layer": "permuter_records",
            "round_scope": "single_workspace_dataset",
            "requires_metric": True,
            "tidy_available": False,
        },
    ),
    "metric_by_mutation_count": PlotMeta(
        id="metric_by_mutation_count",
        summary="Observed metric distribution grouped by mutation count.",
        requires=("records.parquet", "permuter__observed__<metric_id>", "permuter__round"),
        output_formats=("png",),
        data_shape="variant score table",
        failure_modes=("missing observed metric column", "missing mutation count or modifications"),
        capability={
            "data_layer": "permuter_records",
            "round_scope": "single_workspace_dataset",
            "requires_metric": True,
            "tidy_available": False,
        },
    ),
    "aa_category_effects": PlotMeta(
        id="aa_category_effects",
        summary="Amino-acid category effect view for coding or protein mutation scans.",
        requires=("records.parquet", "permuter__observed__<metric_id>", "permuter__aa_* metadata"),
        output_formats=("png",),
        data_shape="amino-acid mutation table",
        failure_modes=("missing amino-acid mutation metadata", "missing observed metric column"),
        capability={
            "data_layer": "permuter_records",
            "round_scope": "single_workspace_dataset",
            "requires_metric": True,
            "tidy_available": False,
        },
    ),
    "hairpin_length_vs_metric": PlotMeta(
        id="hairpin_length_vs_metric",
        summary="Hairpin paired-length diagnostic against one observed metric.",
        requires=("records.parquet", "permuter__observed__<metric_id>", "permuter__hp_length_paired"),
        output_formats=("png",),
        data_shape="hairpin scan table",
        failure_modes=("missing hairpin length metadata", "missing observed metric column"),
        capability={
            "data_layer": "permuter_records",
            "round_scope": "single_workspace_dataset",
            "requires_metric": True,
            "tidy_available": False,
        },
    ),
}


def supported_plot_ids() -> tuple[str, ...]:
    return tuple(PLOT_REGISTRY)


def assert_supported_plot_id(plot_id: str) -> str:
    name = str(plot_id or "").strip()
    if name not in PLOT_REGISTRY:
        raise ValueError(f"Unknown plot {plot_id!r}. Supported plots: {', '.join(supported_plot_ids())}")
    return name


def plot_meta(plot_id: str) -> PlotMeta:
    return PLOT_REGISTRY[assert_supported_plot_id(plot_id)]


def plot_registry_payload() -> dict[str, object]:
    return {
        "schema": "permuter.plot_registry.v1",
        "plots": [PLOT_REGISTRY[plot_id].to_dict() for plot_id in supported_plot_ids()],
    }


def plot_description_payload(plot_id: str) -> dict[str, object]:
    return {
        "schema": "permuter.plot_description.v1",
        "plot": plot_meta(plot_id).to_dict(),
    }
