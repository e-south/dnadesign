"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/aggregate_plots/contracts.py

Contracts for registry-backed DenseGen axis probe aggregate plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

PlotAvailability = Literal[
    "runs",
    "class_composition",
    "round_metrics",
    "configured_vector_reference",
    "configured_feature_stability",
]

_VALID_AVAILABILITY: set[str] = {
    "runs",
    "class_composition",
    "round_metrics",
    "configured_vector_reference",
    "configured_feature_stability",
}


@dataclass(frozen=True)
class ProbeAggregatePlotSpec:
    """Stable contract for one probe-level review plot."""

    name: str
    filename: str
    title: str
    availability: PlotAvailability
    outcome_focus: str


def build_probe_aggregate_plot_registry(
    specs: Iterable[ProbeAggregatePlotSpec],
) -> Mapping[str, ProbeAggregatePlotSpec]:
    """Build a fail-fast aggregate plot registry keyed by stable plot name."""

    registry: dict[str, ProbeAggregatePlotSpec] = {}
    filenames: set[str] = set()
    for spec in specs:
        name = spec.name.strip()
        filename = spec.filename.strip()
        if not name:
            raise ValueError("Probe aggregate plot spec has an empty name")
        if name in registry:
            raise ValueError(f"Duplicate probe aggregate plot name: {name}")
        if not filename.endswith(".png"):
            raise ValueError(f"Probe aggregate plot {name!r} must write a .png filename")
        if filename in filenames:
            raise ValueError(f"Duplicate probe aggregate plot filename: {filename}")
        if spec.availability not in _VALID_AVAILABILITY:
            raise ValueError(f"Unsupported probe aggregate plot availability: {spec.availability!r}")
        registry[name] = spec
        filenames.add(filename)
    if not registry:
        raise ValueError("Probe aggregate plot registry must contain at least one plot")
    return MappingProxyType(registry)


PROBE_AGGREGATE_PLOT_REGISTRY = build_probe_aggregate_plot_registry(
    (
        ProbeAggregatePlotSpec(
            name="target_lift_and_precision",
            filename="target_lift_and_precision.png",
            title="Probe target lift and precision",
            availability="runs",
            outcome_focus="final selected label enrichment and precision for each scored run",
        ),
        ProbeAggregatePlotSpec(
            name="positive_null_lift_delta",
            filename="positive_null_lift_delta.png",
            title="Positive-minus-null lift separation",
            availability="runs",
            outcome_focus="paired separation between intact and matched-null oracle roles",
        ),
        ProbeAggregatePlotSpec(
            name="evaluable_selected_count",
            filename="evaluable_selected_count.png",
            title="Evaluable selected count",
            availability="runs",
            outcome_focus="whether selected IDs remain inside the scored split pool",
        ),
        ProbeAggregatePlotSpec(
            name="trajectory_qa_matrix",
            filename="trajectory_qa_matrix.png",
            title="Trajectory QA matrix",
            availability="runs",
            outcome_focus="positive, null, and paired trajectory evidence by campaign pair",
        ),
        ProbeAggregatePlotSpec(
            name="selected_class_composition",
            filename="selected_class_composition.png",
            title="Selected class composition",
            availability="class_composition",
            outcome_focus="which off-target classes were selected alongside the target class",
        ),
        ProbeAggregatePlotSpec(
            name="round_target_lift_and_precision",
            filename="round_target_lift_and_precision.png",
            title="Round-over-round target lift and precision",
            availability="round_metrics",
            outcome_focus="trajectory trend across OPAL acquisition rounds",
        ),
        ProbeAggregatePlotSpec(
            name="vector_distance_to_reference_over_rounds",
            filename="vector_distance_to_reference_over_rounds.png",
            title="Vector distance to configured reference",
            availability="configured_vector_reference",
            outcome_focus="whether configured OPAL vector plots expose reference-distance movement over rounds",
        ),
        ProbeAggregatePlotSpec(
            name="feature_stability_over_rounds",
            filename="feature_stability_over_rounds.png",
            title="Feature-importance stability",
            availability="configured_feature_stability",
            outcome_focus="adjacent-round stability of model feature-importance rankings",
        ),
    )
)
