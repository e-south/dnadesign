"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/aggregate_plots/writer.py

Writer for registry-backed DenseGen axis probe aggregate plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ....core.artifacts import ProbeArtifactLayout
from .context import ProbeAggregatePlotContext
from .contracts import PROBE_AGGREGATE_PLOT_REGISTRY, ProbeAggregatePlotSpec
from .renderers import (
    render_evaluable_selected_count,
    render_feature_stability,
    render_positive_null_lift_delta,
    render_round_target_lift_and_precision,
    render_selected_class_composition,
    render_target_lift_and_precision,
    render_trajectory_qa_matrix,
    render_vector_reference_distance,
)
from .source_frames import has_class_composition

Renderer = Callable[[ProbeAggregatePlotContext, Path], None]

_RENDERERS: Mapping[str, Renderer] = {
    "target_lift_and_precision": render_target_lift_and_precision,
    "positive_null_lift_delta": render_positive_null_lift_delta,
    "evaluable_selected_count": render_evaluable_selected_count,
    "trajectory_qa_matrix": render_trajectory_qa_matrix,
    "selected_class_composition": render_selected_class_composition,
    "round_target_lift_and_precision": render_round_target_lift_and_precision,
    "vector_distance_to_reference_over_rounds": render_vector_reference_distance,
    "feature_stability_over_rounds": render_feature_stability,
}


def write_probe_aggregate_plots(
    layout: ProbeArtifactLayout,
    *,
    metrics_payload: Mapping[str, Any],
    configured_plots: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Write registered probe-level plots and return their paths in registry order."""

    validate_probe_aggregate_plot_renderers()
    runs = metrics_payload.get("runs") or []
    if not runs:
        return []
    context = ProbeAggregatePlotContext.from_payload(
        metrics_payload=metrics_payload,
        configured_plots=configured_plots,
    )
    layout.review_plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in PROBE_AGGREGATE_PLOT_REGISTRY.values():
        if not _available(spec, context):
            continue
        renderer = _RENDERERS[spec.name]
        path = layout.review_plots_dir / spec.filename
        renderer(context, path)
        paths.append(path)
    return paths


def validate_probe_aggregate_plot_renderers() -> None:
    """Fail fast when aggregate plot specs and renderer bindings drift."""

    spec_names = set(PROBE_AGGREGATE_PLOT_REGISTRY)
    renderer_names = set(_RENDERERS)
    missing = sorted(spec_names - renderer_names)
    extra = sorted(renderer_names - spec_names)
    if missing or extra:
        raise RuntimeError(
            f"Probe aggregate plot registry/renderer mismatch (missing_renderers={missing}, extra_renderers={extra})"
        )


def _available(spec: ProbeAggregatePlotSpec, context: ProbeAggregatePlotContext) -> bool:
    if spec.availability == "runs":
        return True
    if spec.availability == "class_composition":
        return has_class_composition(context.runs_frame)
    if spec.availability == "round_metrics":
        return not context.round_frame.empty
    if spec.availability == "configured_vector_reference":
        return not context.vector_reference_distance_frame.empty
    if spec.availability == "configured_feature_stability":
        return not context.feature_stability_frame.empty
    raise RuntimeError(f"unsupported probe aggregate plot availability: {spec.availability!r}")
