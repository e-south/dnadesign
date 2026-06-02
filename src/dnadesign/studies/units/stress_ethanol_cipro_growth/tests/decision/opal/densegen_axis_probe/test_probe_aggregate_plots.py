from __future__ import annotations

from .helpers import pytest
from .probe_modules import probe_module


def test_probe_aggregate_plot_registry_is_fail_fast() -> None:
    contracts = probe_module("reporting.review.aggregate_plots.contracts")
    spec = contracts.ProbeAggregatePlotSpec(
        name="diagnostic",
        filename="diagnostic.png",
        title="Diagnostic",
        availability="runs",
        outcome_focus="test focus",
    )

    assert tuple(contracts.PROBE_AGGREGATE_PLOT_REGISTRY)[:4] == (
        "target_lift_and_precision",
        "positive_null_lift_delta",
        "evaluable_selected_count",
        "trajectory_qa_matrix",
    )
    assert contracts.PROBE_AGGREGATE_PLOT_REGISTRY["positive_null_lift_delta"].outcome_focus.startswith(
        "paired separation"
    )
    with pytest.raises(ValueError, match="Duplicate probe aggregate plot name"):
        contracts.build_probe_aggregate_plot_registry((spec, spec))
    with pytest.raises(ValueError, match="must write a .png"):
        contracts.build_probe_aggregate_plot_registry(
            (
                contracts.ProbeAggregatePlotSpec(
                    name="bad_extension",
                    filename="bad_extension.svg",
                    title="Bad extension",
                    availability="runs",
                    outcome_focus="test focus",
                ),
            )
        )
    with pytest.raises(ValueError, match="Unsupported probe aggregate plot availability"):
        contracts.build_probe_aggregate_plot_registry(
            (
                contracts.ProbeAggregatePlotSpec(
                    name="bad_availability",
                    filename="bad_availability.png",
                    title="Bad availability",
                    availability="legacy",  # type: ignore[arg-type]
                    outcome_focus="test focus",
                ),
            )
        )


def test_review_axis_style_contract_enforces_spines_ticks_and_square() -> None:
    import matplotlib.pyplot as plt

    plot_style = probe_module("tfbs.plot_style")
    fig, ax = plt.subplots(figsize=plot_style.REVIEW_SQUARE_FIGSIZE)

    plot_style.style_review_axis(ax, square=True)

    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_edgecolor()
    assert ax.get_box_aspect() == pytest.approx(1.0)
    plt.close(fig)


def test_probe_aggregate_plot_renderers_match_registry() -> None:
    writer = probe_module("reporting.review.aggregate_plots.writer")

    writer.validate_probe_aggregate_plot_renderers()
