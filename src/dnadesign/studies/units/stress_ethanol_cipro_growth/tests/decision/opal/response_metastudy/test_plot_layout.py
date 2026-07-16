"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_plot_layout.py

Tests for response metastudy publication plot layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import (
    plot_vocabulary,
    response_assay_plots,
    response_example_plots,
    response_model_plots,
)


def test_matrix_plot_vocabulary_covers_every_fixed_model_and_representation() -> None:
    assert set(response_model_plots._MODEL_ORDER) == set(response_model_plots._MODEL_LABELS)
    assert set(plot_vocabulary.REPRESENTATION_ORDER) <= set(plot_vocabulary.REPRESENTATION_LABELS)
    assert set(plot_vocabulary.REPRESENTATION_ORDER) <= set(plot_vocabulary.REPRESENTATION_ROLES)
    with pytest.raises(ValueError, match="has no publication label"):
        plot_vocabulary.representation_label("unknown")


@pytest.fixture
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    for module in (response_example_plots, response_assay_plots, response_model_plots):
        monkeypatch.setattr(
            module,
            "save_metastudy_figure",
            lambda figure, _path: figures.append(figure),
        )
    return figures


def test_measured_response_examples_use_only_reader_values_and_rmf_components(
    captured_figures: list[plt.Figure],
    tmp_path: Path,
) -> None:
    response_example_plots.write_measured_response_examples(
        _response_example_rows(),
        tmp_path / "measured_response_examples.png",
    )

    figure = captured_figures[0]
    try:
        axes = [axis for axis in figure.axes if axis.get_label() != "<colorbar>"]
        assert len(axes) == 6
        visible_text = " ".join(
            [axis.get_xlabel() for axis in axes]
            + [axis.get_ylabel() for axis in axes]
            + [text.get_text() for axis in axes for text in axis.texts]
        )
        assert "SFXI" not in visible_text
        assert any(axis.get_ylabel().startswith("Unscaled RMF requirement") for axis in axes)
        component_axes = [axis for axis in axes if not axis.images]
        assert len(component_axes) == 3
        assert all(axis.get_box_aspect() == 1.0 for axis in component_axes)
    finally:
        plt.close(figure)


def test_primary_matrix_plots_use_square_tiles(captured_figures: list[plt.Figure], tmp_path: Path) -> None:
    response_assay_plots.write_response_separation_stability(
        _stability_rows(),
        tmp_path / "stability.png",
        primary_reduction_id="event_logmean_4_8h_post",
    )
    response_model_plots.write_label_model_screen(_model_rows(), tmp_path / "models.png")

    try:
        assert len(captured_figures) == 2
        expected_roles = [["Candidate", "Sensitivity"], ["Candidate", "Sensitivity"]]
        for figure, roles in zip(captured_figures, expected_roles, strict=True):
            matrix_axes = [axis for axis in figure.axes if axis.images]
            assert len(matrix_axes) == 1
            assert matrix_axes[0].get_aspect() == 1.0
            assert not matrix_axes[0].child_axes
            assert [
                text.get_text()
                for text in matrix_axes[0].texts
                if str(text.get_gid() or "").startswith("column-group-label:")
            ] == roles
            assert len(
                [line for line in matrix_axes[0].lines if str(line.get_gid() or "").startswith("column-group-bracket:")]
            ) == len(roles)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            tick_boxes = [tick.get_window_extent(renderer) for tick in matrix_axes[0].get_xticklabels()]
            assert all(left.x1 < right.x0 for left, right in zip(tick_boxes, tick_boxes[1:], strict=False))
    finally:
        for figure in captured_figures:
            plt.close(figure)


def test_repeated_design_matrix_groups_measurement_and_condition_semantics(
    captured_figures: list[plt.Figure],
    tmp_path: Path,
) -> None:
    response_assay_plots.write_repeated_design_agreement(_repeat_rows(), tmp_path / "repeats.png")

    figure = captured_figures[0]
    try:
        matrix_axes = [axis for axis in figure.axes if axis.images]
        assert len(matrix_axes) == 2
        assert all(axis.get_aspect() == 1.0 for axis in matrix_axes)
        image_limits = {(axis.images[0].norm.vmin, axis.images[0].norm.vmax) for axis in matrix_axes}
        assert len(image_limits) == 1
        assert next(iter(image_limits)) == pytest.approx((0.0, 1.55))
        expected_ticks = [
            "r00\nNo stress",
            "r10\nEthanol",
            "r01\nCiprofloxacin",
            "r11\nBoth stresses",
            "b00\nNo stress",
            "b10\nEthanol",
            "b01\nCiprofloxacin",
            "b11\nBoth stresses",
        ]
        expected_groups = [
            "Response\nlog2(YFP / CFP)",
            "Relative fluorescence\nlog2(YFP / OD600) vs pDual-10",
        ]
        for axis in matrix_axes:
            assert [tick.get_text() for tick in axis.get_xticklabels()] == expected_ticks
            assert not axis.child_axes
            assert [
                text.get_text() for text in axis.texts if str(text.get_gid() or "").startswith("column-group-label:")
            ] == expected_groups
            assert (
                len([line for line in axis.lines if str(line.get_gid() or "").startswith("column-group-bracket:")]) == 2
            )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in matrix_axes:
            tick_boxes = [tick.get_window_extent(renderer) for tick in axis.get_xticklabels()]
            assert all(left.x1 < right.x0 for left, right in zip(tick_boxes, tick_boxes[1:], strict=False))
        width, height = figure.get_size_inches()
        assert height / width <= 0.75
    finally:
        plt.close(figure)


def _stability_rows() -> pd.DataFrame:
    records = []
    for selection_view_id in plot_vocabulary.TARGET_VIEW_ORDER:
        for reduction_index, reduction_id in enumerate(plot_vocabulary.REDUCTION_ORDER):
            records.append(
                {
                    "selection_view_id": selection_view_id,
                    "reduction_id": reduction_id,
                    "response_separation__spearman_to_primary": 1.0 - reduction_index * 0.04,
                    "on_magnitude_floor__spearman_to_primary": 0.98 - reduction_index * 0.04,
                    "off_magnitude_ceiling__spearman_to_primary": 0.96 - reduction_index * 0.04,
                }
            )
    return pd.DataFrame.from_records(records)


def _model_rows() -> pd.DataFrame:
    records = []
    for model_index, model_id in enumerate(response_model_plots._MODEL_ORDER):
        for representation_index, representation_id in enumerate(plot_vocabulary.REPRESENTATION_ORDER):
            score = 0.2 - 0.01 * model_index - 0.005 * representation_index
            records.append(
                {
                    "model_id": model_id,
                    "representation_id": representation_id,
                    "promotion_eligible": representation_index < 2,
                    "weakest_target_view_response_separation_spearman": score,
                    "weakest_target_view_feasibility_spearman": score,
                    "weakest_required_ordering_spearman": score,
                }
            )
    return pd.DataFrame.from_records(records)


def _repeat_rows() -> pd.DataFrame:
    records = []
    for row_index in range(12):
        row: dict[str, object] = {
            "design_id": f"design-{row_index:02d}",
            "maximum_channel_range": 1.2 - row_index * 0.04,
        }
        for component_index, prefix in enumerate(("r", "b")):
            for state_index, state in enumerate(("00", "10", "01", "11")):
                row[f"{prefix}{state}__range"] = 0.1 + row_index * 0.1 + component_index * 0.2 + state_index * 0.05
        records.append(row)
    return pd.DataFrame.from_records(records)


def _response_example_rows() -> pd.DataFrame:
    records = []
    for target_view_index, selection_view_id in enumerate(("ethanol", "ciprofloxacin", "and")):
        for example_index, label in enumerate(("SpyP control", "sulAp control")):
            value = 0.1 + target_view_index * 0.1 + example_index * 0.05
            record = {
                "selection_view_id": selection_view_id,
                "response_separation": value - 0.2,
                "on_magnitude_floor": value - 0.1,
                "off_magnitude_ceiling": 0.2 - value,
                "off_suppression": value - 0.2,
                "passes_all_zero_constraints": bool(value >= 0.2),
                "example_label": label,
            }
            for prefix in ("r", "b"):
                for state_index, state in enumerate(("00", "10", "01", "11")):
                    record[f"{prefix}{state}"] = value + state_index * 0.1
            records.append(record)
    return pd.DataFrame.from_records(records)
