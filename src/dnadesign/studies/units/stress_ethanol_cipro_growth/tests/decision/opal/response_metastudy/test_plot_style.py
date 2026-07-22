"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_plot_style.py

Tests for response metastudy publication plot styling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import plot_style


def test_plot_writer_forces_white_canvas_under_dark_runtime_style(tmp_path: Path) -> None:
    with plt.style.context("dark_background"):
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0], [0.0, 1.0])
        path = tmp_path / "policy_guardrail_matrix.png"
        plot_style.save_metastudy_figure(figure, path)

    pixels = mpimg.imread(path)
    corners = (pixels[0, 0, :3], pixels[0, -1, :3], pixels[-1, 0, :3], pixels[-1, -1, :3])
    assert all(float(corner.min()) >= 0.95 for corner in corners)


def test_plot_writer_owns_single_axis_title_and_grid_layer(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    bars = axis.bar([0, 1], [1, 2], zorder=3)
    axis.set_title("Writer title that must not survive")

    plot_style.save_metastudy_figure(figure, tmp_path / "policy_guardrail_matrix.png")

    assert axis.get_title(loc="left") == ""
    assert axis.get_title().replace("\n", " ") == "SFXI policy guardrails"
    assert axis.title.get_ha() == "center"
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert axis.spines["left"].get_visible()
    assert axis.spines["bottom"].get_visible()
    assert any(line.get_visible() for line in axis.get_ygridlines())
    assert max(line.get_zorder() for line in axis.get_ygridlines()) < min(bar.get_zorder() for bar in bars)
    assert axis.title.get_fontsize() >= 18
    assert axis.xaxis.label.get_fontsize() >= 13
    assert min(tick.get_fontsize() for tick in axis.get_xticklabels()) >= 11


def test_plot_writer_preserves_panel_titles_below_one_figure_premise(tmp_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].set_title("Ethanol", loc="left")
    axes[1].set_title("Ciprofloxacin")
    figure.suptitle("Writer-level duplicate")

    plot_style.save_metastudy_figure(figure, tmp_path / "measured_response_examples.png")

    assert figure._suptitle is not None
    assert figure._suptitle.get_text().replace("\n", " ") == "Measured responses under each target mask"
    assert figure._suptitle.get_ha() == "center"
    assert figure._suptitle.get_position()[0] == 0.5
    assert [axis.get_title() for axis in axes] == ["Ethanol", "Ciprofloxacin"]
    assert [axis.get_title(loc="left") for axis in axes] == ["", ""]
    assert all(axis.title.get_ha() == "center" for axis in axes)
    assert figure._suptitle.get_fontsize() >= 18
    assert all(axis.title.get_fontsize() >= 15 for axis in axes)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    title_box = figure._suptitle.get_window_extent(renderer)
    assert all(title_box.y0 >= axis.get_window_extent(renderer).y1 for axis in axes)


def test_plot_writer_rejects_multiple_panel_titles(tmp_path: Path) -> None:
    figure, axes = plt.subplots(1, 2)
    axes[0].set_title("Left title", loc="left")
    axes[0].set_title("Right title", loc="right")

    with pytest.raises(ValueError, match="multiple titles"):
        plot_style.save_metastudy_figure(figure, tmp_path / "measured_response_examples.png")
