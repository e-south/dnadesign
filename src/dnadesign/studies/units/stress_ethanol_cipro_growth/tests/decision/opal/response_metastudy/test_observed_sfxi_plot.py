"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_observed_sfxi_plot.py

Publication contracts for the historical observed-label SFXI decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import (
    observed_sfxi_plot,
)


def test_historical_observed_sfxi_plot_is_measured_square_and_colorbar_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(observed_sfxi_plot, "save_metastudy_figure", lambda figure, _path: captured.append(figure))

    observed_sfxi_plot.write_historical_observed_sfxi_decomposition(
        _components(),
        _robustness(),
        tmp_path / "historical_observed_sfxi_decomposition.png",
    )

    figure = captured[0]
    try:
        assert len(figure.axes) == 3
        assert [axis.get_title() for axis in figure.axes] == [
            "Ethanol-associated",
            "Ciprofloxacin-associated",
            "Combined-state-only",
        ]
        assert all(axis.get_box_aspect() == 1.0 for axis in figure.axes)
        assert all(axis.get_xlim() == pytest.approx((-0.025, 1.025)) for axis in figure.axes)
        assert all(axis.get_ylim() == pytest.approx((-0.025, 1.025)) for axis in figure.axes)
        assert all(axis.get_xticks().tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0]) for axis in figure.axes)
        assert all(len(axis.collections[0].get_offsets()) == 35 for axis in figure.axes)
        assert all(len(axis.collections[1].get_offsets()) == 6 for axis in figure.axes)
        assert figure._suptitle is not None and figure._suptitle.get_fontsize() >= 18
        assert all(axis.title.get_fontsize() >= 15 for axis in figure.axes)
        assert all(axis.xaxis.label.get_fontsize() >= 13 for axis in figure.axes)
        assert all(min(tick.get_fontsize() for tick in axis.get_xticklabels()) >= 11 for axis in figure.axes)
        visible = " ".join(
            [axis.get_title() for axis in figure.axes]
            + [axis.get_xlabel() for axis in figure.axes]
            + [axis.get_ylabel() for axis in figure.axes]
            + [text.get_text() for axis in figure.axes for text in axis.texts]
            + [text.get_text() for legend in figure.legends for text in legend.get_texts()]
        )
        assert "SpyP" in visible
        assert "sulAp" in visible
        assert "Highest six measured SFXI scores" in visible
        assert "pDual-10-spyp" not in visible
        assert "selected top-6" not in visible.lower()
        assert not any(axis.get_label() == "<colorbar>" for axis in figure.axes)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            axes_box = axis.get_window_extent(renderer)
            for annotation in axis.texts:
                box = annotation.get_window_extent(renderer)
                assert box.x0 >= axes_box.x0 - 1
                assert box.x1 <= axes_box.x1 + 1
                assert box.y0 >= axes_box.y0 - 1
                assert box.y1 <= axes_box.y1 + 1
    finally:
        plt.close(figure)


def _components() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for view_index, view_id in enumerate(("ethanol", "ciprofloxacin", "and")):
        for index in range(35):
            logic = 0.10 + 0.80 * index / 34.0
            effect = 0.92 - 0.65 * index / 34.0 + 0.01 * view_index
            rows.append(
                {
                    "id": f"candidate-{index}",
                    "selection_view_id": view_id,
                    "logic_fidelity": logic,
                    "effect_scaled": effect,
                    "sfxi": logic * effect,
                    "is_highest_observed_sfxi": index >= 29,
                    "control_role": "SpyP" if index == 22 else "sulAp" if index == 12 else "",
                }
            )
    return pd.DataFrame.from_records(rows)


def _robustness() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "selection_view_id": ["ethanol", "ciprofloxacin", "and"],
            "sensitivity_scope": ["all_observed_labels"] * 3,
            "sfxi_vs_logic_spearman": [-0.21, -0.03, -0.17],
            "sfxi_vs_effect_spearman": [0.97, 0.92, 0.95],
            "correlation_defined": [True, True, True],
        }
    )
