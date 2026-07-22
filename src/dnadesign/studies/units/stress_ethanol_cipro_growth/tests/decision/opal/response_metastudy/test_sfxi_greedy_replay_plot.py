"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_sfxi_greedy_replay_plot.py

Publication contracts for the historical SFXI greedy replay plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import (
    sfxi_greedy_replay_plot,
)


def test_sfxi_greedy_replay_plot_shows_pool_exact_top_k_and_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[plt.Figure] = []
    monkeypatch.setattr(sfxi_greedy_replay_plot, "save_metastudy_figure", lambda figure, _path: captured.append(figure))

    sfxi_greedy_replay_plot.write_historical_sfxi_greedy_replay(
        _scored(),
        _replay(),
        tmp_path / "historical_sfxi_greedy_replay.png",
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
        assert all(axis.get_xlabel() == r"Logic fidelity, $F_{\mathrm{logic}}$" for axis in figure.axes)
        assert [axis.get_ylabel() for axis in figure.axes] == [r"Scaled effect, $E_{\mathrm{scaled}}$", "", ""]
        assert all(len(axis.collections) >= 2 for axis in figure.axes)
        assert all(len(axis.collections[-1].get_offsets()) == 6 for axis in figure.axes)
        visible = " ".join(
            [text.get_text() for axis in figure.axes for text in axis.texts]
            + [text.get_text() for text in figure.texts]
            + [text.get_text() for legend in figure.legends for text in legend.get_texts()]
        )
        assert "18 view slots" in visible
        assert "11 unique sequences" in visible
        assert "2 selected in all three" in visible
        assert "120 eligible predictions per view" in visible
        assert "Rank agreement with SFXI" in visible
        assert "Scaled effect" in visible
        assert "Logic fidelity" in visible
        assert all(
            text.get_position()[1] > 1.0
            for axis in figure.axes
            for text in axis.texts
            if text.get_text().startswith("Rank agreement with SFXI")
        )
        assert "Selected in multiple views" in visible
        assert "All eligible predictions (density)" in visible
        assert len(figure.legends) == 1
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert "Equal SFXI score" not in visible
        assert all(not axis.lines for axis in figure.axes)
        assert not any(axis.get_label() == "<colorbar>" for axis in figure.axes)
        assert figure._suptitle is not None and figure._suptitle.get_text() == "SFXI greedy selection replay"
        assert figure._suptitle.get_fontsize() >= 18
        assert "historical" not in visible.lower()
        assert all(axis.title.get_fontsize() >= 15 for axis in figure.axes)
        assert all(axis.xaxis.label.get_fontsize() >= 13 for axis in figure.axes)
        assert all(min(tick.get_fontsize() for tick in axis.get_xticklabels()) >= 11 for axis in figure.axes)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        canvas_box = figure.get_window_extent(renderer)
        legend_box = figure.legends[0].get_window_extent(renderer)
        assert all(not legend_box.overlaps(axis.get_window_extent(renderer)) for axis in figure.axes)
        assert all(axis.get_window_extent(renderer).height / canvas_box.height >= 0.45 for axis in figure.axes)
        for artist in [figure._suptitle, *figure.texts, *figure.legends]:
            if artist is None:
                continue
            box = artist.get_window_extent(renderer)
            assert box.x0 >= canvas_box.x0 - 1
            assert box.x1 <= canvas_box.x1 + 1
            assert box.y0 >= canvas_box.y0 - 1
            assert box.y1 <= canvas_box.y1 + 1
        for axis in figure.axes:
            rank_boxes = [
                Text.get_window_extent(text, renderer)
                for text in axis.texts
                if text.get_text() in {"1", "2", "3", "4", "5", "6"}
            ]
            assert len(rank_boxes) == 6
            assert not any(
                left.overlaps(right) for index, left in enumerate(rank_boxes) for right in rank_boxes[index + 1 :]
            )
    finally:
        plt.close(figure)


def test_sfxi_greedy_replay_export_stays_compact_at_notebook_width(tmp_path: Path) -> None:
    path = tmp_path / "historical_sfxi_greedy_replay.png"

    sfxi_greedy_replay_plot.write_historical_sfxi_greedy_replay(_scored(), _replay(), path)

    pixels = mpimg.imread(path)
    height, width = pixels.shape[:2]
    assert width / height <= 3.3
    assert width >= 2400
    assert height >= 900


def _scored() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for view_index, view_id in enumerate(("ethanol", "ciprofloxacin", "and")):
        logic = np.linspace(0.2 + view_index * 0.05, 0.75 - view_index * 0.03, 120)
        effect = np.linspace(0.42, 0.05, 120) + 0.025 * np.sin(np.linspace(0, 5, 120))
        result[view_id] = pd.DataFrame(
            {
                "id": [f"candidate-{index}" for index in range(120)],
                "logic_fidelity": logic,
                "effect_scaled": effect,
                "score": logic * effect,
            }
        )
    return result


def _replay() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = {
        "ethanol": tuple(f"candidate-{index}" for index in (0, 1, 2, 3, 4, 5)),
        "ciprofloxacin": tuple(f"candidate-{index}" for index in (0, 1, 2, 3, 6, 7)),
        "and": tuple(f"candidate-{index}" for index in (0, 1, 4, 8, 9, 10)),
    }
    view_counts = {
        candidate_id: sum(candidate_id in ids for ids in selected.values())
        for candidate_id in {candidate for ids in selected.values() for candidate in ids}
    }
    for view_id, ids in selected.items():
        for rank, candidate_id in enumerate(ids, start=1):
            rows.append(
                {
                    "selection_view_id": view_id,
                    "rank": rank,
                    "id": candidate_id,
                    "logic_fidelity": 0.26 + rank * 0.02,
                    "effect_scaled": 0.40 - rank * 0.02,
                    "score": (0.26 + rank * 0.02) * (0.40 - rank * 0.02),
                    "selection_view_count": view_counts[candidate_id],
                    "score_vs_effect_spearman": 0.99,
                    "score_vs_logic_spearman": -0.40,
                    "total_selection_slots": 18,
                    "unique_selected_sequences": 11,
                    "selected_in_all_views": 2,
                }
            )
    return pd.DataFrame.from_records(rows)
