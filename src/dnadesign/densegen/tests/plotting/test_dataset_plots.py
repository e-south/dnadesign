"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_dataset_plots.py

Focused tests for DenseGen dataset-native plotting.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.ticker import MaxNLocator

from dnadesign.densegen.src.viz import plot_dataset as plot_dataset_module


@pytest.fixture(autouse=True)
def _close_figures_after_test() -> None:
    yield
    plt.close("all")


def test_compact_count_label_shortens_large_values() -> None:
    assert plot_dataset_module._compact_count_label(7110) == "7.1k"
    assert plot_dataset_module._compact_count_label(12000) == "12k"
    assert plot_dataset_module._compact_count_label(999) == "999"


def test_dataset_source_inventory_uses_wide_vertical_layout_and_compact_source_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _capture_save(fig, out: Path, *, style=None) -> None:
        del style
        captured["fig"] = fig
        out.write_bytes(b"plot")

    monkeypatch.setattr(plot_dataset_module, "_save_figure", _capture_save)
    df = pd.DataFrame(
        {
            "source": [
                "plan_pool__background_only__sig35_f",
                "plan_pool__background_only__sig35_f",
                "plan_pool__ethanol__sig35_f",
                "plan_pool__ciprofloxacin__sig35_d",
                "plan_pool__ethanol_ciprofloxacin__sig35_e",
            ]
        }
    )

    outputs = plot_dataset_module.plot_dataset_source_inventory(
        df,
        tmp_path / "dataset_source_inventory.png",
        style={},
    )

    assert outputs == [tmp_path / "dataset" / "dataset_source_inventory.png"]
    fig = captured["fig"]
    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    label_colors = [tick.get_color() for tick in ax.get_xticklabels()]
    bar_colors = {tuple(round(channel, 3) for channel in patch.get_facecolor()) for patch in ax.patches}
    bar_heights = [patch.get_height() for patch in ax.patches]
    fig_width, fig_height = fig.get_size_inches()
    assert fig_width > fig_height * 1.5
    assert fig_height < 5.0
    assert len(ax.collections) == 0
    assert len(ax.patches) == 4
    assert len(bar_colors) == 4
    assert bar_heights == sorted(bar_heights, reverse=True)
    assert labels[0] == "Neutral σ70-35(f)"
    assert "Neutral σ70-35(f)" in labels
    assert "EtOH σ70-35(f)" in labels
    assert "Cipro σ70-35(d)" in labels
    assert "EtOH+Cipro σ70-35(e)" in labels
    assert {tick.get_rotation() for tick in ax.get_xticklabels()} == {45.0}
    assert set(label_colors) == {"#111111"}
    assert ax.get_title() == "Dense arrays broken down by part-type composition"
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == "Counts"
    assert isinstance(ax.yaxis.get_major_locator(), MaxNLocator)


def test_dataset_metadata_heatmap_reports_source_recovery_footnote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _capture_save(fig, out: Path, *, style=None) -> None:
        del style
        captured["fig"] = fig
        out.write_bytes(b"plot")

    monkeypatch.setattr(plot_dataset_module, "_save_figure", _capture_save)
    df = pd.DataFrame(
        {
            "source": [
                "plan_pool__ethanol__sig35_f",
                "plan_pool__ethanol__sig35_f",
                "plan_pool__ciprofloxacin__sig35_d",
            ],
            "densegen__plan": ["ethanol__sig35=f", "ethanol__sig35=f", "ciprofloxacin__sig35=d"],
            "densegen__input_name": [
                "plan_pool__ethanol__sig35_f",
                "plan_pool__ethanol__sig35_f",
                "plan_pool__ciprofloxacin__sig35_d",
            ],
            "densegen__metadata_inferred_from_source": [True, True, False],
        }
    )

    outputs = plot_dataset_module.plot_dataset_metadata_heatmap(
        df,
        tmp_path / "dataset_metadata_heatmap.png",
        style={},
    )

    assert outputs == [tmp_path / "dataset" / "dataset_metadata_heatmap.png"]
    fig = captured["fig"]
    footer = "\n".join(text.get_text() for text in fig.texts)
    assert "recovered from `source` for 2 rows" in footer
