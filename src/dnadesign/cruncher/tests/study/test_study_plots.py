"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_plots.py

Validate Study plotting contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

import dnadesign.cruncher.study.plots as study_plots

matplotlib.use("Agg")


def test_sequence_length_tradeoff_plot_uses_integer_x_locator(tmp_path: Path, monkeypatch) -> None:
    captured_calls: list[dict[str, object]] = []
    captured_vlines: list[dict[str, object]] = []
    captured_scatter: list[dict[str, object]] = []

    import matplotlib.axes as maxes

    original_errorbar = maxes.Axes.errorbar
    original_axvline = maxes.Axes.axvline
    original_scatter = maxes.Axes.scatter

    def _capture_errorbar(self, *args, **kwargs):
        captured_calls.append(dict(kwargs))
        return original_errorbar(self, *args, **kwargs)

    def _capture_axvline(self, *args, **kwargs):
        captured_vlines.append(dict(kwargs))
        return original_axvline(self, *args, **kwargs)

    def _capture_scatter(self, *args, **kwargs):
        captured_scatter.append(dict(kwargs))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(maxes.Axes, "errorbar", _capture_errorbar)
    monkeypatch.setattr(maxes.Axes, "axvline", _capture_axvline)
    monkeypatch.setattr(maxes.Axes, "scatter", _capture_scatter)

    class _CapturedPyplot:
        def __init__(self) -> None:
            import matplotlib.pyplot as _plt

            self._plt = _plt
            self.fig = None
            self.ax_score = None

        def subplots(self, *args, **kwargs):
            fig, ax = self._plt.subplots(*args, **kwargs)
            self.fig = fig
            self.ax_score = ax
            return fig, ax

        def close(self, fig) -> None:
            self._plt.close(fig)

    captured = _CapturedPyplot()
    monkeypatch.setattr(study_plots, "_pyplot", lambda: captured)

    df = pd.DataFrame(
        {
            "series_label": ["set=1", "set=1", "set=1"],
            "sequence_length": [16, 20, 25],
            "score_mean": [0.1, 0.2, 0.3],
            "score_sem": [0.01, 0.01, 0.01],
            "diversity_metric_mean": [1.0, 1.1, 1.2],
            "diversity_metric_sem": [0.05, 0.05, 0.05],
            "diversity_metric_label": ["Median NN full-seq Hamming (bp)"] * 3,
            "is_base_value": [False, True, False],
        }
    )
    out_path = tmp_path / "plot__sequence_length_tradeoff.pdf"
    study_plots.plot_sequence_length_tradeoff(df, out_path)

    assert out_path.exists()
    assert captured.ax_score is not None
    locator = captured.ax_score.xaxis.get_major_locator()
    assert isinstance(locator, MaxNLocator)
    assert bool(getattr(locator, "_integer", False))
    assert captured.fig is not None
    width, height = captured.fig.get_size_inches()
    assert width == height
    captured.fig.canvas.draw()
    assert captured.ax_score.xaxis.label.get_fontsize() >= 13
    assert captured.ax_score.yaxis.label.get_fontsize() >= 13
    assert captured.ax_score.get_xticklabels()[0].get_fontsize() >= 11
    assert captured.ax_score.get_yticklabels()[0].get_fontsize() >= 11
    legend = captured.ax_score.get_legend()
    assert legend is not None
    assert legend.get_frame_on() is False
    assert legend._loc == 8
    assert getattr(legend, "_ncols", 1) == 1
    renderer = captured.fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer).transformed(captured.ax_score.transAxes.inverted())
    center_x = (bbox.x0 + bbox.x1) / 2.0
    assert abs(center_x - 0.5) < 0.08
    assert 0.0 <= bbox.y0 <= 0.15
    legend_labels = [item.get_text() for item in legend.get_texts()]
    assert legend_labels
    assert "Core config value" in legend_labels
    assert all(label == "Core config value" or label.startswith(("Score:", "Diversity:")) for label in legend_labels)
    assert legend.get_texts()[0].get_fontsize() >= 10
    title = captured.ax_score.get_title()
    assert title
    assert not title.endswith(".")
    assert len(title) <= 60
    assert captured_calls
    for kwargs in captured_calls:
        assert float(kwargs.get("capsize", 0.0)) > 0.0
        color = str(kwargs.get("color"))
        ecolor = str(kwargs.get("ecolor"))
        assert color and ecolor
        assert ecolor != color
        cr, cg, cb = to_rgb(color)
        er, eg, eb = to_rgb(ecolor)
        assert er >= cr and eg >= cg and eb >= cb
    assert captured_vlines
    assert any(kwargs.get("linestyle") == "--" for kwargs in captured_vlines)
    assert captured_scatter
    assert any(str(kwargs.get("edgecolors")) == "black" for kwargs in captured_scatter)


def test_mmr_diversity_tradeoff_plot_uses_square_figure_and_frameless_legend(tmp_path: Path, monkeypatch) -> None:
    captured_calls: list[dict[str, object]] = []
    captured_vlines: list[dict[str, object]] = []
    captured_scatter: list[dict[str, object]] = []

    import matplotlib.axes as maxes

    original_errorbar = maxes.Axes.errorbar
    original_axvline = maxes.Axes.axvline
    original_scatter = maxes.Axes.scatter

    def _capture_errorbar(self, *args, **kwargs):
        captured_calls.append(dict(kwargs))
        return original_errorbar(self, *args, **kwargs)

    def _capture_axvline(self, *args, **kwargs):
        captured_vlines.append(dict(kwargs))
        return original_axvline(self, *args, **kwargs)

    def _capture_scatter(self, *args, **kwargs):
        captured_scatter.append(dict(kwargs))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(maxes.Axes, "errorbar", _capture_errorbar)
    monkeypatch.setattr(maxes.Axes, "axvline", _capture_axvline)
    monkeypatch.setattr(maxes.Axes, "scatter", _capture_scatter)

    class _CapturedPyplot:
        def __init__(self) -> None:
            import matplotlib.pyplot as _plt

            self._plt = _plt
            self.fig = None
            self.ax_score = None

        def subplots(self, *args, **kwargs):
            fig, ax = self._plt.subplots(*args, **kwargs)
            self.fig = fig
            self.ax_score = ax
            return fig, ax

        def close(self, fig) -> None:
            self._plt.close(fig)

    captured = _CapturedPyplot()
    monkeypatch.setattr(study_plots, "_pyplot", lambda: captured)

    df = pd.DataFrame(
        {
            "series_label": ["set=1", "set=1", "set=1"],
            "diversity": [0.0, 0.5, 1.0],
            "score_mean": [0.1, 0.2, 0.3],
            "score_sem": [0.01, 0.01, 0.01],
            "diversity_metric_mean": [1.0, 1.1, 1.2],
            "diversity_metric_sem": [0.05, 0.05, 0.05],
            "diversity_metric_label": ["Median NN full-seq Hamming (bp)"] * 3,
            "is_base_value": [True, False, False],
        }
    )
    out_path = tmp_path / "plot__mmr_diversity_tradeoff.pdf"
    study_plots.plot_mmr_diversity_tradeoff(df, out_path)

    assert out_path.exists()
    assert captured.fig is not None
    width, height = captured.fig.get_size_inches()
    assert width == height
    assert captured.ax_score is not None
    captured.fig.canvas.draw()
    assert captured.ax_score.xaxis.label.get_fontsize() >= 13
    assert captured.ax_score.yaxis.label.get_fontsize() >= 13
    assert captured.ax_score.get_xticklabels()[0].get_fontsize() >= 11
    assert captured.ax_score.get_yticklabels()[0].get_fontsize() >= 11
    legend = captured.ax_score.get_legend()
    assert legend is not None
    assert legend.get_frame_on() is False
    assert legend._loc == 8
    assert getattr(legend, "_ncols", 1) == 1
    renderer = captured.fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer).transformed(captured.ax_score.transAxes.inverted())
    center_x = (bbox.x0 + bbox.x1) / 2.0
    assert abs(center_x - 0.5) < 0.08
    assert 0.0 <= bbox.y0 <= 0.15
    legend_labels = [item.get_text() for item in legend.get_texts()]
    assert legend_labels
    assert "Core config value" in legend_labels
    assert all(label == "Core config value" or label.startswith(("Score:", "Diversity:")) for label in legend_labels)
    assert legend.get_texts()[0].get_fontsize() >= 10
    title = captured.ax_score.get_title()
    assert title
    assert not title.endswith(".")
    assert len(title) <= 60
    assert captured_calls
    for kwargs in captured_calls:
        assert float(kwargs.get("capsize", 0.0)) > 0.0
        color = str(kwargs.get("color"))
        ecolor = str(kwargs.get("ecolor"))
        assert color and ecolor
        assert ecolor != color
        cr, cg, cb = to_rgb(color)
        er, eg, eb = to_rgb(ecolor)
        assert er >= cr and eg >= cg and eb >= cb
    assert captured_vlines
    assert any(kwargs.get("linestyle") == "--" for kwargs in captured_vlines)
    assert captured_scatter
    assert any(str(kwargs.get("edgecolors")) == "black" for kwargs in captured_scatter)
