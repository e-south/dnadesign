"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_observed_objective_over_rounds.py

Rendering tests for generic observed-objective evidence over campaign rounds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dnadesign.opal.src.analysis.observed_objective_history import ObservedObjectiveHistory
from dnadesign.opal.src.plots import observed_objective_over_rounds as plot_mod
from dnadesign.opal.src.plots._context import PlotContext
from dnadesign.opal.src.registries.plots import describe_plot_kind


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self.workdir = outputs_dir.parent


def _history() -> ObservedObjectiveHistory:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "display_label": ["A", "B", "C", "D", "E"],
            "sequence": ["ACGT"] * 5,
            "observed_round": [0, 0, 0, 1, 1],
            "batch_id": ["batch-0", "batch-0", "batch-0", "batch-1", "batch-1"],
            "y_space": ["reader_response_window_vector_v1"] * 5,
            "y_obs": [[0.0] * 8] * 5,
            "label_source_kind": ["usr_sidecar"] * 5,
            "evidence_as_of_round": [0, 0, 0, 1, 1],
            "evidence_run_id": ["r0", "r0", "r0", "r1", "r1"],
            "evidence_observed_events_sha256": ["a" * 64] * 3 + ["b" * 64] * 2,
            "objective_value": [-0.5, 0.0, 0.5, 0.4, 0.8],
            "selection_view_id": ["ethanol"] * 5,
            "objective_name": ["response_magnitude_feasibility_v1"] * 5,
            "score_ref": ["ethanol/feasibility_margin"] * 5,
            "score_channel": ["feasibility_margin"] * 5,
            "objective_mode": ["maximize"] * 5,
        }
    )
    summary = pd.DataFrame(
        {
            "observed_round": [0, 1],
            "batch_id": ["batch-0", "batch-1"],
            "candidate_count": [3, 2],
            "batch_median": [0.0, 0.6],
            "between_candidate_q25": [-0.25, 0.5],
            "between_candidate_q75": [0.25, 0.7],
            "cumulative_best": [0.5, 0.8],
        }
    )
    return ObservedObjectiveHistory(
        frame=frame,
        summary=summary,
        selection_view_id="ethanol",
        objective_name="response_magnitude_feasibility_v1",
        score_ref="ethanol/feasibility_margin",
        score_channel="feasibility_margin",
        objective_mode="maximize",
        y_space="reader_response_window_vector_v1",
        comparability_sha256="c" * 64,
    )


def test_plot_preserves_candidate_points_and_labels_between_candidate_spread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history = _history()
    monkeypatch.setattr(plot_mod, "load_observed_objective_history", lambda **_kwargs: history)
    captured: dict[str, object] = {}

    def _capture(fig, out, *, dpi, tight=True):
        captured.update(fig=fig, out=out, dpi=dpi, tight=tight)

    monkeypatch.setattr(plot_mod, "save_notebook_square_figure", _capture)
    context = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path / "outputs"),
        rounds="all",
        run_id=None,
        selection_view_id="ethanol",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="observed-history.png",
        dpi=120,
        format="png",
        logger=logging.getLogger("opal.test.observed-history"),
        save_data=True,
    )

    plot_mod.render(
        context,
        {
            "run_series": {"schema_version": "opal.observed_objective_run_series.v1", "runs": []},
            "zero_boundary": True,
            "show_cumulative_best": True,
        },
    )

    tidy = pd.read_csv(context.saved_data_paths[0])
    assert tidy.loc[tidy["row_kind"] == "candidate", "id"].tolist() == ["a", "b", "c", "d", "e"]
    assert tidy.loc[tidy["row_kind"] == "batch_summary", "candidate_count"].tolist() == [3.0, 2.0]
    fig = captured["fig"]
    ax = fig.axes[0]
    legend_labels = fig.legends[0].get_texts()
    assert [text.get_text() for text in legend_labels] == [
        "Candidate",
        "Batch median",
        "Between-candidate IQR",
        "Cumulative best",
    ]
    assert ax.title.get_ha() == "center"
    assert ax.get_title() == "Ethanol observed feasibility margin by batch"
    assert ax.get_ylabel() == r"RMF feasibility margin, $S_{\mathrm{RMF}}$"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "Batch 0\nRound 0 · n=3",
        "Batch 1\nRound 1 · n=2",
    ]
    fig.canvas.draw()
    legend_bounds = fig.legends[0].get_window_extent()
    assert legend_bounds.x0 >= fig.bbox.x0
    assert legend_bounds.x1 <= fig.bbox.x1


def test_plot_capability_declares_observed_round_history() -> None:
    description = describe_plot_kind("observed_objective_over_rounds")
    assert description["capability"] == {
        "objective_family": "generic",
        "data_layer": "labels_objective",
        "round_scope": "round_history",
        "label_requirement": "required",
        "requires_labels": True,
        "requires_model_artifact": False,
        "tidy_available": True,
    }
