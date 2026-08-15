"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/api/test_selection_view_performance_api.py

Tests the public cross-view observed-performance analysis and renderer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from dnadesign.opal import (
    SELECTION_VIEW_PERFORMANCE_API_VERSION,
    render_selection_view_performance,
    selection_view_performance,
)


def _observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        (1, "candidate-a", "ethanol"): 3.0,
        (1, "candidate-a", "ciprofloxacin"): 0.0,
        (1, "candidate-b", "ethanol"): 2.0,
        (1, "candidate-b", "ciprofloxacin"): 1.0,
        (1, "candidate-c", "ethanol"): -1.0,
        (1, "candidate-c", "ciprofloxacin"): 4.0,
        (1, "candidate-d", "ethanol"): 0.0,
        (1, "candidate-d", "ciprofloxacin"): 3.0,
    }
    selected_for = {
        "candidate-a": "ethanol",
        "candidate-b": "ethanol",
        "candidate-c": "ciprofloxacin",
        "candidate-d": "ciprofloxacin",
    }
    for (round_index, candidate_id, objective_view_id), objective_value in values.items():
        rows.append(
            {
                "observed_round": round_index,
                "candidate_id": candidate_id,
                "selected_for_view_id": selected_for[candidate_id],
                "objective_view_id": objective_view_id,
                "objective_value": objective_value,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_selection_view_performance_compares_selected_cohorts_within_each_objective() -> None:
    result = selection_view_performance(_observations())

    assert SELECTION_VIEW_PERFORMANCE_API_VERSION == "1"
    assert len(result.observations) == 8
    summary = result.summary.set_index(["objective_view_id", "selected_for_view_id"])
    assert summary.loc[("ethanol", "ethanol"), "candidate_count"] == 2
    assert summary.loc[("ethanol", "ethanol"), "median"] == pytest.approx(2.5)
    assert summary.loc[("ethanol", "ciprofloxacin"), "median"] == pytest.approx(-0.5)
    assert bool(summary.loc[("ethanol", "ethanol"), "selected_for_objective_view"])
    assert not bool(summary.loc[("ethanol", "ciprofloxacin"), "selected_for_objective_view"])


def test_selection_view_performance_requires_every_candidate_objective_pair() -> None:
    incomplete = _observations().iloc[:-1].copy()

    with pytest.raises(ValueError, match="complete objective grid"):
        selection_view_performance(incomplete)


def test_selection_view_performance_rejects_duplicate_candidate_objectives() -> None:
    duplicate = pd.concat((_observations(), _observations().iloc[[0]]), ignore_index=True)

    with pytest.raises(ValueError, match="duplicate candidate/objective"):
        selection_view_performance(duplicate)


def test_selection_view_performance_preserves_logical_union_memberships() -> None:
    observations = _observations()
    shared_membership = observations.loc[observations["candidate_id"].eq("candidate-a")].copy()
    shared_membership["selected_for_view_id"] = "ciprofloxacin"

    result = selection_view_performance(pd.concat((observations, shared_membership), ignore_index=True))

    assert len(result.observations) == 10
    summary = result.summary.set_index(["objective_view_id", "selected_for_view_id"])
    assert summary.loc[("ethanol", "ciprofloxacin"), "candidate_count"] == 3


def test_selection_view_performance_requires_one_view_universe_across_rounds() -> None:
    second_round = (
        _observations()
        .loc[
            lambda frame: frame["candidate_id"].isin(["candidate-a", "candidate-b"])
            & frame["objective_view_id"].eq("ethanol")
        ]
        .copy()
    )
    second_round["observed_round"] = 2

    with pytest.raises(ValueError, match="same objective and selection views across rounds"):
        selection_view_performance(pd.concat((_observations(), second_round), ignore_index=True))


def test_selection_view_performance_renders_a_publication_figure(tmp_path: Path) -> None:
    output = tmp_path / "selection-view-performance.png"

    render_selection_view_performance(
        _observations(),
        output_path=output,
        title="Observed performance by selection view",
        objective_value_label="Observed objective score",
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_selection_view_performance_uses_square_large_type_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "selection-view-performance.svg"
    pyplot = importlib.import_module("matplotlib.pyplot")
    close_figure = pyplot.close
    closed_figures: list[Figure] = []
    monkeypatch.setattr(pyplot, "close", closed_figures.append)

    render_selection_view_performance(
        _observations(),
        output_path=output,
        title="The selected cohort performs best under its intended objective",
        objective_value_label="Observed assay score",
        view_labels={"ethanol": "Ethanol", "ciprofloxacin": "Ciprofloxacin"},
    )

    assert len(closed_figures) == 1
    figure = closed_figures[0]
    panels = figure.axes
    assert len(panels) == 2
    assert all(axis.get_box_aspect() == pytest.approx(1.0) for axis in panels)
    assert figure._suptitle is not None
    assert figure._suptitle.get_fontsize() >= 22
    assert figure._suptitle.get_position()[0] == pytest.approx(0.5)
    assert all(axis.title.get_fontsize() >= 18 for axis in panels)
    assert all(axis.xaxis.label.get_fontsize() >= 16 for axis in panels)
    assert all(axis.get_xlabel() == "Observed assay score" for axis in panels)
    assert min(label.get_fontsize() for axis in panels for label in axis.get_yticklabels()) >= 14
    close_figure(figure)


def test_selection_view_performance_uses_neutral_round_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "selection-view-performance.svg"
    pyplot = importlib.import_module("matplotlib.pyplot")
    close_figure = pyplot.close
    closed_figures: list[Figure] = []
    monkeypatch.setattr(pyplot, "close", closed_figures.append)

    render_selection_view_performance(
        _observations(),
        output_path=output,
        objective_value_label="Observed objective score",
    )

    assert {axis.get_title().splitlines()[-1] for axis in closed_figures[0].axes} == {"Round 1"}
    close_figure(closed_figures[0])


def test_selection_view_performance_uses_candidate_neutral_legend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "selection-view-performance.svg"
    pyplot = importlib.import_module("matplotlib.pyplot")
    close_figure = pyplot.close
    closed_figures: list[Figure] = []
    monkeypatch.setattr(pyplot, "close", closed_figures.append)

    render_selection_view_performance(
        _observations(),
        output_path=output,
        objective_value_label="Observed objective score",
    )

    assert [text.get_text() for text in closed_figures[0].legends[0].get_texts()] == [
        "Measured candidate",
        "Cohort median",
    ]
    close_figure(closed_figures[0])


def test_selection_view_performance_svg_is_deterministic_and_has_no_date(tmp_path: Path) -> None:
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"

    for output in (first, second):
        render_selection_view_performance(
            _observations(),
            output_path=output,
            title="The selected cohort performs best under its intended objective",
            objective_value_label="Observed objective score",
        )

    assert first.read_bytes() == second.read_bytes()
    assert "dc:date" not in first.read_text(encoding="utf-8")
