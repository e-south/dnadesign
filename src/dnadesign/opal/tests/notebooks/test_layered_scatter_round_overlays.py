"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_layered_scatter_round_overlays.py

Tests manifest-backed round overlays for layered notebook scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.analysis.notebook_components.layered_scatter import (
    build_notebook_layered_scatter_contract,
    build_notebook_layered_scatter_controls,
    filter_notebook_layered_scatter_rows,
)
from dnadesign.opal.src.analysis.notebook_components.layered_scatter_rendering import (
    render_layered_scatter_figure,
)
from dnadesign.opal.src.core.utils import file_sha256
from dnadesign.opal.tests.notebooks.test_layered_scatter_notebook import _choice, _marker_vertices


def test_layered_scatter_contract_loads_exact_selected_cohorts_for_every_round(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    for choice, selected_id in ((round_zero, "selected-r0"), (round_one, "selected-r1")):
        manifest = choice["manifest"]
        tidy_path = Path(str(manifest["tidy_csv"]))
        tidy = pd.read_csv(tidy_path)
        tidy.loc[tidy["selected"].eq(True), "id"] = selected_id
        tidy.to_csv(tidy_path, index=False)
        manifest["outputs"][0]["sha256"] = file_sha256(tidy_path)

    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})

    assert contract is not None
    assert contract["active_selection_round"] == 1
    assert contract["selection_rounds"] == [0, 1]
    assert contract["selection_rows"][["id", "__notebook_selection_round"]].to_dict("records") == [
        {"id": "selected-r0", "__notebook_selection_round": 0},
        {"id": "selected-r1", "__notebook_selection_round": 1},
    ]


def test_layered_scatter_round_overlay_unions_round_specific_display_limits(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    round_zero["manifest"]["artifact_metadata"]["notebook_view"]["x_limits"] = [-2.0, 3.0]
    round_zero["manifest"]["artifact_metadata"]["notebook_view"]["y_limits"] = [-3.0, 4.0]

    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})

    assert contract is not None
    assert contract["runtime"]["x_limits"] == [-2.0, 3.0]
    assert contract["runtime"]["y_limits"] == [-3.0, 4.0]
    assert contract["selection_rounds"] == [0, 1]


def test_layered_scatter_round_overlay_uses_one_shared_color_extent(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    round_zero["manifest"]["artifact_metadata"]["notebook_view"]["color_scale"]["extent"] = 2.5

    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})

    assert contract is not None
    assert contract["runtime"]["color_scale"]["extent"] == pytest.approx(2.5)
    assert contract["runtime"]["color_scale"]["context"] == (
        "shared across loaded rounds; endpoint values remain in the plotted data"
    )
    assert contract["selection_rounds"] == [0, 1]


def test_layered_scatter_round_overlay_derives_one_combined_color_disclosure(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    round_zero_color = round_zero["manifest"]["artifact_metadata"]["notebook_view"]["color_scale"]
    round_zero_color["context"] += "; 1 visible value saturates at the color endpoint"
    round_zero_color["extend"] = "max"

    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})

    assert contract is not None
    shared_color = contract["runtime"]["color_scale"]
    assert shared_color["context"] == "shared across loaded rounds; endpoint values remain in the plotted data"
    assert shared_color["extend"] == "neither"


def test_layered_scatter_can_overlay_selected_cohorts_categorically_by_round(tmp_path: Path) -> None:
    round_zero = _choice(tmp_path, filename="frontier_r0.csv")
    round_one = _choice(tmp_path, filename="frontier_r1.csv")
    round_zero["manifest"].update({"run_id": "r0", "rounds": [0]})
    round_one["manifest"].update({"run_id": "r1", "rounds": [1]})
    for choice, selected_id in ((round_zero, "selected-r0"), (round_one, "selected-r1")):
        manifest = choice["manifest"]
        tidy_path = Path(str(manifest["tidy_csv"]))
        tidy = pd.read_csv(tidy_path)
        tidy.loc[tidy["selected"].eq(True), "id"] = selected_id
        tidy.to_csv(tidy_path, index=False)
        manifest["outputs"][0]["sha256"] = file_sha256(tidy_path)
    contract = build_notebook_layered_scatter_contract({**round_one, "scope_options": [round_zero, round_one]})
    assert contract is not None

    visible = filter_notebook_layered_scatter_rows(
        contract["rows"],
        contract=contract,
        state={
            "show_prediction_pool": False,
            "show_selected": True,
            "selection_rounds": [0, 1],
            "observed_batches": [],
            "label_scope": "none",
        },
    )
    figure = render_layered_scatter_figure(visible, contract=contract)
    round_one_visible = filter_notebook_layered_scatter_rows(
        contract["rows"],
        contract=contract,
        state={
            "show_prediction_pool": False,
            "show_selected": True,
            "selection_rounds": [1],
            "observed_batches": [],
            "label_scope": "none",
        },
    )
    round_one_figure = render_layered_scatter_figure(round_one_visible, contract=contract)
    try:
        assert visible[["id", "__notebook_selection_round"]].to_dict("records") == [
            {"id": "selected-r0", "__notebook_selection_round": 0},
            {"id": "selected-r1", "__notebook_selection_round": 1},
        ]
        assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
            "Selected for Round 0 (n=1)",
            "Selected for Round 1 (n=1)",
        ]
        np.testing.assert_allclose(
            _marker_vertices(figure, "Selected for Round 1"),
            _marker_vertices(round_one_figure, "Selected for Round 1"),
        )
    finally:
        plt.close(figure)
        plt.close(round_one_figure)


def test_layered_scatter_round_markers_remain_distinct_after_eight_rounds(tmp_path: Path) -> None:
    choice = _choice(tmp_path)
    contract = build_notebook_layered_scatter_contract(choice)
    assert contract is not None
    selected = contract["selection_rows"].iloc[[0]].copy()
    round_zero = selected.assign(id="selected-zero", __notebook_selection_round=0)
    round_eight = selected.assign(id="selected-eight", __notebook_selection_round=8)
    rows = pd.concat([round_zero, round_eight], ignore_index=True)
    rows.attrs.update(show_prediction_pool=False, show_selected=True)
    contract["selection_rounds"] = list(range(9))

    figure = render_layered_scatter_figure(rows, contract=contract)
    try:
        assert not np.array_equal(
            _marker_vertices(figure, "Selected for Round 0"),
            _marker_vertices(figure, "Selected for Round 8"),
        )
    finally:
        plt.close(figure)


def test_layered_scatter_controls_offer_exact_manifest_backed_selection_rounds() -> None:
    class _Ui:
        @staticmethod
        def dropdown(_options, *, value, **_kwargs):
            return SimpleNamespace(value=value)

        @staticmethod
        def switch(*, value, **_kwargs):
            return SimpleNamespace(value=value)

        @staticmethod
        def multiselect(options, *, value, **_kwargs):
            return SimpleNamespace(value=[options[item] for item in value])

    memory_state: dict[str, object] = {}
    controls = build_notebook_layered_scatter_controls(
        {
            "key": "plot",
            "interactive": {"adapter": "three_axis_scatter_v1"},
            "active_selection_round": 1,
            "selection_rounds": [0, 1],
            "observed_batches": [],
        },
        memory=lambda: memory_state,
        set_memory=lambda value: memory_state.update(value),
        mo=SimpleNamespace(ui=_Ui()),
    )

    assert controls["selection_rounds"].value == [1]
