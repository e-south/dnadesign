"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_three_axis_scatter_notebook.py

Test contracts for generic interactive three-axis notebook scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest


def _contract() -> dict[str, object]:
    return {
        "adapter": "layered_scatter_v1",
        "key": "layered_scatter_v1:fixture",
        "view": {
            "record_kind_column": "record_kind",
            "prediction_value": "prediction",
            "observed_value": "observed_label",
            "selection_column": "selected",
            "batch_column": "batch_key",
            "label_column": "display_label",
            "x_column": "response_family_score",
            "y_column": "on_signal_family_score",
            "color_column": "off_signal_suppression_family_score",
        },
        "runtime": {
            "title": "Multistate response behavior · Example view",
            "context": "Target ON: State B, State D | OFF: State A, State C",
            "x_label": r"Response-ordering family score, $S_R$",
            "y_label": r"Intended-ON signal family score, $S_{\mathrm{ON}}$",
            "color_label": r"Intended-OFF suppression family score, $S_{\mathrm{OFF}}$",
        },
        "interactive": {
            "adapter": "three_axis_scatter_v1",
            "score_column": "behavior_score",
            "score_label": r"Behavior score, $S_{\mathrm{MSRB}}$",
            "prediction_sample_limit": 8_000,
            "sampling_method": "sha256_id_v1",
        },
        "observed_batches": [{"id": "batch_0", "label": "Batch 0"}],
        "rows": _rows(),
    }


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["pool-a", "selected-b", "observed-c"],
            "record_kind": ["prediction", "prediction", "observed_label"],
            "selected": [False, True, False],
            "batch_key": [None, None, "batch_0"],
            "display_label": [None, "Selected B", "Observed C"],
            "response_family_score": [0.2, 1.1, -0.4],
            "on_signal_family_score": [0.4, 1.2, 0.5],
            "off_signal_suppression_family_score": [-0.1, 0.9, 0.3],
            "behavior_score": [0.1, 1.0, 0.0],
        }
    )


def test_three_axis_figure_uses_exact_family_axes_and_campaign_layers() -> None:
    from dnadesign.opal.src.analysis.notebook_components.three_axis_scatter import (
        build_notebook_three_axis_scatter_figure,
    )

    figure = build_notebook_three_axis_scatter_figure(_rows(), contract=_contract())

    assert [trace.type for trace in figure.data] == ["scatter3d", "scatter3d", "scatter3d"]
    assert [trace.name for trace in figure.data] == [
        "Deterministic prediction sample (n=1 of 1)",
        "Selected (n=1)",
        "Observed · Batch 0 (n=1)",
    ]
    assert list(figure.data[0].x) == [0.2]
    assert list(figure.data[0].y) == [0.4]
    assert list(figure.data[0].z) == [-0.1]
    assert figure.layout.scene.xaxis.title.text == r"Response-ordering family score, $S_R$"
    assert figure.layout.scene.yaxis.title.text == r"Intended-ON signal family score, $S_{\mathrm{ON}}$"
    assert figure.layout.scene.zaxis.title.text == r"Intended-OFF suppression family score, $S_{\mathrm{OFF}}$"
    assert figure.layout.title.x == pytest.approx(0.5)
    assert float(figure.layout.title.y) <= 0.96
    assert figure.layout.paper_bgcolor == "white"
    assert figure.layout.scene.bgcolor == "white"
    assert figure.layout.font.size >= 14
    assert figure.layout.scene.xaxis.tickfont.size >= 12
    assert figure.layout.scene.yaxis.tickfont.size >= 12
    assert figure.layout.scene.zaxis.tickfont.size >= 12


def test_three_axis_hover_identity_is_ledger_backed() -> None:
    from dnadesign.opal.src.analysis.notebook_components.three_axis_scatter import (
        build_notebook_three_axis_scatter_figure,
    )

    figure = build_notebook_three_axis_scatter_figure(_rows(), contract=_contract())
    customdata = list(figure.data[1].customdata[0])

    assert customdata == ["selected-b", "Selected B", 1.0]
    assert "Candidate: %{customdata[0]}" in figure.data[1].hovertemplate
    assert "Behavior score" in figure.data[1].hovertemplate
    assert figure.layout.clickmode is None


def test_three_axis_widget_uses_marimo_plotly_happy_path() -> None:
    from dnadesign.opal.src.analysis.notebook_components.three_axis_scatter import (
        render_notebook_three_axis_scatter,
    )

    captured: dict[str, object] = {}

    class _Ui:
        @staticmethod
        def plotly(figure, **kwargs):
            captured["figure"] = figure
            captured["kwargs"] = kwargs
            return SimpleNamespace(points=[])

    class _Mo:
        ui = _Ui()

        @staticmethod
        def md(text):
            return {"kind": "md", "text": text}

        @staticmethod
        def vstack(items, *, gap):
            return {"kind": "vstack", "items": items, "gap": gap}

    widget = render_notebook_three_axis_scatter(
        _rows(),
        contract=_contract(),
        mo=_Mo(),
    )

    assert widget["items"][0].points == []
    assert "deterministic SHA-256-ID sample" in widget["items"][1]["text"]
    assert captured["figure"].data[0].type == "scatter3d"
    assert captured["kwargs"] == {
        "config": {
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
        "label": "Interactive three-family candidate landscape",
    }


def test_three_axis_renderer_requires_an_explicit_adapter_contract() -> None:
    from dnadesign.opal.src.analysis.notebook_components.three_axis_scatter import (
        build_notebook_three_axis_scatter_figure,
    )

    contract = _contract()
    contract["interactive"] = {}

    with pytest.raises(ValueError, match="three_axis_scatter_v1"):
        build_notebook_three_axis_scatter_figure(_rows(), contract=contract)


def test_three_axis_sampling_retains_all_selected_and_observed_rows() -> None:
    from dnadesign.opal.src.analysis.notebook_components.three_axis_scatter import (
        sample_notebook_three_axis_rows,
    )

    background_template = _rows().iloc[0].to_dict()
    background = pd.DataFrame([{**background_template, "id": f"pool-{index}"} for index in range(5)])
    rows = pd.concat([background, _rows().iloc[1:]], ignore_index=True)
    contract = _contract()
    contract["interactive"]["prediction_sample_limit"] = 2

    first = sample_notebook_three_axis_rows(rows, contract=contract)
    second = sample_notebook_three_axis_rows(
        rows.sample(frac=1.0, random_state=17),
        contract=contract,
    )

    assert set(first.loc[first["record_kind"].eq("prediction") & first["selected"], "id"]) == {"selected-b"}
    assert set(first.loc[first["record_kind"].eq("observed_label"), "id"]) == {"observed-c"}
    assert set(first["id"]) == set(second["id"])
    assert first.attrs["complete_background_count"] == 5
    assert first.attrs["displayed_background_count"] == 2


def test_three_axis_generated_cells_route_mode_and_selected_baserender_companion() -> None:
    from dnadesign.opal.src.analysis.notebook_set_template.layered_scatter_cells import (
        render_layered_scatter_cells,
    )
    from dnadesign.opal.src.analysis.notebook_set_template.visual_panel_cells import (
        render_visual_panel_cell,
    )

    text = render_layered_scatter_cells()
    panel = render_visual_panel_cell()

    assert 'scatter_figure_ui = layered_scatter_controls["figure"]' in text
    assert '"figure": scatter_figure_ui' in text
    assert "baserender_record_selector=baserender_record_selector" in panel
    assert "baserender_record_row=baserender_record_row" in panel
    assert "baserender_selection_record=baserender_selection_record" in panel
