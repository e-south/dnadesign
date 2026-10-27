"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_notebook_review_controls.py

Tests the semantic layout and visibility contracts for notebook review controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dnadesign.opal.src.analysis.notebook_components import render_notebook_review_control_surface


class _ControlSurfaceFakeMo:
    def hstack(self, items: list[object], **_: object) -> dict[str, object]:
        return {"kind": "hstack", "items": items}

    def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
        return {"kind": "vstack", "items": items, "gap": gap}


def test_notebook_review_control_surface_groups_plot_controls_by_semantics() -> None:
    mo = _ControlSurfaceFakeMo()
    rendered = render_notebook_review_control_surface(
        active_view_mode="Campaign",
        campaign_ui="campaign",
        selection_view_ui="selection-view",
        view_mode_ui="view",
        visual_group_ui="section",
        plot_ui="plot",
        plot_scope_ui="plot-scope",
        reader_evidence_artifact_ui="reader-artifact",
        selected_visual_choice={"surface_kind": "plot"},
        mo=mo,
    )

    assert rendered == {
        "kind": "vstack",
        "items": [
            {"kind": "hstack", "items": ["campaign", "selection-view", "view"]},
            {"kind": "hstack", "items": ["section", "plot"]},
            {"kind": "hstack", "items": ["plot-scope"]},
        ],
        "gap": 0.45,
    }


def test_notebook_review_control_surface_aligns_semantic_rows_to_equal_grid() -> None:
    class _GridMo:
        def hstack(self, items: list[object], **kwargs: object) -> dict[str, object]:
            return {"kind": "hstack", "items": items, "kwargs": kwargs}

        def vstack(self, items: list[object], **kwargs: object) -> dict[str, object]:
            return {"kind": "vstack", "items": items, "kwargs": kwargs}

    rendered = render_notebook_review_control_surface(
        active_view_mode="Campaign",
        campaign_ui="campaign",
        selection_view_ui="selection-view",
        view_mode_ui="view",
        visual_group_ui="section",
        plot_ui="deliverable",
        plot_scope_ui="round-scope",
        layered_scatter_controls={
            "figure": SimpleNamespace(value="interactive_3d"),
            "prediction_pool": "prediction-pool",
            "selected": SimpleNamespace(value=True),
            "selection_rounds": "selected-rounds",
            "observed_batches": "observed-layer",
        },
        selected_visual_choice={"surface_kind": "plot"},
        mo=_GridMo(),
    )

    assert [[*row["items"]] for row in rendered["items"]] == [
        ["campaign", "selection-view", "view"],
        ["section", "deliverable"],
        ["round-scope"],
        [
            SimpleNamespace(value="interactive_3d"),
            "prediction-pool",
            SimpleNamespace(value=True),
        ],
        ["selected-rounds", "observed-layer"],
    ]
    assert all(
        row["kwargs"]
        == {
            "justify": "start",
            "align": "end",
            "wrap": True,
            "gap": 0.5,
            "widths": "equal",
        }
        for row in rendered["items"]
    )
    assert rendered["kwargs"] == {"gap": 0.45}


def test_notebook_review_control_surface_hides_selection_view_for_campaign_scoped_visual() -> None:
    rendered = render_notebook_review_control_surface(
        active_view_mode="Campaign",
        campaign_ui="campaign",
        selection_view_ui="selection-view",
        view_mode_ui="view",
        visual_group_ui="section",
        plot_ui="deliverable",
        selected_visual_choice={
            "surface_kind": "selection_batch",
            "selection_scope": "campaign",
        },
        mo=_ControlSurfaceFakeMo(),
    )

    assert [row["items"] for row in rendered["items"]] == [
        ["campaign", "view", "section"],
        ["deliverable"],
    ]


def test_notebook_review_control_surface_groups_reader_controls_with_deliverable() -> None:
    mo = _ControlSurfaceFakeMo()
    rendered = render_notebook_review_control_surface(
        active_view_mode="Campaign",
        campaign_ui="campaign",
        view_mode_ui="view",
        visual_group_ui="section",
        plot_ui="deliverable",
        reader_evidence_artifact_ui="reader-artifact",
        selected_visual_choice={"surface_kind": "reader_evidence"},
        mo=mo,
    )

    assert rendered == {
        "kind": "vstack",
        "items": [
            {"kind": "hstack", "items": ["campaign", "view", "section"]},
            {"kind": "hstack", "items": ["deliverable"]},
            {"kind": "hstack", "items": ["reader-artifact"]},
        ],
        "gap": 0.45,
    }


def test_notebook_review_control_surface_only_shows_baserender_scope_controls_with_choices() -> None:
    mo = _ControlSurfaceFakeMo()
    singleton_scope = render_notebook_review_control_surface(
        active_view_mode="Campaign",
        selection_view_ui="selection-view",
        visual_group_ui="section",
        plot_ui="deliverable",
        baserender_round_ui=None,
        baserender_run_ui=None,
        baserender_record_selector="selected-sequence",
        selected_visual_choice={"surface_kind": "baserender"},
        mo=mo,
    )
    multiple_scope = render_notebook_review_control_surface(
        active_view_mode="Campaign",
        selection_view_ui="selection-view",
        visual_group_ui="section",
        plot_ui="deliverable",
        baserender_selection_view_ui="render-selection-view",
        baserender_round_ui="selection-round",
        baserender_run_ui="selection-run",
        baserender_record_selector="selected-sequence",
        selected_visual_choice={"surface_kind": "baserender"},
        mo=mo,
    )

    assert [row["items"] for row in singleton_scope["items"]] == [
        ["selection-view", "section", "deliverable"],
        ["selected-sequence"],
    ]
    assert [row["items"] for row in multiple_scope["items"]] == [
        ["selection-view", "section", "deliverable"],
        ["render-selection-view", "selection-round", "selection-run"],
        ["selected-sequence"],
    ]


def test_notebook_review_control_surface_excludes_sequence_lookup_from_three_axis_controls() -> None:
    class _ResponsiveMo:
        def hstack(self, items: list[object], **kwargs: object) -> dict[str, object]:
            return {"kind": "hstack", "items": items, "kwargs": kwargs}

        def vstack(self, items: list[object], **kwargs: object) -> dict[str, object]:
            return {"kind": "vstack", "items": items, "kwargs": kwargs}

    figure_control = SimpleNamespace(value="interactive_3d")
    rendered = render_notebook_review_control_surface(
        active_view_mode="Campaign",
        selection_view_ui="selection-view",
        visual_group_ui="section",
        plot_ui="deliverable",
        layered_scatter_controls={
            "figure": figure_control,
            "prediction_pool": "prediction-pool",
            "selected": "selected-layer",
            "observed_batches": "observed-layer",
        },
        baserender_record_selector="selected-sequence",
        selected_visual_choice={"surface_kind": "plot"},
        mo=_ResponsiveMo(),
    )

    assert [row["items"] for row in rendered["items"]] == [
        ["selection-view", "section", "deliverable"],
        [figure_control, "prediction-pool", "selected-layer"],
        ["observed-layer"],
    ]
    assert all(row["kwargs"]["widths"] == "equal" for row in rendered["items"])


def test_notebook_review_control_surface_rejects_unknown_view_mode() -> None:
    with pytest.raises(ValueError, match="active_view_mode must be 'Campaign' or 'Campaign set'"):
        render_notebook_review_control_surface(
            active_view_mode="Legacy",
            campaign_ui="campaign",
            mo=_ControlSurfaceFakeMo(),
        )
