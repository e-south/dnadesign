"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_notebook_template.py

Tests OPAL generated notebook template behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import ast
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.opal.src.analysis.notebook_components import (
    CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND,
    SELECTION_BATCH_SURFACE_KIND,
    annotate_notebook_visual_choices,
    build_notebook_artifact_garden_lines,
    build_notebook_artifact_garden_rows,
    build_notebook_at_a_glance_rows,
    build_notebook_baserender_candidate_catalog,
    build_notebook_baserender_contract,
    build_notebook_baserender_contract_rows,
    build_notebook_baserender_panel_title,
    build_notebook_baserender_record_annotation_counts,
    build_notebook_baserender_record_choices,
    build_notebook_baserender_record_choices_with_counts,
    build_notebook_baserender_record_controls,
    build_notebook_baserender_record_memory_key,
    build_notebook_baserender_record_options,
    build_notebook_baserender_selector_model,
    build_notebook_campaign_header_lines,
    build_notebook_campaign_set_metric_comparison_rows,
    build_notebook_campaign_set_selection_overlap_card_rows,
    build_notebook_campaign_set_selection_overlap_choice,
    build_notebook_campaign_summary_row,
    build_notebook_change_lines,
    build_notebook_change_rows,
    build_notebook_collection_baserender_role_choices,
    build_notebook_collection_set_choices,
    build_notebook_collection_visual_choices,
    build_notebook_collection_visual_description,
    build_notebook_evidence_rows,
    build_notebook_metric_definition_rows,
    build_notebook_no_plot_scope_rows,
    build_notebook_plot_card_rows,
    build_notebook_plot_inventory_rows,
    build_notebook_plot_method_rows,
    build_notebook_plot_method_sections,
    build_notebook_plot_scope_options,
    build_notebook_reader_evidence_artifact_rows,
    build_notebook_run_summary_lines,
    build_notebook_selected_baserender_record_sets,
    build_notebook_selected_baserender_records,
    build_notebook_selection_batch_choice,
    build_notebook_selection_batch_rows,
    build_notebook_selection_batch_summary_rows,
    build_notebook_selection_view_options,
    build_notebook_validity_lines,
    build_notebook_visual_group_options,
    build_notebook_visual_surface_model,
    filter_notebook_visual_choices_by_group,
    has_notebook_baserender_record_options,
    load_notebook_baserender_record_row,
    render_notebook_baserender_record,
    render_notebook_campaign_set_metric_comparison_image,
    render_notebook_campaign_set_selection_overlap_image,
    render_notebook_review_control_surface,
    render_notebook_visual_panel,
    render_visual_surface_cells,
    resolve_notebook_baserender_preferred_record_id,
    resolve_notebook_baserender_selection_batch_scope,
    resolve_notebook_selection_view,
    select_notebook_baserender_default_record_id,
    select_notebook_plot_scope,
)
from dnadesign.opal.src.analysis.notebook_components.plot_text import plot_alt_text, plot_math_description
from dnadesign.opal.src.analysis.notebook_components.visual_panel_baserender import _candidate_alt_suffix
from dnadesign.opal.src.analysis.notebook_set_template import render_campaign_set_notebook
from dnadesign.opal.src.analysis.notebook_template import render_campaign_notebook
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.registries.plots import describe_plot_kind, get_plot, list_plot_kinds


class _ControlSurfaceFakeMo:
    def hstack(self, items: list[object], **_: object) -> dict[str, object]:
        return {"kind": "hstack", "items": items}

    def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
        return {"kind": "vstack", "items": items, "gap": gap}


def test_notebook_template_data_source_options() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert 'label="Campaign"' in text
    assert 'label="Round"' not in text
    assert "predictions (selected run)" not in text
    assert "labels (all rounds)" not in text


def test_notebook_template_uses_medium_width() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert text.startswith("import marimo\n\n__generated_with = ")
    assert "\n# fmt: off\n# ruff: noqa\n" in text[:160]
    assert 'marimo.App(width="medium")' in text
    assert 'marimo.App(width="full")' not in text


def test_notebook_template_uses_one_compact_title_and_disclosed_campaign_context() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    context_cell = text.split("selected_campaign_context_panel =", maxsplit=1)[1].split("_overview_rows", maxsplit=1)[0]

    assert "# OPAL Review Notebook" not in text
    assert "selected_campaign_brief_md" not in text
    assert text.count("selected_campaign_title_md = mo.md(_header_lines[0])") == 1
    assert "heading_level=1" in text
    assert "selected_campaign_title_md = mo.md(_header_lines[0])" in text
    assert '"Campaign context": mo.md("\\n".join(_header_lines[2:]))' in text
    assert "selected_campaign_context_panel = mo.accordion(" in text
    assert "multiple=" not in context_cell
    assert "selected_collection_set_title_md" in text
    assert "else selected_campaign_title_md" in text


def test_notebook_campaign_summary_exposes_masked_objective_target_partition() -> None:
    selection_view = {
        "id": "ethanol",
        "objective": {
            "name": "response_magnitude_feasibility_v1",
            "params": {
                "state_ids": ["00", "10", "01", "11"],
                "target_mask": [0, 1, 0, 1],
            },
        },
        "selection": {
            "name": "top_n",
            "params": {"score_ref": "feasibility_margin", "objective_mode": "maximize"},
        },
    }
    view_model = {
        "campaign": {
            "slug": "rmf_ethanol",
            "selection_views": [selection_view],
        },
        "status": {},
    }

    target = "Response magnitude feasibility; maximize feasibility margin; ON=10, 11; OFF=00, 01"
    rows = build_notebook_at_a_glance_rows(view_model, selection_view=selection_view)
    assert {"field": "selection view", "value": "ethanol"} in rows
    assert {"field": "objective target", "value": target} in rows
    assert build_notebook_campaign_header_lines(view_model, selection_view=selection_view)[-1] == (
        f"**Objective target:** {target}."
    )


def test_notebook_campaign_summary_uses_declared_acronym_for_masked_objective() -> None:
    selection_view = {
        "id": "and",
        "objective": {
            "name": "multistate_response_behavior_v1",
            "params": {
                "state_ids": ["00", "10", "01", "11"],
                "target_mask": [0, 0, 0, 1],
                "softmin_scale": 1.0,
            },
        },
        "selection": {
            "name": "top_n",
            "params": {"score_ref": "behavior_score", "objective_mode": "maximize"},
        },
    }
    view_model = {
        "campaign": {
            "slug": "behavior_campaign",
            "metadata": {"metric_acronym": "MSRB"},
            "selection_views": [selection_view],
        },
        "status": {},
    }

    target = "Multistate response behavior (MSRB); maximize behavior score; ON=11; OFF=00, 10, 01"
    assert {"field": "objective target", "value": target} in build_notebook_at_a_glance_rows(
        view_model,
        selection_view=selection_view,
    )
    assert build_notebook_campaign_header_lines(view_model, selection_view=selection_view)[-1] == (
        f"**Objective target:** {target}."
    )

    view_model["campaign"]["selection_views"].append(
        {
            "id": "alternate",
            "objective": {
                "name": "response_magnitude_feasibility_v1",
                "params": {"state_ids": ["00", "11"], "target_mask": [0, 1]},
            },
        }
    )
    mixed_target = "Multistate response behavior; maximize behavior score; ON=11; OFF=00, 10, 01"
    assert {"field": "objective target", "value": mixed_target} in build_notebook_at_a_glance_rows(
        view_model,
        selection_view=selection_view,
    )


def test_notebook_template_removes_extra_tables() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "mo.ui.dataframe(summary_df)" not in text
    assert "mo.ui.dataframe(labels_df)" not in text
    assert "mo.ui.data_explorer(filtered_df)" not in text


def test_notebook_template_has_visual_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "campaign_ui = None if len(campaigns) == 1 else mo.ui.dropdown(" in text
    assert "collection_set_choices = [] if len(campaigns) == 1 else" in text
    assert 'label="Review scope"' in text
    assert "view_mode_ui = None" in text
    assert 'label="Review section"' in text
    assert 'label="Deliverable"' in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "Select one operative visual surface" not in text
    assert '"Plot deliverables": plot_panel' not in text


def test_notebook_template_initializes_mapped_view_dropdown_by_label() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "options=_labels" in text
    assert "value=_preferred_label" in text
    assert "selection_view_id_memory, set_selection_view_id_memory = mo.state(None)" in text
    assert "on_change=set_selection_view_id_memory" in text
    assert 'value=str(_views[0]["id"])' not in text


def test_notebook_selection_view_controls_resolve_exact_ids() -> None:
    view_model = {
        "campaign": {
            "selection_views": [
                {"id": "ethanol", "objective": {"name": "rmf"}},
                {"id": "and", "objective": {"name": "rmf"}},
            ]
        }
    }

    assert build_notebook_selection_view_options(view_model) == {"Ethanol": "ethanol", "AND": "and"}
    assert resolve_notebook_selection_view(view_model, "and")["id"] == "and"
    with pytest.raises(ValueError, match="must resolve exactly once"):
        resolve_notebook_selection_view(view_model, "missing")


def test_notebook_template_has_opal_schema_sentinel() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert '__opal_notebook_template_schema__ = "opal.generated_campaign_review_notebook.v6"' in text


def test_notebook_template_does_not_read_widget_values_in_definition_cells() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    for cell in text.split("@app.cell"):
        if " = mo.ui.dropdown(" in cell or " = mo.ui.switch(" in cell:
            assert ".value" not in cell


def test_notebook_template_uses_direct_visual_choice_and_layered_controls() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "selected_visual_choice" in text
    assert "layered_scatter_controls" in text


def test_notebook_deliverable_memory_is_scoped_by_review_section() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "visual_label_memory, set_visual_label_memory = mo.state({})" in text
    assert "selected_visual_group_label" in text
    assert '_memory_key = str(selected_visual_group_label or "ungrouped")' in text
    assert "_memory = dict(visual_label_memory())" in text
    assert "_preferred = _memory.get(_memory_key)" in text
    assert "def _remember_visual(value):" in text
    assert "on_change=_remember_visual" in text
    assert 'str((selected_visual_choice or {}).get("selection_scope") or "selection_view")' in text


def test_notebook_reader_record_memory_is_scoped_by_campaign_and_deliverable() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "reader_evidence_record_label_memory, set_reader_evidence_record_label_memory = mo.state({})" in text
    assert "render_notebook_reader_evidence_record_control(" in text
    assert 'campaign_slug=str((selected_campaign_model.get("campaign") or {}).get("slug") or "")' in text
    assert "selected_plot_type_label=selected_reader_evidence_plot_type_label" in text
    assert "memory=reader_evidence_record_label_memory" in text
    assert "set_memory=set_reader_evidence_record_label_memory" in text


def test_notebook_template_uses_visual_surface_component() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    surface_text = render_visual_surface_cells()

    assert render_visual_surface_cells.__module__.endswith(".notebook_components.visual_surface")
    assert "def render_visual_surface_cells" not in text
    assert surface_text not in text
    helper_text = Path("src/dnadesign/opal/src/analysis/notebook_components/visual_panel.py").read_text()
    assert "plot deliverables are available" in helper_text
    assert "build_notebook_no_plot_scope_rows" not in text
    assert "Current campaign and plot evidence" not in text
    assert 'label="Deliverable"' in text
    assert '"label": plot_choice["title"]' not in text
    assert "render_notebook_plot_choice_image" in text
    assert "render_notebook_visual_panel(" in text
    assert "Plot:" not in text
    assert "build_notebook_plot_method_sections" in text
    assert "thumbnail_gallery" not in text
    assert "plot_scope_controls" not in text
    assert "selected_visual_choice" in text


def test_notebook_template_centralizes_visual_control_surface() -> None:
    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )
    helper_text = Path("src/dnadesign/opal/src/analysis/notebook_components/reader_evidence.py").read_text()

    assert "render_notebook_review_control_surface(" in text
    assert "review_control_surface = render_notebook_review_control_surface(" in text
    assert "if review_control_surface is not None:" in text
    assert "visual_group_ui=visual_group_ui" in text
    assert 'label="Review section"' in text
    assert "_review_control_rows = []" not in text
    assert "_primary_control_items" not in text
    assert "_visual_control_items" not in text
    assert "_reader_control_items" not in text
    assert 'control_surface="external"' in text
    assert "_reader_plot_panel" not in text
    assert "_plot_items" not in text
    assert "_items.append(plot_panel)" in text
    assert "mo.hstack(_reader_controls" not in text
    assert "build_notebook_reader_evidence_visual_choices(" in text
    assert "render_notebook_reader_evidence_plot_type_control(" not in text
    assert 'label="Reader plot type"' not in text
    assert 'label="Reader record"' in helper_text


def test_notebook_template_keeps_three_axis_inspector_single_purpose() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    control_text = Path("src/dnadesign/opal/src/analysis/notebook_components/review_controls.py").read_text()
    sequence_panel_text = Path(
        "src/dnadesign/opal/src/analysis/notebook_components/visual_panel_baserender.py"
    ).read_text()
    visual_panel_text = Path("src/dnadesign/opal/src/analysis/notebook_components/visual_panel.py").read_text()
    layout_text = Path("src/dnadesign/opal/src/analysis/notebook_set_template/layout_cells.py").read_text()

    assert "baserender_record_selector=baserender_record_selector" in text
    assert 'baserender_record_selector if figure_mode == "interactive_3d" else None' not in control_text
    assert 'mo.hstack(controls, justify="start", align="end", wrap=True, gap=0.35)' in control_text
    assert "mo.hstack([baserender_record_selector]" not in sequence_panel_text
    assert "render_notebook_three_axis_sequence_companion" not in visual_panel_text
    assert 'in {"baserender", "campaign_set_baserender"}' in layout_text


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
        "kind": "hstack",
        "items": [
            "campaign",
            "selection-view",
            "view",
            "section",
            "plot",
            "plot-scope",
        ],
    }


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

    assert rendered["items"] == ["campaign", "view", "section", "deliverable"]


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
        "kind": "hstack",
        "items": ["campaign", "view", "section", "deliverable", "reader-artifact"],
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

    assert singleton_scope["items"] == ["selection-view", "section", "deliverable", "selected-sequence"]
    assert multiple_scope["items"] == [
        "selection-view",
        "section",
        "deliverable",
        "render-selection-view",
        "selection-round",
        "selection-run",
        "selected-sequence",
    ]


def test_notebook_review_control_surface_excludes_sequence_lookup_from_three_axis_controls() -> None:
    class _ResponsiveMo:
        def hstack(self, items: list[object], **kwargs: object) -> dict[str, object]:
            return {"kind": "hstack", "items": items, "kwargs": kwargs}

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

    assert rendered["items"] == [
        "selection-view",
        "section",
        "deliverable",
        figure_control,
        "prediction-pool",
        "selected-layer",
        "observed-layer",
    ]
    assert rendered["kwargs"] == {
        "justify": "start",
        "align": "end",
        "wrap": True,
        "gap": 0.35,
    }


def test_notebook_review_control_surface_rejects_unknown_view_mode() -> None:
    with pytest.raises(ValueError, match="active_view_mode must be 'Campaign' or 'Campaign set'"):
        render_notebook_review_control_surface(
            active_view_mode="Legacy",
            campaign_ui="campaign",
            mo=_ControlSurfaceFakeMo(),
        )


def test_notebook_visual_hierarchy_groups_without_hiding_deliverables() -> None:
    choices = annotate_notebook_visual_choices(
        [
            {"label": "Feature importance", "kind": "feature_importance_bars", "name": "feature_importance_bars"},
            {"label": "Reader time series", "surface_kind": "reader_evidence"},
            {"label": "Selected vector", "kind": "sfxi_vector_summary", "name": "selected_vec8_summary"},
            {"label": "Sequence render", "surface_kind": "baserender"},
            {
                "label": "Effect scaled vs logic fidelity",
                "kind": "fold_change_vs_logic_fidelity",
                "name": "effect_scaled_vs_logic_fidelity_latest",
            },
            {"label": "Setpoint sweep", "kind": "sfxi_setpoint_sweep", "name": "sfxi_setpoint_sweep_latest"},
            {
                "label": "Selection batch",
                "surface_kind": SELECTION_BATCH_SURFACE_KIND,
            },
        ]
    )

    assert build_notebook_visual_group_options(choices) == [
        "Decision review",
        "Assay evidence",
        "EDA comparisons",
        "Model diagnostics",
        "Method diagnostics",
        "Handoff",
    ]
    assert [choice["label"] for choice in filter_notebook_visual_choices_by_group(choices, "Decision review")] == [
        "Selected vector"
    ]
    assert [choice["label"] for choice in filter_notebook_visual_choices_by_group(choices, "EDA comparisons")] == [
        "Effect scaled vs logic fidelity"
    ]
    assert [choice["label"] for choice in filter_notebook_visual_choices_by_group(choices, "Handoff")] == [
        "Selection batch",
        "Sequence render",
    ]
    assert sum(
        len(filter_notebook_visual_choices_by_group(choices, group))
        for group in build_notebook_visual_group_options(choices)
    ) == len(choices)
    with pytest.raises(ValueError, match="Unknown review section"):
        filter_notebook_visual_choices_by_group(choices, "Legacy bucket")


def test_notebook_selection_batch_choice_preserves_view_memberships() -> None:
    payload = {
        "schema_version": "opal.selection_batch.v3",
        "as_of_round": 1,
        "run_id": "run-1",
        "deduplicate_by": "sequence",
        "allocation_strategy": "round_robin_next_best_unallocated",
        "unique_count": 2,
        "rows": [
            {
                "id": "candidate-a-with-a-long-stable-identifier",
                "sequence": "AAAA",
                "selection_view_ids": ["ethanol"],
                "selection_memberships": [
                    {
                        "selection_view_id": "ethanol",
                        "rank": 1,
                        "score": -0.2,
                        "allocation_slot": 1,
                        "selection_origin": "preferred_top_k",
                    },
                ],
                "preferred_view_ids": ["ethanol", "and"],
                "allocation_view_id": "ethanol",
                "allocation_slot": 1,
                "selection_batch_key": "AAAA",
                "deduplicate_by": "sequence",
            },
            {
                "id": "candidate-b",
                "sequence": "CCCC",
                "selection_view_ids": ["ciprofloxacin"],
                "selection_memberships": [
                    {
                        "selection_view_id": "ciprofloxacin",
                        "rank": 2,
                        "score": 1.3,
                        "allocation_slot": 2,
                        "selection_origin": "next_best_unallocated",
                    }
                ],
                "preferred_view_ids": [],
                "allocation_view_id": "ciprofloxacin",
                "allocation_slot": 2,
                "selection_batch_key": "CCCC",
                "deduplicate_by": "sequence",
            },
        ],
    }

    choice = build_notebook_selection_batch_choice(payload)

    assert choice["surface_kind"] == SELECTION_BATCH_SURFACE_KIND
    assert choice["selection_scope"] == "campaign"
    assert choice["label"] == "Selection batch proposal"
    assert choice["review_group"] == "handoff"
    assert build_notebook_selection_batch_rows(choice) == [
        {
            "candidate": "candidate-a-...entifier",
            "allocated view": "Ethanol",
            "competition rank": 1,
            "preferred by": "Ethanol, AND",
            "allocation origin": "Preferred top k",
            "view slot": 1,
        },
        {
            "candidate": "candidate-b",
            "allocated view": "Ciprofloxacin",
            "competition rank": 2,
            "preferred by": "None",
            "allocation origin": "Next best unallocated",
            "view slot": 2,
        },
    ]
    assert build_notebook_selection_batch_summary_rows(choice)[3:6] == [
        {
            "field": "row order",
            "value": "Allocation slot, then competition rank and view.",
        },
        {
            "field": "batch formation",
            "value": "Coordinated unique-slot allocation",
        },
        {"field": "deduplicated by", "value": "sequence"},
    ]


def test_notebook_selection_batch_orders_logical_union_by_competition_rank() -> None:
    payload = {
        "schema_version": "opal.selection_batch.v3",
        "as_of_round": 0,
        "run_id": "run-0",
        "deduplicate_by": "id",
        "allocation_strategy": "logical_union",
        "unique_count": 3,
        "rows": [
            {
                "id": candidate_id,
                "selection_view_ids": ["primary"],
                "selection_memberships": [
                    {
                        "selection_view_id": "primary",
                        "rank": rank,
                        "allocation_slot": None,
                        "selection_origin": "preferred_top_k",
                    }
                ],
                "preferred_view_ids": ["primary"],
                "allocation_view_id": None,
                "allocation_slot": None,
            }
            for candidate_id, rank in (("candidate-five", 5), ("candidate-one", 1), ("candidate-three", 3))
        ],
    }

    choice = build_notebook_selection_batch_choice(payload)

    assert [row["candidate"] for row in build_notebook_selection_batch_rows(choice)] == [
        "candidate-one",
        "candidate-three",
        "candidate-five",
    ]
    assert build_notebook_selection_batch_summary_rows(choice)[3:6] == [
        {
            "field": "row order",
            "value": "Competition rank, then view and candidate.",
        },
        {
            "field": "batch formation",
            "value": "Logical union of view selections",
        },
        {"field": "deduplicated by", "value": "id"},
    ]


def test_notebook_selection_batch_rejects_stale_presentation_contract() -> None:
    with pytest.raises(ValueError, match="requires schema 'opal.selection_batch.v3'"):
        build_notebook_selection_batch_choice(
            {
                "schema_version": "opal.selection_batch.v2",
                "deduplicate_by": "id",
                "allocation_strategy": "logical_union",
                "unique_count": 0,
                "rows": [],
            }
        )


def test_campaign_set_selection_overlap_choice_and_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes

    imshow_calls: list[dict[str, object]] = []
    original_imshow = Axes.imshow

    def _capture_imshow(self, *args, **kwargs):
        imshow_calls.append(dict(kwargs))
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)

    def _campaign(slug: str, rows: list[tuple[str, int, float]]) -> dict[str, object]:
        workdir = tmp_path / slug
        selection_dir = workdir / "outputs" / "rounds" / "round_0" / "selection"
        selection_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "selection_view_id": ["primary"] * len(rows),
                "id": [candidate_id for candidate_id, _, _ in rows],
                "as_of_round": [0] * len(rows),
                "run_id": ["r0"] * len(rows),
                "rank_competition": [rank for _, rank, _ in rows],
                "score": [score for _, _, score in rows],
                "selection_score": [score for _, _, score in rows],
                "score_ref": ["primary/sfxi"] * len(rows),
                "sequence": [f"{candidate_id}ACGT" for candidate_id, _, _ in rows],
            }
        ).write_parquet(selection_dir / "selections.parquet")
        return {
            "campaign": {
                "slug": slug,
                "name": f"SECG {slug} RF + SFXI",
                "workdir": str(workdir),
                "selection_views": [{"id": "primary"}],
            }
        }

    campaigns = [
        _campaign("ethanol", [("candidate_A_full", 1, 0.9), ("candidate_B_full", 2, 0.7)]),
        _campaign("cipro", [("candidate_C_full", 1, 0.8), ("candidate_A_full", 2, 0.75)]),
        _campaign("and", [("candidate_A_full", 1, 0.82), ("candidate_D_full", 2, 0.6)]),
    ]

    choice = build_notebook_campaign_set_selection_overlap_choice(campaigns, round_selector="latest")

    assert choice is not None
    assert choice["surface_kind"] == CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND
    assert choice["review_group"] == "EDA comparisons"
    assert choice["summary"] == {
        "campaign_count": 3,
        "slot_count": 6,
        "unique_candidate_count": 4,
        "shared_all_count": 1,
        "max_overlap": 3,
    }
    payload = render_notebook_campaign_set_selection_overlap_image(choice)
    assert payload is not None
    assert payload["image_bytes"].startswith(b"\x89PNG")
    assert imshow_calls[0]["aspect"] == "equal"
    assert payload["visual_contract"] == {
        "cell_geometry": "unit_square_cells",
        "cell_edges": "white",
        "minimum_tick_font_size_pt": 10,
    }
    assert "4 unique selected candidates" in payload["alt_text"]
    assert "1 candidates are selected by every campaign" in payload["alt_text"]
    assert build_notebook_campaign_set_selection_overlap_card_rows(choice) == [
        {"field": "campaigns", "value": 3},
        {"field": "selected slots", "value": 6},
        {"field": "unique candidates", "value": 4},
        {"field": "selected by every campaign", "value": 1},
        {"field": "max campaign overlap", "value": 3},
        {
            "field": "claim boundary",
            "value": "Overlap is a selection-policy diagnostic, not measured biological validation.",
        },
    ]


def test_notebook_visual_panel_rejects_unknown_control_surface() -> None:
    with pytest.raises(ValueError, match="control_surface must be 'inline' or 'external'"):
        render_notebook_visual_panel(
            active_view_mode="Campaign",
            build_notebook_plot_card_rows=lambda _: [],
            build_notebook_plot_method_sections=lambda _: {},
            mo=object(),
            opal_table=lambda *_, **__: None,
            pl=object(),
            render_notebook_plot_choice_image=lambda *_, **__: None,
            selected_visual_choice=None,
            select_notebook_plot_scope=lambda *_, **__: {},
            control_surface="legacy",
        )


def test_notebook_baserender_panel_titles_candidate_by_selection_view_and_rank() -> None:
    rendered: dict[str, object] = {}

    class _FakeMo:
        def md(self, text: str) -> dict[str, object]:
            return {"kind": "md", "text": text}

        def image(self, data: bytes, **kwargs: object) -> dict[str, object]:
            return {"kind": "image", "data": data, **kwargs}

        def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
            return {"kind": "vstack", "items": items, "gap": gap}

        def accordion(self, items: dict[str, object], **kwargs: object) -> dict[str, object]:
            return {"kind": "accordion", "items": items, **kwargs}

    panel = render_notebook_visual_panel(
        active_view_mode="Campaign",
        baserender_campaign_model={"campaign": {"slug": "secg_msrb_greedy"}},
        baserender_record_id="candidate-record-alpha-with-long-id",
        baserender_record_row={"id": "candidate-record-alpha-with-long-id", "sequence": "ACGT"},
        baserender_candidate_evidence={
            "record_id": "candidate-record-alpha-with-long-id",
            "active_selection_view_id": "and",
            "active_view_rank": 7,
            "selection_memberships": [{"selection_view_id": "and", "view_rank": 7}],
            "observed_rounds": [],
        },
        build_notebook_baserender_contract_rows=lambda _: [],
        build_notebook_baserender_label_rows=lambda *_, **__: [],
        build_notebook_plot_card_rows=lambda _: [],
        build_notebook_plot_method_sections=lambda _: {},
        control_surface="external",
        mo=_FakeMo(),
        opal_table=lambda *_, **__: {"kind": "table"},
        pl=pl,
        render_notebook_baserender_record=lambda *_, **kwargs: (
            rendered.update(kwargs)
            or {
                "record_id": "candidate-record-alpha-with-long-id",
                "image_bytes": b"png",
                "caption": "DenseGen TFBS annotation · 60 bp · 5 annotated elements",
                "alt_text": "DenseGen TFBS annotation",
            }
        ),
        render_notebook_plot_choice_image=lambda *_, **__: None,
        selected_baserender_round=0,
        selected_baserender_status_rows=(),
        selected_campaign_baserender_contract={"available": True},
        selected_campaign_labels_df=None,
        selected_visual_choice={"surface_kind": "baserender"},
        select_notebook_plot_scope=lambda *_, **__: {},
    )

    visual = panel["items"][0]
    assert visual["kind"] == "image"
    assert rendered["title"] == ("AND selection · competition rank 7 · candidate candidate-re...-long-id")
    assert visual["caption"] == "DenseGen TFBS annotation · 60 bp · 5 annotated elements"
    assert visual["alt"].endswith("Selected in campaign secg_msrb_greedy, round 0.")
    assert "candidate-record-alpha-with-long-id" not in str(rendered["title"])
    assert "Candidate and campaign evidence" in panel["items"][1]["items"]


@pytest.mark.parametrize(
    ("candidate_evidence", "message"),
    [
        ({"record_id": "", "selection_memberships": [], "observed_rounds": [0]}, "record_id"),
        (
            {
                "record_id": "candidate-1",
                "selection_memberships": [{"selection_view_id": "", "view_rank": 1}],
            },
            "membership",
        ),
        (
            {
                "record_id": "candidate-1",
                "selection_memberships": [{"selection_view_id": "ethanol", "view_rank": 0}],
            },
            "membership",
        ),
        ({"record_id": "candidate-1", "selection_memberships": [], "observed_rounds": []}, "selected or observed"),
    ],
)
def test_notebook_baserender_panel_title_rejects_incomplete_candidate_evidence(
    candidate_evidence: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_notebook_baserender_panel_title(candidate_evidence)


def test_notebook_baserender_panel_titles_observed_only_candidate() -> None:
    assert (
        build_notebook_baserender_panel_title(
            {
                "record_id": "candidate-record-alpha-with-long-id",
                "active_selection_view_id": "ethanol",
                "selection_memberships": [],
                "observed_rounds": [0, 2],
            }
        )
        == "Observed candidate · round 0, 2 · candidate-re...-long-id"
    )


@pytest.mark.parametrize(
    ("evidence", "selected_round", "expected"),
    [
        (
            {
                "selection_memberships": [{"selection_view_id": "ethanol", "view_rank": 1}],
                "observed_rounds": [],
            },
            None,
            " Selected in campaign fixture.",
        ),
        (
            {"selection_memberships": [], "observed_rounds": [0, 2]},
            3,
            " Observed in campaign fixture, rounds 0, 2.",
        ),
        (
            {
                "selection_memberships": [{"selection_view_id": "and", "view_rank": 4}],
                "observed_rounds": [0],
            },
            2,
            " Selected in campaign fixture, round 2. Observed in campaign fixture, round 0.",
        ),
    ],
)
def test_notebook_baserender_alt_text_matches_candidate_evidence_roles(
    evidence: dict[str, object],
    selected_round: int | None,
    expected: str,
) -> None:
    assert _candidate_alt_suffix(evidence, campaign_slug="fixture", selected_round=selected_round) == expected


def test_notebook_baserender_panel_rejects_record_identity_drift() -> None:
    class _FakeMo:
        def md(self, text: str) -> dict[str, object]:
            return {"kind": "md", "text": text}

        def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
            return {"kind": "vstack", "items": items, "gap": gap}

        def accordion(self, items: dict[str, object], **kwargs: object) -> dict[str, object]:
            return {"kind": "accordion", "items": items, **kwargs}

    with pytest.raises(ValueError, match="authoritative campaign evidence"):
        render_notebook_visual_panel(
            active_view_mode="Campaign",
            baserender_record_id="candidate-expected",
            baserender_record_row={"id": "candidate-other", "sequence": "ACGT"},
            baserender_candidate_evidence={
                "record_id": "candidate-expected",
                "active_selection_view_id": "ethanol",
                "selection_memberships": [{"selection_view_id": "ethanol", "view_rank": 1}],
                "observed_rounds": [],
            },
            build_notebook_baserender_contract_rows=lambda _: [],
            build_notebook_baserender_label_rows=lambda *_, **__: [],
            build_notebook_plot_card_rows=lambda _: [],
            build_notebook_plot_method_sections=lambda _: {},
            control_surface="external",
            mo=_FakeMo(),
            opal_table=lambda *_, **__: {"kind": "table"},
            pl=pl,
            render_notebook_baserender_record=lambda *_, **__: pytest.fail("mismatched record was rendered"),
            render_notebook_plot_choice_image=lambda *_, **__: None,
            selected_baserender_status_rows=(),
            selected_campaign_baserender_contract={"available": True},
            selected_campaign_labels_df=None,
            selected_visual_choice={"surface_kind": "baserender"},
            select_notebook_plot_scope=lambda *_, **__: {},
        )


def test_three_axis_panel_is_not_appended_with_baserender_evidence() -> None:
    rendered: dict[str, object] = {}

    class _FakeMo:
        def md(self, text: str) -> dict[str, object]:
            return {"kind": "md", "text": text}

        def image(self, data: bytes, **kwargs: object) -> dict[str, object]:
            return {"kind": "image", "data": data, **kwargs}

        def vstack(self, items: list[object], *, gap: float) -> dict[str, object]:
            return {"kind": "vstack", "items": items, "gap": gap}

        def hstack(self, items: list[object], **kwargs: object) -> dict[str, object]:
            return {"kind": "hstack", "items": items, **kwargs}

        def accordion(self, items: dict[str, object], **kwargs: object) -> dict[str, object]:
            return {"kind": "accordion", "items": items, **kwargs}

        def callout(self, item: object, **kwargs: object) -> dict[str, object]:
            return {"kind": "callout", "item": item, **kwargs}

    selector = {"kind": "selected-candidate-selector"}
    panel = render_notebook_visual_panel(
        active_view_mode="Campaign",
        baserender_record_id="candidate-record-alpha-with-long-id",
        baserender_record_row={"id": "candidate-record-alpha-with-long-id", "sequence": "ACGT"},
        baserender_record_selector=selector,
        baserender_candidate_evidence={
            "record_id": "candidate-record-alpha-with-long-id",
            "active_selection_view_id": "ciprofloxacin",
            "selection_memberships": [{"selection_view_id": "ciprofloxacin", "view_rank": 3}],
            "observed_rounds": [],
        },
        build_notebook_plot_card_rows=lambda _: [],
        build_notebook_plot_method_sections=lambda _: {},
        control_surface="external",
        layered_scatter_contract={"interactive": {"adapter": "three_axis_scatter_v1"}},
        mo=_FakeMo(),
        opal_table=lambda *_, **__: {"kind": "table"},
        pl=pl,
        plot_view_state={"figure": "interactive_3d"},
        render_notebook_baserender_record=lambda *_, **kwargs: (
            rendered.update(kwargs)
            or {
                "record_id": "candidate-record-alpha-with-long-id",
                "image_bytes": b"png",
                "caption": "DenseGen TFBS annotation · 60 bp · 5 annotated elements",
                "alt_text": "DenseGen TFBS annotation",
            }
        ),
        render_notebook_plot_choice_image=lambda *_, **__: {"kind": "three-axis-plot"},
        selected_campaign_baserender_contract={"available": True},
        selected_visual_choice={"surface_kind": "plot", "label": "MSRB family landscape"},
        select_notebook_plot_scope=lambda choice, _scope: choice,
    )

    assert panel["items"][0] == {"kind": "three-axis-plot"}
    assert all(item.get("kind") != "image" for item in panel["items"] if isinstance(item, dict))
    assert rendered == {}


def test_notebook_template_does_not_hide_generic_plots_for_sfxi_campaigns() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "SFXI plots only" not in text
    assert "Non-SFXI plots only" not in text
    assert "plot_entries_filtered" not in text


def test_notebook_template_is_campaign_specific_accordion_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "build_notebook_campaign_header_lines" in text
    assert "`{cfg.campaign.slug}` uses" not in text
    assert "Campaign analysis command surface" not in text
    assert "mo.accordion(" in text
    for section in [
        "Campaigns at a glance",
        "Campaign status",
        "Data and evidence records",
    ]:
        assert section in text
    assert "Reader evidence records" not in text
    assert "Data inputs and artifacts" not in text
    assert "Warnings and stale artifacts" not in text
    assert "_items.append(plot_panel)" in text
    assert "_reader_plot_panel" not in text


def test_notebook_template_uses_public_opal_helpers() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "from dnadesign.opal.notebooks.api.generated import" in text
    assert "from dnadesign.opal.notebooks.api import" not in text
    assert "dnadesign.opal.src" not in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "build_notebook_campaign_summary_row" in text
    assert "build_notebook_label_staging_rows" in text
    assert "render_notebook_reader_evidence_artifact_visual" in text
    assert "render_notebook_reader_evidence_panel" in text
    assert "build_notebook_reader_evidence_visual_choices" in text
    assert "render_notebook_reader_evidence_plot_type_control" not in text
    assert "render_notebook_reader_evidence_record_control" in text
    assert "render_notebook_reader_evidence_time_control" not in text
    assert "render_notebook_review_control_surface" in text
    assert "build_notebook_visual_surface_model" in text
    assert "build_notebook_collection_set_choices" in text
    assert "build_notebook_collection_visual_choices" in text
    assert "selected_campaign_baserender_contract, selected_campaign_model," in text
    assert "build_notebook_collection_visual_card_rows" in text
    assert "build_notebook_visual_group_options" in text
    assert "filter_notebook_visual_choices_by_group" in text
    assert "build_notebook_campaign_set_selection_overlap_card_rows" in text
    assert "render_notebook_campaign_set_selection_overlap_image" in text
    assert "build_notebook_campaign_set_visual_choices" not in text
    assert "render_notebook_visual_panel" in text


def test_notebook_template_reader_evidence_cells_are_runtime_safe() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert 'reader_evidence_surface = _reader_evidence["surface"]' in text
    assert 'selected_visual_choice.get("surface_kind") == "reader_evidence"' in text
    assert "render_notebook_reader_evidence_plot_type_control(" not in text
    assert "render_notebook_reader_evidence_record_control(" in text
    assert "render_notebook_reader_evidence_time_control(" not in text
    assert "render_notebook_reader_evidence_artifact_visual(" in text
    assert "render_notebook_reader_evidence_panel(" in text
    assert "reader_evidence_time_ui" not in text
    assert 'label="Reader plot type"' not in text
    assert 'label="Reader plot instance"' not in text
    helper_text = Path("src/dnadesign/opal/src/analysis/notebook_components/reader_evidence.py").read_text()
    assert 'label="Reader plot type"' in helper_text
    assert 'label="Reader record"' in helper_text
    assert "mo.accordion(_accordion_items, multiple=True)" in text
    assert "mo.accordion(_accordion_items, multiple=True, lazy=True)" not in text
    assert "_table(_df(_metric_rows))" not in text
    assert "_table(_df(_change_rows))" not in text
    assert "_table(_df(_artifact_rows))" not in text
    assert "_table(_df(_artifact_summary_rows))" not in text


def test_notebook_template_degrades_without_runs() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    helper_text = Path("src/dnadesign/opal/src/analysis/notebook_components/visual_panel.py").read_text()

    assert "No OPAL plot deliverables are available" in helper_text
    assert "mo.stop(len(rounds) == 0" not in text


def test_notebook_template_can_pin_initial_run_scope() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="0", run_id="run-1")

    assert "selected_round_selector = '0'" in text
    assert "run_id='run-1'" in text
    summary = "\n".join(
        build_notebook_run_summary_lines(
            "run-1",
            {"as_of_round": 0, "selection__name": "top_n", "model__name": "rf"},
            "sfxi",
            selected_round=0,
            default_round=0,
            run_options=["run-0", "run-1"],
        )
    )
    assert "Run scope: selected round `0`, selected run `run-1`." in summary


def test_notebook_template_keeps_lateral_tools_out() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "dnadesign.baserender" not in text
    assert "densegen__visual" not in text
    assert "cluster__ldn_v1__umap_x" not in text
    assert "cluster__ldn_v1__umap_y" not in text
    assert "obj__logic_fidelity" not in text
    assert "obj__effect_raw" not in text
    assert "obj__effect_scaled" not in text


def test_campaign_set_metric_comparison_uses_campaign_metadata(tmp_path: Path) -> None:
    def _campaign(slug: str, group: str, values: list[float]) -> dict:
        workdir = tmp_path / slug
        plots_dir = workdir / "outputs" / "plots"
        plots_dir.mkdir(parents=True)
        tidy_path = plots_dir / "score_selected_over_rounds_rall.csv"
        tidy_path.write_text(
            "round,cohort,metric,summary,value\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,median,{value}"
                for round_index, value in enumerate(values)
            )
            + "\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,count,6" for round_index, _value in enumerate(values)
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "campaign": {
                "slug": slug,
                "workdir": str(workdir),
                "metadata": {
                    "label_oracle_kind": group,
                    "label_family_id": "densegen_plan_logic4",
                    "label_split_id": "random_id",
                    "probe_oracle_id": f"{group}_id",
                },
            },
            "plot_manifests": [
                {
                    "name": "score_selected_over_rounds",
                    "kind": "metric_over_rounds",
                    "status": "written",
                    "rounds": "all",
                    "tidy_csv": str(tidy_path),
                    "outputs": [{"role": "tidy_csv", "path": str(tidy_path), "exists": True}],
                }
            ],
        }

    campaigns = [
        _campaign("cipro_positive_random_id", "positive", [0.2, 0.5]),
        _campaign("cipro_null_random_id", "null", [0.1, 0.15]),
    ]

    visual_choices = build_notebook_collection_visual_choices(
        [
            {
                "label": "Selected score over rounds",
                "title": "Selected score over rounds",
                "source_plot_name": "score_selected_over_rounds",
                "surface_kind": "campaign_set_metric_comparison",
            }
        ]
    )
    assert visual_choices[0]["surface_kind"] == "campaign_set_metric_comparison"
    assert visual_choices[0]["source_plot_name"] == "score_selected_over_rounds"
    rows = build_notebook_campaign_set_metric_comparison_rows(
        campaigns,
        plot_name="score_selected_over_rounds",
        group_key="label_oracle_kind",
    )
    assert {row["group"] for row in rows} == {"positive", "null"}
    payload = render_notebook_campaign_set_metric_comparison_image(
        rows,
        title="Selected score over rounds",
        group_key="label_oracle_kind",
    )
    assert payload is not None
    assert payload["image_bytes"].startswith(b"\x89PNG")
    assert "Label source" in payload["alt_text"]
    assert "Selected n=6" in payload["alt_text"]
    mixed_rows = [
        {**row, "cohort": "all_pool" if row["campaign"] == "cipro_null_random_id" else row["cohort"]} for row in rows
    ]
    with pytest.raises(ValueError, match="one metric/cohort pair"):
        render_notebook_campaign_set_metric_comparison_image(
            mixed_rows,
            title="Selected score over rounds",
            group_key="label_oracle_kind",
        )


def test_campaign_set_metric_comparison_uses_relationship_pairs_for_iqr_band(tmp_path: Path) -> None:
    def _campaign(slug: str, group: str, seed: int, values: list[float]) -> dict:
        workdir = tmp_path / slug
        plots_dir = workdir / "outputs" / "plots"
        plots_dir.mkdir(parents=True)
        tidy_path = plots_dir / "score_selected_over_rounds_rall.csv"
        tidy_path.write_text(
            "round,cohort,metric,summary,value\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,median,{value}"
                for round_index, value in enumerate(values)
            )
            + "\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,q25,{value - 0.1}"
                for round_index, value in enumerate(values)
            )
            + "\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,q75,{value + 0.1}"
                for round_index, value in enumerate(values)
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "campaign": {
                "slug": slug,
                "metadata": {
                    "target": "cipro",
                    "label_oracle_kind": group,
                    "label_family_id": "densegen_plan_logic4",
                    "label_split_id": "random_id",
                    "seed": seed,
                },
            },
            "plot_manifests": [
                {
                    "name": "score_selected_over_rounds",
                    "kind": "metric_over_rounds",
                    "status": "written",
                    "rounds": "all",
                    "params": {
                        "y_axis": {
                            "scale_class": "densegen_plan_logic4_negative_mse",
                            "limits": [-0.25, 0.0],
                            "include_zero_tick": True,
                        }
                    },
                    "tidy_csv": str(tidy_path),
                }
            ],
        }

    campaigns = [
        _campaign("cipro_positive_s7", "positive", 7, [0.2, 0.4]),
        _campaign("cipro_null_s7", "null", 7, [0.1, 0.15]),
        _campaign("cipro_positive_s17", "positive", 17, [0.6, 0.8]),
        _campaign("cipro_null_s17", "null", 17, [0.05, 0.2]),
        _campaign("cipro_positive_unpaired", "positive", 29, [0.99, 1.2]),
    ]
    relationship = {
        "relationship_kind": "control_pair",
        "role_dimension": "label_oracle_kind",
        "left_role": "positive",
        "right_role": "null",
        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
        "replicate_on": ["seed"],
        "pair_count": 2,
        "pairs": [
            {
                "left": "cipro_positive_s7",
                "right": "cipro_null_s7",
                "match": {
                    "target": "cipro",
                    "label_family_id": "densegen_plan_logic4",
                    "label_split_id": "random_id",
                    "seed": "7",
                },
            },
            {
                "left": "cipro_positive_s17",
                "right": "cipro_null_s17",
                "match": {
                    "target": "cipro",
                    "label_family_id": "densegen_plan_logic4",
                    "label_split_id": "random_id",
                    "seed": "17",
                },
            },
        ],
    }

    rows = build_notebook_campaign_set_metric_comparison_rows(
        campaigns,
        plot_name="score_selected_over_rounds",
        group_key="label_oracle_kind",
        relationship=relationship,
    )

    assert {row["campaign"] for row in rows} == {
        "cipro_positive_s7",
        "cipro_null_s7",
        "cipro_positive_s17",
        "cipro_null_s17",
    }
    assert {row["replicate_key"] for row in rows} == {"seed=7", "seed=17"}
    assert {row["metadata__seed"] for row in rows} == {"7", "17"}

    payload = render_notebook_campaign_set_metric_comparison_image(
        rows,
        title="Selected score over rounds",
        group_key="label_oracle_kind",
    )

    assert payload is not None
    assert payload["interval"]["kind"] == "iqr"
    assert payload["interval"]["unit"] == "relationship pairs"
    assert payload["interval"]["rounds_with_interval"] == 4
    assert payload["interval"]["min_unit_count"] == 2
    assert payload["interval"]["is_confidence_interval"] is False
    assert payload["axis_scale"]["limits"] == [-0.25, 0.0]
    assert "axis scale class" in payload["caption"]
    assert "not statistical confidence intervals" in payload["caption"]


def test_campaign_set_template_keeps_view_and_set_selectors_at_top() -> None:
    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="all",
        collection_manifest_path=Path("campaign_collection.yaml"),
        collection_visual_index_path=Path("collection_visuals/collection_visual_manifest.json"),
    )

    assert 'label="Review scope"' in text
    assert "view_mode_ui = mo.ui.dropdown(" in text
    assert '"Cross-campaign comparison": "Campaign set"' in text
    assert 'value="Campaign"' in text
    assert 'active_view_mode = str(view_mode_ui.value) if view_mode_ui is not None else "Campaign"' in text
    assert 'label="Comparison group"' in text
    assert 'selected_collection_set_title_md = mo.md(f"# {_title}")' in text
    assert 'if active_view_mode == "Campaign set"' in text
    assert "else selected_campaign_title_md" in text
    assert 'label="Campaign"' in text
    assert 'label="Review section"' in text
    assert 'label="Deliverable"' in text
    assert "build_notebook_collection_set_choices" in text
    assert "review_control_surface = render_notebook_review_control_surface(" in text
    assert "collection_set_ui=collection_set_ui" in text
    assert "visual_group_ui=visual_group_ui" in text
    assert "_primary_control_items = [campaign_ui]" not in text
    assert "_review_control_rows.append(" not in text
    assert "mo.vstack(_primary_control_items" not in text
    visual_panel_cell = text[
        text.index("def _(\n    CAMPAIGN_SET_BASERENDER_SURFACE_KIND,") : text.index(
            "def _(build_notebook_evidence_rows"
        )
    ]
    assert "render_notebook_visual_panel(" in visual_panel_cell
    assert "view_mode_ui" not in visual_panel_cell
    assert "collection_set_ui" not in visual_panel_cell


def test_collection_visual_choices_can_filter_by_campaign_set() -> None:
    visuals = [
        {
            "visual_id": "review",
            "label": "Realized review",
            "surface_kind": "study_realized_label_review",
            "comparison_set_key": "stage_b_realized_label_review",
            "comparison_set_label": "Stage B realized-label review",
        },
        {
            "visual_id": "score_cipro",
            "label": "Selected score",
            "surface_kind": "campaign_set_metric_comparison",
            "comparison_set_key": "target=cipro",
            "comparison_set_label": "Cipro",
        },
        {
            "visual_id": "score_ethanol",
            "label": "Selected score",
            "surface_kind": "campaign_set_metric_comparison",
            "comparison_set_key": "target=ethanol",
            "comparison_set_label": "Ethanol",
        },
    ]

    assert build_notebook_collection_set_choices(visuals) == [
        {
            "key": "stage_b_realized_label_review",
            "label": "Stage B realized-label review",
            "visual_count": 1,
            "match": {},
        },
        {"key": "target=cipro", "label": "Cipro", "visual_count": 1, "match": {}},
        {"key": "target=ethanol", "label": "Ethanol", "visual_count": 1, "match": {}},
    ]
    choices = build_notebook_collection_visual_choices(visuals, comparison_set_key="target=ethanol")
    assert [choice["comparison_set_label"] for choice in choices] == ["Ethanol"]
    assert choices[0]["label"] == "Selected score"


def test_collection_set_choices_disambiguate_duplicate_display_labels() -> None:
    visuals = [
        {
            "visual_id": "score_a",
            "label": "Selected score",
            "surface_kind": "campaign_set_metric_comparison",
            "comparison_set_key": "target=cipro",
            "comparison_set_label": "Stress condition",
        },
        {
            "visual_id": "score_b",
            "label": "Selected score",
            "surface_kind": "campaign_set_metric_comparison",
            "comparison_set_key": "target=ethanol",
            "comparison_set_label": "Stress condition",
        },
    ]

    choices = build_notebook_collection_set_choices(visuals)

    assert [choice["key"] for choice in choices] == ["target=cipro", "target=ethanol"]
    assert [choice["label"] for choice in choices] == ["Stress condition", "Stress condition (2)"]


def test_collection_set_choices_surface_evidence_tiers() -> None:
    visuals = [
        {
            "visual_id": "boundary",
            "label": "Selected-label enrichment",
            "surface_kind": "study_realized_label_review",
            "comparison_set_key": "slot_position_count_fixed",
            "comparison_set_label": "Count-fixed slot sentinel",
            "evidence_tier_label": "Current boundary",
            "evidence_tier_rank": 20,
        },
        {
            "visual_id": "claim",
            "label": "Selected-label enrichment",
            "surface_kind": "study_realized_label_review",
            "comparison_set_key": "count_fraction",
            "comparison_set_label": "Count-fraction composition",
            "evidence_tier_label": "Current claim",
            "evidence_tier_rank": 10,
        },
    ]

    choices = build_notebook_collection_set_choices(visuals)

    assert [choice["key"] for choice in choices] == ["count_fraction", "slot_position_count_fixed"]
    assert [choice["label"] for choice in choices] == [
        "Count-fraction composition",
        "Count-fixed slot sentinel",
    ]
    assert [choice["evidence_tier_label"] for choice in choices] == ["Current claim", "Current boundary"]


def test_collection_visual_choices_require_surface_kind() -> None:
    visuals = [
        {
            "visual_id": "score_ethanol",
            "label": "Selected score",
            "comparison_set_key": "target=ethanol",
            "comparison_set_label": "Ethanol",
        },
    ]

    with pytest.raises(ValueError, match="surface_kind"):
        build_notebook_collection_visual_choices(visuals, comparison_set_key="target=ethanol")


def test_campaign_summary_label_is_compact_for_probe_campaigns() -> None:
    row = build_notebook_campaign_summary_row(
        {
            "campaign": {
                "slug": "opal_axis_probe_v0_cipro_null_leave_sigma35_variant",
                "name": "Stress ethanol/ciprofloxacin cipro factor RF + SFXI + top N",
                "metadata": {
                    "probe_target": "cipro",
                    "probe_oracle_kind": "null",
                    "probe_split_id": "leave_sigma35_variant",
                    "probe_label_family_id": "densegen_plan_logic4",
                    "probe_seed": 29,
                },
            },
            "status": {"progress_status": "done"},
        }
    )

    assert row["label"] == "Cipro | matched-null | sigma35 | logic4 | s29 | done"
    assert len(row["label"]) <= 64
    assert "probe_label_family_id" not in row["label"]
    assert row["label_context"] == (
        "label_family_id=densegen_plan_logic4; label_oracle_kind=null; label_split_id=leave_sigma35_variant"
    )
    assert "Stress ethanol/ciprofloxacin" not in row["label"]


def test_notebook_template_omits_altair_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "import altair as alt" not in text


def test_notebook_template_uses_schema_pruned_records_loading() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "store.load()" not in text
    assert "store.load_columns(records_loaded_columns)" not in text
    assert "records_df = pl.from_pandas" not in text


def test_notebook_component_primitives_build_shared_evidence_models() -> None:
    view_model = {
        "campaign": {
            "slug": "campaign_a",
            "config_path": "campaign.yaml",
            "workdir": "workdir",
            "x_column": "x_vec",
            "label_source": "usr_sidecar",
            "model": "random_forest",
            "selection_views": [
                {
                    "id": "ciprofloxacin",
                    "objective": {"name": "sfxi_v1", "params": {"setpoint_vector": [0, 0, 1, 1]}},
                    "selection": {"name": "top_n", "params": {"top_k": 6}},
                }
            ],
            "metadata": {
                "response_axis": "ciprofloxacin",
                "comparison_group": "Ciprofloxacin factor",
            },
        },
        "status": {
            "progress_status": "attention",
            "round_selector": "latest",
            "round_count": 3,
            "latest_run_id": "run-3",
        },
        "progress": {
            "schema_version": "opal.campaign_progress.v1",
            "state": {"exists": True, "path": "workdir/state.json"},
            "round_selector": "latest",
            "event_contract": {
                "schema_version": "opal.progress_event_rollup.v1",
                "command_events": 1,
                "preflight_events": 2,
                "run_events": 6,
                "finalize_events": 1,
                "abort_events": 0,
                "attempt_ids": ["attempt-3"],
                "aborted_rounds": [],
                "ambiguous_rounds": [],
            },
            "rounds": [
                {
                    "round_index": 3,
                    "status": "done",
                    "last_stage": "round_done",
                    "elapsed_sec": 12.5,
                    "events": 10,
                    "predict": {"batch": 4, "of": 4, "rows": 157},
                    "summary": {
                        "aborted": False,
                        "run_scope": {
                            "resolved_run_id": "run-3",
                            "run_ids": ["run-3"],
                            "attempt_ids": ["attempt-3"],
                            "ambiguous_run_scope": False,
                        },
                    },
                    "path": "workdir/outputs/rounds/round_3/logs/round.log.jsonl",
                }
            ],
        },
        "review_manifests": {
            "primary": {
                "schema_version": "opal.campaign_review.v1",
                "review_scope": {"selection_view_id": "primary"},
                "selection": {"selected_count": 6},
            },
        },
        "plot_manifests": [
            {
                "name": "score",
                "kind": "metric_over_rounds",
                "status": "written",
                "generated_at": "2026-05-21T00:00:00Z",
                "run_id": "run-3",
                "rounds": [3],
                "tidy_csv": "plots/score.csv",
                "params": {"metric": "pred__score_selected"},
                "freshness": {"status": "fresh"},
                "metadata": {
                    "data_shape": "scalar over rounds",
                    "tidy_schema": ["round", "cohort", "metric", "summary", "value"],
                    "failure_modes": ["missing metric column", "metric is not numeric"],
                },
                "inputs": [{"path": "outputs/ledger/predictions", "role": "run_pred"}],
                "outputs": [{"role": "media", "path": "plots/score.png", "exists": True}],
            }
        ],
        "warnings": [
            {
                "category": "ReviewManifestWarning",
                "severity": "warning",
                "message": "Review manifest not found",
            }
        ],
        "stale_artifacts": [
            {
                "category": "StaleArtifactWarning",
                "severity": "warning",
                "path": "plots/old.png",
                "message": "old plot",
            }
        ],
        "artifact_garden": {
            "schema_version": "opal.artifact_garden.v1",
            "root": "workdir",
            "local_only": True,
            "artifact_roots": [
                {"name": "outputs", "path": "workdir/outputs", "exists": True, "file_count": 7, "size_bytes": 2048}
            ],
            "active_manifests": [{"kind": "plot_index", "status": "loaded"}],
            "stale_artifacts": [
                {"scope": "configured_plots", "path": "plots/old.png", "size_bytes": 12, "reason": "manifest absent"}
            ],
            "bytes": {"artifact_roots": 2048, "stale_artifacts": 12},
            "prune_plan": {"item_count": 1, "bytes_to_delete": 12, "requires_apply": True},
        },
    }

    selection_view = view_model["campaign"]["selection_views"][0]
    glance_rows = build_notebook_at_a_glance_rows(view_model, selection_view=selection_view)
    assert {"field": "campaign", "value": "campaign_a"} in glance_rows
    assert {
        "field": "description",
        "value": (
            "Campaign ID `campaign_a`. Campaign A fits `random_forest` once and evaluates 1 selection view "
            "from the shared predictions. The active X contract is `x_vec`."
        ),
    } in glance_rows
    assert {"field": "description source", "value": "derived"} in glance_rows
    assert {"field": "X column", "value": "x_vec"} in glance_rows
    assert {"field": "objective target", "value": "sfxi_v1 setpoint_vector=[0, 0, 1, 1]"} in glance_rows
    assert {"field": "stale artifacts", "value": 1} in glance_rows
    header_lines = build_notebook_campaign_header_lines(view_model, selection_view=selection_view)
    assert header_lines[0] == "# Campaign A"
    assert "Campaign ID `campaign_a`." in header_lines[2]
    assert "fits `random_forest` once and evaluates 1 selection view" in header_lines[2]
    assert "The active X contract is `x_vec`." in header_lines[2]
    assert header_lines[-1] == "**Objective target:** sfxi_v1 setpoint_vector=[0, 0, 1, 1]."
    assert (
        build_notebook_campaign_header_lines(
            view_model,
            selection_view=selection_view,
            heading_level=2,
        )[0]
        == "## Campaign A"
    )
    named_header = build_notebook_campaign_header_lines(
        {
            "campaign": {
                "name": "Stress ethanol/ciprofloxacin cipro factor RF + SFXI + top_n [cipro_positive_random_id]",
                "slug": "opal_axis_probe_v0_cipro_positive_random_id",
                "model": "random_forest",
                "selection_views": [selection_view],
            }
        },
        selection_view=selection_view,
    )
    assert named_header[0] == "# Stress ethanol/ciprofloxacin cipro factor RF + SFXI + top N"
    assert "Opal Axis Probe" not in named_header[2]

    visual_surface = build_notebook_visual_surface_model(view_model)
    assert visual_surface["missing_outputs"] == []
    assert visual_surface["stale_artifacts"] == view_model["stale_artifacts"]
    assert visual_surface["inventory_status_counts"] == {
        "generated_current": 1,
        "stale_unmanifested": 1,
    }
    assert visual_surface["choices"][0]["label"] == "Score"
    assert visual_surface["choices"][0]["path_label"] == "plots/score.png"
    assert visual_surface["choices"][0]["capability"]["objective_family"] == "generic"
    assert "Scope: round 3" in visual_surface["choices"][0]["alt_text"]
    assert "freshness fresh" in visual_surface["choices"][0]["alt_text"]
    labeled_surface = build_notebook_visual_surface_model(
        {
            **view_model,
            "plot_manifests": [
                {
                    **view_model["plot_manifests"][0],
                    "params": {
                        "title": "Short plot title",
                        "surface_label": "Specific objective expression",
                    },
                }
            ],
        }
    )
    assert labeled_surface["choices"][0]["label"] == "Specific objective expression"
    assert labeled_surface["choices"][0]["title"] == "Short plot title"

    scope_view_model = {
        **view_model,
        "plot_manifests": [
            {
                **view_model["plot_manifests"][0],
                "run_id": None,
                "rounds": "all",
                "tidy_csv": "plots/score_rall.csv",
                "outputs": [{"role": "media", "path": "plots/score_rall.png", "exists": True}],
            },
            {
                **view_model["plot_manifests"][0],
                "rounds": [3],
                "tidy_csv": "plots/score_r3.csv",
                "outputs": [{"role": "media", "path": "plots/score_r3.png", "exists": True}],
            },
        ],
    }
    scope_surface = build_notebook_visual_surface_model(scope_view_model)
    scope_choice = scope_surface["choices"][0]
    assert scope_choice["scope_count"] == 2
    scope_options = build_notebook_plot_scope_options(scope_choice)
    assert [option["label"] for option in scope_options] == ["all rounds", "round 3; run run-3"]
    assert select_notebook_plot_scope(scope_choice, "round 3; run run-3")["path_label"] == "plots/score_r3.png"

    inventory_rows = build_notebook_plot_inventory_rows(visual_surface)
    assert {
        "plot": "score",
        "kind": "metric_over_rounds",
        "status": "generated_current",
        "rounds": "round 3",
        "objective": "generic",
        "data": "predictions",
        "round behavior": "round_history",
        "labels": "none",
        "model artifact": False,
        "tidy": True,
        "path": "plots/score.png",
    } in inventory_rows
    assert any(row["plot"] == "old" and row["status"] == "stale_unmanifested" for row in inventory_rows)

    configured_surface = build_notebook_visual_surface_model(
        view_model,
        plot_entries=[
            {"name": "score", "kind": "metric_over_rounds"},
            {"name": "missing_plot", "kind": "scatter_score_vs_rank", "round_selector": "latest"},
        ],
    )
    assert configured_surface["missing_outputs"] == ["missing_plot"]
    configured_inventory = build_notebook_plot_inventory_rows(configured_surface)
    assert any(
        row["plot"] == "missing_plot"
        and row["status"] == "configured_missing_output"
        and row["round behavior"] == "single_or_round_history"
        for row in configured_inventory
    )
    with pytest.raises(KeyError, match="not_registered"):
        build_notebook_visual_surface_model(
            view_model,
            plot_entries=[{"name": "bad_plot", "kind": "not_registered"}],
        )
    no_plot_rows = {
        row["field"]: row["value"]
        for row in build_notebook_no_plot_scope_rows(
            {
                **view_model,
                "plot_manifests": [],
                "configured_plots": [
                    {"name": "score", "kind": "metric_over_rounds"},
                    {"name": "missing_plot", "kind": "scatter_score_vs_rank"},
                ],
            }
        )
    }
    assert no_plot_rows["campaign metadata"] == ("response_axis=ciprofloxacin; comparison_group=Ciprofloxacin factor")
    assert no_plot_rows["objective setpoint"] == "sfxi_v1 setpoint_vector=[0, 0, 1, 1]"
    assert "configured=2" in no_plot_rows["plot state"]
    assert "media_choices=0" in no_plot_rows["plot state"]
    assert "missing_outputs=2" in no_plot_rows["plot state"]
    assert "do not draw visual or biological conclusions" in no_plot_rows["evidence boundary"]
    assert "uv run opal plot -c campaign.yaml --round all" in no_plot_rows["next commands"]
    assert "opal review" not in no_plot_rows["next commands"]

    card_rows = build_notebook_plot_card_rows(visual_surface["choices"][0])
    assert {"field": "media", "value": "plots/score.png"} in card_rows
    assert {"field": "freshness", "value": "fresh"} in card_rows
    assert any(row["field"] == "capability" and "objective_family=generic" in row["value"] for row in card_rows)
    assert {"field": "tidy data", "value": "plots/score.csv"} in card_rows
    assert any(row["field"] == "source data" for row in card_rows)
    per_round_card_rows = build_notebook_plot_card_rows(select_notebook_plot_scope(scope_choice, "round 3; run run-3"))
    assert {"field": "rounds", "value": "round 3"} in per_round_card_rows
    assert {"field": "warnings", "value": "0"} in per_round_card_rows
    method_rows = build_notebook_plot_method_rows(visual_surface["choices"][0])
    assert any(row["section"] == "math" and "mean = sum(x) / n" in row["detail"] for row in method_rows)
    method_sections = build_notebook_plot_method_sections(visual_surface["choices"][0])
    assert "Read" in method_sections
    assert "mean = sum(x) / n" in method_sections["Math"]
    assert "Freshness: `fresh`" in method_sections["Data contract"]

    evidence = build_notebook_evidence_rows(view_model)
    assert [row["source"] for row in evidence] == ["path", "path", "warning", "stale_artifact"]
    assert evidence[0]["path"] == "campaign.yaml"

    validity_lines = "\n".join(build_notebook_validity_lines(view_model))
    assert "Review manifests: `1`" in validity_lines
    assert "Plot manifests: `1`" in validity_lines
    assert "Missing plot outputs: `0`" in validity_lines
    assert "Artifact garden: `opal.artifact_garden.v1`" in validity_lines

    change_lines = "\n".join(build_notebook_change_lines(view_model))
    assert "Latest run ID: `run-3`" in change_lines
    assert "Event phases: `command=1, preflight=2, run=6, finalize=1`" in change_lines
    change_rows = build_notebook_change_rows(view_model)
    assert change_rows == [
        {
            "round": 3,
            "status": "done",
            "last_stage": "round_done",
            "run_id": "run-3",
            "attempts": 1,
            "events": 10,
            "elapsed_sec": 12.5,
            "predict": "4/4 batches, 157 rows",
            "aborted": False,
            "ambiguous_run_scope": False,
            "log": "outputs/rounds/round_3/logs/round.log.jsonl",
        }
    ]

    metric_rows = build_notebook_metric_definition_rows(view_model)
    assert metric_rows == [
        {
            "plot": "score",
            "kind": "metric_over_rounds",
            "data_shape": "scalar over rounds",
            "tidy_schema": "round, cohort, metric, summary, value",
            "failure_modes": "missing metric column; metric is not numeric",
            "freshness": "fresh",
            "purpose": "not recorded",
        }
    ]

    artifact_lines = "\n".join(build_notebook_artifact_garden_lines(view_model))
    assert "local-only" in artifact_lines
    assert "Stale artifacts: `1`" in artifact_lines
    artifact_rows = build_notebook_artifact_garden_rows(view_model)
    assert [row["source"] for row in artifact_rows] == ["artifact_root", "stale_artifact", "prune_plan"]


def test_notebook_campaign_description_prefers_structured_target_metadata() -> None:
    view_model = {
        "campaign": {
            "slug": "tfbs_baeR_count_fraction_positive_random_id_seed7",
            "name": "Dense Array TFBS metadata probe: BaeR count fraction, metadata, seed 7",
            "description": "Stage B sentinel OPAL campaign for a synthetic DenseGen TFBS construction label.",
            "metadata": {
                "target_dropdown_label": "BaeR count fraction (count / 3)",
                "label_oracle_kind": "positive",
                "replicate_seed": 7,
                "rounds": 24,
                "selection_k": 6,
            },
        },
        "status": {},
        "runs": [],
        "rounds": [],
        "selection_summary": {"row_count": 0},
    }

    selection_view = {
        "id": "primary",
        "objective": {"name": "metadata_probe_v1", "params": {}},
    }
    rows = build_notebook_at_a_glance_rows(view_model, selection_view=selection_view)
    description = next(row["value"] for row in rows if row["field"] == "description")
    assert description == (
        "Pre-assay metadata probe for BaeR count fraction (count / 3) using the "
        "sequence-matched metadata table, seed 7. It tests whether the X representation supports active enrichment "
        "for this metadata, not measured phenotype prediction. The selection budget is 24 rounds x 6 records."
    )
    assert "sentinel" not in description
    assert "OPAL" not in description
    assert "DenseGen" not in description


def test_registered_plot_kinds_have_explicit_math_disclosure() -> None:
    fallback = "See the plot kind metadata"

    builtin_kinds = [
        kind for kind in list_plot_kinds() if get_plot(kind).__module__.startswith("dnadesign.opal.src.plots.")
    ]
    assert builtin_kinds

    for kind in builtin_kinds:
        meta = describe_plot_kind(kind)
        choice = {
            "kind": kind,
            "name": kind,
            "rounds": [2],
            "freshness": "fresh",
            "manifest": {
                "kind": kind,
                "rounds": [2],
                "run_id": "run-2",
                "generated_at": "2026-05-21T00:00:00Z",
                "manifest_path": "outputs/plots/example.manifest.json",
                "metadata": meta,
                "inputs": [{"role": "input", "path": "outputs/ledger/predictions.parquet"}],
                "params": {"metric": "pred__score_selected", "sample_n": 50, "min_n": 5, "top_k": 10},
            },
        }
        method_rows = build_notebook_plot_method_rows(choice)
        math_rows = [row for row in method_rows if row["section"] == "math"]
        method_sections = build_notebook_plot_method_sections(choice)

        assert math_rows, kind
        assert math_rows[0]["detail"], kind
        assert fallback not in math_rows[0]["detail"], kind
        assert fallback not in method_sections["Math"], kind
        assert "Input data layer:" in method_sections["Data contract"], kind
        assert "Provenance:" in method_sections["Data contract"], kind
        assert "Counts and replicates:" in method_sections["Data contract"], kind
        assert "manifest=outputs/plots/example.manifest.json" in method_sections["Data contract"], kind


def test_plot_method_sections_surface_manifest_decision_metadata() -> None:
    choice = {
        "kind": "response_magnitude_feasibility_frontier",
        "name": "rmf_frontier",
        "title": "Predicted RMF components locate selections relative to campaign thresholds",
        "rounds": [0],
        "freshness": "fresh",
        "warning_count": 0,
        "manifest": {
            "metadata": {
                "summary": "Predicted candidate constraints with observed labels identified.",
                "premise": "Selections occupy explicit response and fluorescence constraint space.",
                "decision_value": "Locates selected candidates relative to each configured boundary.",
                "rationale": "Separate components expose which requirement limits selection.",
                "non_claim_boundary": "Predictions do not establish measured response.",
                "capability": {
                    "objective_family": "response_magnitude_feasibility",
                    "data_layer": "predictions_plus_labels",
                    "round_scope": "single_round",
                    "label_requirement": "required",
                },
            },
            "params": {},
            "rounds": [0],
        },
    }

    sections = build_notebook_plot_method_sections(choice)

    assert sections["Decision"] == (
        "**Premise.** Selections occupy explicit response and fluorescence constraint space.\n\n"
        "**Decision use.** Locates selected candidates relative to each configured boundary.\n\n"
        "**Rationale.** Separate components expose which requirement limits selection.\n\n"
        "**Claim boundary.** Predictions do not establish measured response."
    )


def test_registered_plot_alt_text_exposes_primary_visual_encoding() -> None:
    builtin_kinds = [
        kind for kind in list_plot_kinds() if get_plot(kind).__module__.startswith("dnadesign.opal.src.plots.")
    ]
    assert builtin_kinds

    for kind in builtin_kinds:
        meta = describe_plot_kind(kind)
        alt_text = plot_alt_text(
            title=kind,
            kind=kind,
            summary=meta["summary"],
            params={
                "metric": "pred__score_selected",
                "metric_label": "Score = -MSE(y_hat, [0, 0, 1, 1])",
                "metric_expression": "score = -mean((y_hat - [0, 0, 1, 1])^2)",
                "score_field": "pred__score_selected",
                "y_axis": "score",
                "hue": "logic_fidelity",
                "size_by": "obj__effect_scaled",
                "vector_field": "pred__y_hat_model",
            },
            metadata=meta,
            rounds=[3],
            run_id=None,
            freshness="fresh",
            warning_count=0,
        )

        assert "Encoded fields:" in alt_text, kind
        assert "Score = -MSE(y_hat, [0, 0, 1, 1])" in alt_text, kind
        assert "score = -mean((y_hat - [0, 0, 1, 1])^2)" in alt_text, kind
        assert any(token in alt_text for token in ("x=", "left panel x=", "panels=")), kind
        assert "Scope: round 3" in alt_text, kind


def test_multistate_behavior_plot_text_uses_raw_units_and_one_softmin_scale() -> None:
    description = plot_math_description("multistate_response_behavior_frontier")

    assert "raw coordinates are r_on - r_off, b_on, and -b_off" in description
    assert "shared soft-min scale" in description
    assert "response_scale" not in description
    assert "signal_scale" not in description


def test_notebook_template_is_valid_python() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    ast.parse(text)


def test_campaign_set_notebook_has_campaign_and_plot_dropdowns() -> None:
    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    assert "# OPAL Review Notebook" not in text
    assert "selected_campaign_title_md = mo.md(_header_lines[0])" in text
    assert "selected_campaign_context_panel = mo.accordion(" in text
    assert "from dnadesign.opal.notebooks.api.generated import (" in text
    assert "from dnadesign.opal.notebooks.api import (" not in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "build_notebook_campaign_header_lines" in text
    assert "dnadesign.opal.src" not in text
    assert "__generated_with" in text
    assert 'generated_with = "' in text
    assert "Generated with marimo: `{__generated_with}`" not in text
    assert 'label="Round"' not in text
    assert "selected_round_selector = 'latest'" in text
    assert 'label="Campaign"' in text
    assert "campaign_labels = [f\"{index + 1}. {row['label']}\"" in text
    assert "selected_index = campaign_labels.index(selected_label)" in text
    assert 'label="Review section"' in text
    assert 'label="Deliverable"' in text
    assert "visual_group_label_memory, set_visual_group_label_memory = mo.state(None)" in text
    assert "visual_label_memory, set_visual_label_memory = mo.state({})" in text
    assert "plot_scope_label_memory, set_plot_scope_label_memory = mo.state({})" in text
    assert "on_change=set_visual_group_label_memory" in text
    assert "_preferred = _memory.get(_memory_key)" in text
    assert "on_change=_remember_visual" in text
    assert "on_change=_remember_scope" in text
    assert "Plot:" not in text
    assert "Campaigns at a glance" in text
    assert "Campaign status" in text
    assert "build_notebook_artifact_garden_rows" in text
    assert "build_notebook_change_rows" in text
    assert "build_notebook_metric_definition_rows" in text
    assert "build_notebook_visual_surface_model" in text
    assert "build_notebook_plot_card_rows" in text
    assert "build_notebook_plot_method_sections" in text
    assert "build_notebook_no_plot_scope_rows" not in text
    assert "build_notebook_validity_rows" in text
    assert "build_notebook_reader_evidence_visual_choices" in text
    assert "render_notebook_reader_evidence_plot_type_control" not in text
    assert "render_notebook_reader_evidence_record_control" in text
    assert "render_notebook_reader_evidence_time_control" not in text
    assert "render_notebook_review_control_surface" in text
    assert "render_notebook_visual_panel" in text
    assert "Current campaign and plot evidence" not in text
    assert 'Campaign-set comparison"))' not in text
    assert "build_notebook_validity_lines" not in text
    assert "build_notebook_artifact_garden_lines" not in text
    assert "build_notebook_change_lines" not in text
    assert "campaign_set_view_model" in text
    assert "LatentDNA" not in text
    assert "UMAP" not in text
    ast.parse(text)


def test_campaign_review_notebook_schema_matches_single_or_set_surface() -> None:
    single = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    campaign_set = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    schema_line = '__opal_notebook_template_schema__ = "opal.generated_campaign_review_notebook.v6"'
    assert schema_line in single
    assert schema_line in campaign_set
    assert "opal.generated_campaign_notebook" not in single
    assert "opal.generated_campaign_notebook" not in campaign_set


def test_campaign_set_notebook_render_fails_without_distinct_campaign_configs() -> None:
    with pytest.raises(OpalError, match="at least one campaign config"):
        render_campaign_set_notebook([], round_selector="latest")

    with pytest.raises(OpalError, match="distinct campaign configs"):
        render_campaign_set_notebook(
            [Path("campaign.yaml"), Path("campaign.yaml")],
            round_selector="latest",
        )


def test_campaign_set_template_embeds_absolute_collection_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml")],
        round_selector="latest",
        collection_manifest_path=Path("campaign_collection.yaml"),
        collection_visual_index_path=Path("collection_visuals/collection_visual_manifest.json"),
    )

    assigned_paths = {
        target.id: ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"collection_manifest_path", "collection_visual_index_path"}
    }
    assert assigned_paths["collection_manifest_path"] == str(Path("campaign_collection.yaml").resolve())
    assert assigned_paths["collection_visual_index_path"] == str(
        Path("collection_visuals/collection_visual_manifest.json").resolve()
    )


def test_reader_evidence_artifact_rows_preserve_zero_hour_snapshot() -> None:
    rows = build_notebook_reader_evidence_artifact_rows(
        {
            "reader_evidence_artifacts": [
                {
                    "label": "baseline",
                    "time_selected_h": 0.0,
                    "semantic_kind": "reader.sfxi_triptych",
                    "path": "baseline.png",
                    "exists": True,
                }
            ]
        }
    )

    assert rows[0]["time_selected_h"] == 0.0


def test_notebook_templates_stay_bounded_wiring_surfaces() -> None:
    single = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    campaign_set = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    assert '_scope_control_label = str(plot_scope_options[0].get("control_label") or "Plot scope")' in single
    assert "label=_scope_control_label" in single
    assert '_scope_control_label = str(plot_scope_options[0].get("control_label") or "Plot scope")' in campaign_set
    assert "label=_scope_control_label" in campaign_set
    assert "mo.vstack(_items, gap=0.35)" in campaign_set
    assert "return mo.vstack(_items)" not in campaign_set
    assert "mo.vstack(_items)\n    return" not in campaign_set
    assert len(single.splitlines()) <= 1050
    assert len(campaign_set.splitlines()) <= 1_000


def test_campaign_set_notebook_has_contract_backed_selected_sequence_render_surface() -> None:
    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    assert "selected_campaign_baserender_contract" in text
    assert "build_notebook_collection_baserender_role_choices" in text
    assert '"surface_kind": "baserender"' in text
    assert '"surface_kind": CAMPAIGN_SET_BASERENDER_SURFACE_KIND' in text
    assert "build_notebook_baserender_evidence_bundle" in text
    assert "build_notebook_baserender_record_controls" in text
    assert "baserender_record_memory, set_baserender_record_memory = mo.state({})" in text
    evidence_cell = text[
        text.index("baserender_record_evidence_bundle =") : text.index("return baserender_record_evidence_bundle,")
    ]
    assert "selected_campaign_analysis" not in evidence_cell
    assert "baserender_record_memory" not in evidence_cell
    assert "resolve_notebook_baserender_candidate_record" in text
    assert "load_notebook_baserender_campaign_context" in text
    assert "baserender_record_id" in text
    assert 'label="Selection round"' not in text
    assert 'label="Selection run"' not in text
    assert "build_notebook_collection_baserender_role_control" in text
    assert 'str(selected_round_selector).strip().lower() == "all"' not in text
    assert "baserender_campaign_model" in text
    assert "selected_baserender_round" in text
    assert "baserender_selected_round" not in text
    assert "baserender_candidate_records" in text
    assert "selected_baserender_selection_view_id" in text
    assert "baserender_selection_view_ui" in text
    assert "_view_id = str(selected_baserender_selection_view_id)" in text
    assert "baserender_candidate_evidence" in text
    assert "baserender_has_renderable_records" in text
    assert "baserender_diagnostic_panel" in text
    assert "selected_campaign_analysis.read_run_labels_used(" in text
    assert "selected_campaign_analysis.read_labels()" not in text
    assert "render_notebook_baserender_record" in text
    assert "render_notebook_visual_panel(" in text
    assert render_notebook_visual_panel.__module__.endswith(".notebook_components.visual_panel")
    baserender_helper_text = Path(
        "src/dnadesign/opal/src/analysis/notebook_components/visual_panel_baserender.py"
    ).read_text()
    collection_helper_text = Path(
        "src/dnadesign/opal/src/analysis/notebook_components/visual_panel_collection.py"
    ).read_text()
    assert '"width": "100%"' in baserender_helper_text
    assert '"background-color": "#FFFFFF"' in baserender_helper_text
    assert "Candidate and campaign evidence" in baserender_helper_text
    assert "Collection plot evidence" in collection_helper_text
    assert "densegen__used_tfbs_detail" not in text
    ast.parse(text)


def test_campaign_set_notebook_pins_baserender_scope_to_verified_selection_batch() -> None:
    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    assert "resolve_notebook_baserender_selection_batch_scope(" in text
    assert 'baserender_campaign_model.get("selection_batch")' in text
    assert "baserender_round_ui = None" in text
    assert "baserender_run_ui = None" in text
    assert "run_id=selected_baserender_run_id" in text
    assert "run_id=str(baserender_run_ui.value)" not in text
    assert "build_notebook_baserender_selection_view_control" in text
    assert "resolve_notebook_baserender_selection_view_id" in text


def test_collection_baserender_role_choices_follow_selected_campaign_set() -> None:
    campaigns = [
        {"campaign": {"slug": "tfbs_lexA_positive", "config_path": "positive.yaml"}},
        {"campaign": {"slug": "tfbs_lexA_matched_null", "config_path": "null.yaml"}},
    ]
    collection = {
        "comparison_lenses": [
            {
                "kind": "control_pair",
                "left_role": "positive",
                "left_role_label": "Dense Array metadata",
                "right_role": "null",
                "right_role_label": "row-shuffled metadata control",
                "pairs": [
                    {
                        "left": "tfbs_lexA_positive",
                        "right": "tfbs_lexA_matched_null",
                        "match": {"target": "lexA_count_fraction", "seed": "7"},
                    }
                ],
            }
        ]
    }
    choices = build_notebook_collection_baserender_role_choices(
        campaigns,
        collection,
        {"match": {"label_name": "lexA_count_fraction", "review_surface": "realized_label_review"}},
    )

    assert choices == [
        {"label": "Sequence-matched metadata", "role": "positive", "campaign_slug": "tfbs_lexA_positive"},
        {"label": "Row-shuffled control", "role": "null", "campaign_slug": "tfbs_lexA_matched_null"},
    ]

    count_fixed_collection = {
        "comparison_lenses": [
            {
                "kind": "control_pair",
                "left_role": "positive",
                "left_role_label": "Dense Array metadata",
                "right_role": "null",
                "right_role_label": "count-fixed slot-shuffle control",
                "pairs": [
                    {
                        "left": "tfbs_lexA_positive",
                        "right": "tfbs_lexA_matched_null",
                        "match": {"target": "lexA_in_slot0", "seed": "7"},
                    }
                ],
            }
        ]
    }
    count_fixed_choices = build_notebook_collection_baserender_role_choices(
        campaigns,
        count_fixed_collection,
        {
            "match": {
                "label_name": "lexA_in_slot0",
                "review_surface": "realized_label_review",
                "control_role": "count_fixed_shuffled_slot_negative_control",
            }
        },
    )
    assert count_fixed_choices[0]["label"] == "Sequence-matched metadata"
    assert count_fixed_choices[1]["label"] == "Slot-shuffled control"


def test_collection_baserender_role_choices_fall_back_to_selected_set_metadata() -> None:
    campaigns = [
        {
            "campaign": {
                "slug": "lexA_positive_seed7",
                "metadata": {
                    "label_name": "lexA_count_fraction",
                    "label_oracle_kind": "positive",
                    "seed": 7,
                },
            }
        },
        {
            "campaign": {
                "slug": "lexA_control_seed7",
                "metadata": {
                    "label_name": "lexA_count_fraction",
                    "label_oracle_kind": "null",
                    "seed": 7,
                },
            }
        },
        {
            "campaign": {
                "slug": "slot_positive_seed7",
                "metadata": {
                    "label_name": "lexA_in_slot0",
                    "label_oracle_kind": "positive",
                    "seed": 7,
                    "candidate_scope_policy_id": "tfbs_slot_position_target_count_eq_1_v1",
                },
            }
        },
        {
            "campaign": {
                "slug": "slot_control_seed7",
                "metadata": {
                    "label_name": "lexA_in_slot0",
                    "label_oracle_kind": "null",
                    "seed": 7,
                    "candidate_scope_policy_id": "tfbs_slot_position_target_count_eq_1_v1",
                },
            }
        },
        {
            "campaign": {
                "slug": "old_slot_positive_seed7",
                "metadata": {
                    "label_name": "lexA_in_slot0",
                    "label_oracle_kind": "positive",
                    "seed": 7,
                },
            }
        },
    ]

    composition_choices = build_notebook_collection_baserender_role_choices(
        campaigns,
        {"comparison_lenses": []},
        {
            "match": {
                "label_name": "lexA_count_fraction",
                "control_role": "matched_label_permutation_negative_control",
            }
        },
    )
    placement_choices = build_notebook_collection_baserender_role_choices(
        campaigns,
        {"comparison_lenses": []},
        {
            "match": {
                "label_name": "lexA_in_slot0",
                "control_role": "count_fixed_shuffled_slot_negative_control",
            }
        },
    )

    assert composition_choices == [
        {"label": "Sequence-matched metadata", "role": "positive", "campaign_slug": "lexA_positive_seed7"},
        {"label": "Row-shuffled control", "role": "null", "campaign_slug": "lexA_control_seed7"},
    ]
    assert placement_choices == [
        {"label": "Sequence-matched metadata", "role": "positive", "campaign_slug": "slot_positive_seed7"},
        {"label": "Slot-shuffled control", "role": "null", "campaign_slug": "slot_control_seed7"},
    ]


def test_collection_baserender_role_choices_use_generic_fallback_labels() -> None:
    campaigns = [
        {"campaign": {"slug": "positive", "metadata": {"label_oracle_kind": "positive", "target": "label"}}},
        {"campaign": {"slug": "control", "metadata": {"label_oracle_kind": "null", "target": "label"}}},
    ]
    collection = {
        "comparison_lenses": [
            {
                "kind": "control_pair",
                "left_role": "positive",
                "right_role": "null",
                "pairs": [{"left": "positive", "right": "control", "match": {"target": "label"}}],
            }
        ]
    }

    choices = build_notebook_collection_baserender_role_choices(
        campaigns,
        collection,
        {"match": {"target": "label"}},
    )

    assert choices == [
        {"label": "Positive label source", "role": "positive", "campaign_slug": "positive"},
        {"label": "Control label source", "role": "null", "campaign_slug": "control"},
    ]


def test_selected_baserender_records_preserve_view_and_competition_rank() -> None:
    class _CampaignAnalysis:
        def read_selection_view_predictions(self, **kwargs: object) -> pl.DataFrame:
            assert kwargs["selection_view_id"] == "ethanol"
            assert kwargs["round_selector"] == [3]
            assert kwargs["run_id"] == "run-3"
            return pl.DataFrame(
                {
                    "id": ["second", "unselected", "first", "null-selection"],
                    "as_of_round": [3, 3, 3, 3],
                    "run_id": ["run-3", "run-3", "run-3", "run-3"],
                    "view__rank_competition": [2, 1, 1, 3],
                    "view__is_selected": [True, False, True, None],
                }
            )

    records, rows = build_notebook_selected_baserender_records(
        _CampaignAnalysis(),
        selection_view_id="ethanol",
        round_value=3,
        run_id="run-3",
    )

    assert records == [
        {
            "record_id": "first",
            "selection_view_id": "ethanol",
            "view_rank": 1,
        },
        {
            "record_id": "second",
            "selection_view_id": "ethanol",
            "view_rank": 2,
        },
    ]
    assert rows == [
        {"field": "selection view", "value": "Ethanol"},
        {"field": "selection round", "value": 3},
        {"field": "selection run", "value": "run-3"},
        {"field": "selected records", "value": 2},
    ]
    assert build_notebook_selected_baserender_records(
        _CampaignAnalysis(), selection_view_id="ethanol", round_value=3, run_id=None
    )[1] == [{"field": "selection scope", "value": "no run available"}]


def test_selected_baserender_record_sets_cover_each_view_exactly_once() -> None:
    selection_batch = {
        "schema_version": "opal.selection_batch.v3",
        "campaign": {"slug": "campaign"},
        "as_of_round": 0,
        "run_id": "run-0",
        "verification": {"status": "pass"},
        "rows": [
            {
                "id": "ethanol-first",
                "campaign_slug": "campaign",
                "as_of_round": 0,
                "run_id": "run-0",
                "selection_view_ids": ["ethanol"],
                "selection_memberships": [{"selection_view_id": "ethanol", "rank": 1}],
            },
            {
                "id": "ciprofloxacin-first",
                "campaign_slug": "campaign",
                "as_of_round": 0,
                "run_id": "run-0",
                "selection_view_ids": ["ciprofloxacin"],
                "selection_memberships": [{"selection_view_id": "ciprofloxacin", "rank": 1}],
            },
        ],
    }
    records, status = build_notebook_selected_baserender_record_sets(
        selection_batch,
        campaign_slug="campaign",
        selection_view_ids=["ethanol", "ciprofloxacin"],
        round_value=0,
        run_id="run-0",
    )

    assert list(records) == ["ethanol", "ciprofloxacin"]
    assert records["ethanol"][0]["record_id"] == "ethanol-first"
    assert records["ciprofloxacin"][0]["record_id"] == "ciprofloxacin-first"
    assert status["ethanol"][0] == {"field": "selection view", "value": "Ethanol"}
    with pytest.raises(ValueError, match="unique"):
        build_notebook_selected_baserender_record_sets(
            selection_batch,
            campaign_slug="campaign",
            selection_view_ids=["ethanol", "ethanol"],
            round_value=0,
            run_id="run-0",
        )
    invalid_verification = {**selection_batch, "verification": {"status": "fail"}}
    with pytest.raises(ValueError, match="verification must pass"):
        build_notebook_selected_baserender_record_sets(
            invalid_verification,
            campaign_slug="campaign",
            selection_view_ids=["ethanol"],
            round_value=0,
            run_id="run-0",
        )
    invalid_views = {
        **selection_batch,
        "rows": [{**selection_batch["rows"][0], "selection_view_ids": "ethanol"}],
    }
    with pytest.raises(ValueError, match="requires selection_view_ids"):
        build_notebook_selected_baserender_record_sets(
            invalid_views,
            campaign_slug="campaign",
            selection_view_ids=["ethanol"],
            round_value=0,
            run_id="run-0",
        )


def test_baserender_candidate_catalog_unions_observed_and_all_selected_views() -> None:
    selection_batch = {
        "schema_version": "opal.selection_batch.v3",
        "campaign": {"slug": "campaign"},
        "as_of_round": 1,
        "run_id": "run-1",
        "verification": {"status": "pass"},
        "rows": [
            {
                "id": "selected-ethanol",
                "campaign_slug": "campaign",
                "as_of_round": 1,
                "run_id": "run-1",
                "selection_view_ids": ["ethanol"],
                "selection_memberships": [{"selection_view_id": "ethanol", "rank": 1}],
            },
            {
                "id": "selected-cipro",
                "campaign_slug": "campaign",
                "as_of_round": 1,
                "run_id": "run-1",
                "selection_view_ids": ["ciprofloxacin"],
                "selection_memberships": [{"selection_view_id": "ciprofloxacin", "rank": 2}],
            },
        ],
    }
    labels = pl.DataFrame(
        {
            "id": ["observed-only", "selected-ethanol"],
            "observed_round": [0, 0],
            "src": ["batch-0", "batch-0"],
        }
    )

    catalogs, status = build_notebook_baserender_candidate_catalog(
        selection_batch,
        labels,
        campaign_slug="campaign",
        selection_view_ids=["ethanol", "ciprofloxacin"],
        round_value=1,
        run_id="run-1",
    )

    assert [row["record_id"] for row in catalogs["ethanol"]] == [
        "selected-ethanol",
        "selected-cipro",
        "observed-only",
    ]
    assert [row["record_id"] for row in catalogs["ciprofloxacin"]] == [
        "selected-cipro",
        "selected-ethanol",
        "observed-only",
    ]
    observed = next(row for row in catalogs["ethanol"] if row["record_id"] == "observed-only")
    assert observed["evidence_roles"] == ["observed"]
    assert observed["observed_rounds"] == [0]
    shared = next(row for row in catalogs["ethanol"] if row["record_id"] == "selected-ethanol")
    assert shared["evidence_roles"] == ["observed", "selected"]
    assert shared["active_view_rank"] == 1
    assert {row["field"]: row["value"] for row in status["ethanol"]}["candidate records"] == 3


def test_baserender_selection_batch_scope_is_exact_and_fail_fast() -> None:
    assert resolve_notebook_baserender_selection_batch_scope(None) == (None, None)
    assert resolve_notebook_baserender_selection_batch_scope({}) == (None, None)
    assert resolve_notebook_baserender_selection_batch_scope({"as_of_round": 3, "run_id": "run-3"}) == (3, "run-3")
    with pytest.raises(ValueError, match="both round and run id"):
        resolve_notebook_baserender_selection_batch_scope({"as_of_round": 3})
    with pytest.raises(ValueError, match="invalid value"):
        resolve_notebook_baserender_selection_batch_scope({"as_of_round": 0.5, "run_id": "run"})


def test_baserender_record_memory_round_trip_is_scoped_to_selection_view() -> None:
    ethanol_key = build_notebook_baserender_record_memory_key(
        campaign_slug="campaign",
        run_id="run-0",
        round_value=0,
        selection_view_id="ethanol",
        review_group_key="handoff",
        deliverable_key="baserender",
    )
    cipro_key = build_notebook_baserender_record_memory_key(
        campaign_slug="campaign",
        run_id="run-0",
        round_value=0,
        selection_view_id="ciprofloxacin",
        review_group_key="handoff",
        deliverable_key="baserender",
    )
    other_deliverable_key = build_notebook_baserender_record_memory_key(
        campaign_slug="campaign",
        run_id="run-0",
        round_value=0,
        selection_view_id="ethanol",
        review_group_key="handoff",
        deliverable_key="other",
    )
    assert len({ethanol_key, cipro_key, other_deliverable_key}) == 3
    memory = {ethanol_key: "ethanol-rank-2"}

    assert (
        resolve_notebook_baserender_preferred_record_id(
            ["ethanol-rank-1", "ethanol-rank-2"],
            {"ethanol-rank-1": 5, "ethanol-rank-2": 5},
            preferred_record_id=memory.get(ethanol_key),
        )
        == "ethanol-rank-2"
    )
    assert (
        resolve_notebook_baserender_preferred_record_id(
            ["cipro-rank-1", "cipro-rank-2"],
            {"cipro-rank-1": 5, "cipro-rank-2": 5},
            preferred_record_id=memory.get(cipro_key),
        )
        == "cipro-rank-1"
    )
    assert (
        resolve_notebook_baserender_preferred_record_id(
            ["ethanol-rank-1", "ethanol-rank-2"],
            {"ethanol-rank-1": 5, "ethanol-rank-2": 5},
            preferred_record_id=memory.get(ethanol_key),
        )
        == "ethanol-rank-2"
    )


def test_baserender_record_controls_restore_each_selection_view_independently() -> None:
    class _Dropdown:
        def __init__(self, choices: dict[str, str], value: str, on_change: object, kwargs: dict[str, object]) -> None:
            self.value = choices[value]
            self.on_change = on_change
            self.kwargs = kwargs

    class _UI:
        def dropdown(
            self,
            choices: dict[str, str],
            *,
            value: str,
            on_change: object,
            **kwargs: object,
        ) -> _Dropdown:
            return _Dropdown(choices, value, on_change, kwargs)

    evidence = {
        view_id: {
            "selector_model": {
                "has_renderable_records": True,
                "record_options": [f"{prefix}-rank-1", f"{prefix}-rank-2"],
                "annotation_counts": {f"{prefix}-rank-1": 5, f"{prefix}-rank-2": 5},
                "record_choices": {
                    "Rank 1": f"{prefix}-rank-1",
                    "Rank 2": f"{prefix}-rank-2",
                },
            }
        }
        for view_id, prefix in (("ethanol", "ethanol"), ("ciprofloxacin", "cipro"))
    }
    memory: dict[str, str] = {}

    def _set_memory(value: dict[str, str]) -> None:
        memory.clear()
        memory.update(value)

    kwargs = {
        "campaign_slug": "campaign",
        "run_id": "run-0",
        "round_value": 0,
        "review_group_key": "handoff",
        "deliverable_key": "baserender",
        "memory": lambda: memory,
        "set_memory": _set_memory,
        "mo": SimpleNamespace(ui=_UI()),
    }
    controls = build_notebook_baserender_record_controls(evidence, **kwargs)
    controls["ethanol"].on_change("ethanol-rank-2")
    controls = build_notebook_baserender_record_controls(evidence, **kwargs)

    assert controls["ethanol"].value == "ethanol-rank-2"
    assert controls["ciprofloxacin"].value == "cipro-rank-1"
    controls["ciprofloxacin"].on_change("cipro-rank-2")
    controls = build_notebook_baserender_record_controls(evidence, **kwargs)
    assert controls["ethanol"].value == "ethanol-rank-2"
    assert controls["ciprofloxacin"].value == "cipro-rank-2"
    assert controls["ethanol"].kwargs["label"] == "Candidate lookup"
    assert controls["ethanol"].kwargs["searchable"] is True
    assert controls["ethanol"].kwargs["full_width"] is True


def test_baserender_record_memory_allows_pre_run_unavailable_state() -> None:
    records, status = build_notebook_selected_baserender_record_sets(
        None,
        campaign_slug="campaign",
        selection_view_ids=["ethanol"],
        run_id=None,
        round_value=None,
    )
    controls = build_notebook_baserender_record_controls(
        {"ethanol": {"selector_model": {"has_renderable_records": False}}},
        campaign_slug="campaign",
        run_id=None,
        round_value=None,
        review_group_key="handoff",
        deliverable_key="baserender",
        memory=lambda: {},
        set_memory=lambda _value: None,
        mo=object(),
    )

    assert records == {"ethanol": []}
    assert status == {"ethanol": [{"field": "selection scope", "value": "no rounds available"}]}
    assert controls == {"ethanol": None}


def test_selected_baserender_records_do_not_expose_partial_invalid_evidence() -> None:
    class _CampaignAnalysis:
        def read_selection_view_predictions(self, **_: object) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "id": ["valid-first", "invalid-second"],
                    "view__rank_competition": [1, 0],
                    "view__is_selected": [True, True],
                }
            )

    records, status = build_notebook_selected_baserender_records(
        _CampaignAnalysis(),
        selection_view_id="ethanol",
        round_value=0,
        run_id="run-0",
    )

    assert records == []
    assert status[0]["field"] == "selection ledger"
    assert "invalid competition rank" in str(status[0]["value"])


def test_baserender_record_choices_compact_record_ids_without_losing_identity() -> None:
    choices = build_notebook_baserender_record_choices(
        [
            "fixture-record-alpha-with-left-site",
            "fixture-record-beta-with-right-site",
        ]
    )

    assert choices == [
        {
            "label": "1. fixture-reco...eft-site",
            "record_id": "fixture-record-alpha-with-left-site",
        },
        {
            "label": "2. fixture-reco...ght-site",
            "record_id": "fixture-record-beta-with-right-site",
        },
    ]


def test_baserender_record_choices_label_counts_and_default_to_annotated_record() -> None:
    record_ids = [
        "fixture-record-no-annotations",
        "fixture-record-five-tfbs-sites",
    ]
    counts = {
        "fixture-record-no-annotations": 0,
        "fixture-record-five-tfbs-sites": 5,
    }

    choices = build_notebook_baserender_record_choices_with_counts(
        record_ids,
        counts,
        annotation_label="annotated elements",
        display_aliases={"fixture-record-five-tfbs-sites": "Candidate five"},
        candidate_evidence={
            "fixture-record-no-annotations": {
                "active_view_rank": 7,
                "selection_memberships": [{"selection_view_id": "ethanol", "view_rank": 7}],
                "observed_rounds": [],
            },
            "fixture-record-five-tfbs-sites": {
                "active_view_rank": None,
                "selection_memberships": [],
                "observed_rounds": [0],
            },
        },
    )

    assert choices == [
        {
            "label": "Selected rank 7 · fixture-reco...otations · 0 annotated elements",
            "record_id": "fixture-record-no-annotations",
        },
        {
            "label": "Observed R0 · Candidate five · fixture-reco...bs-sites · 5 annotated elements",
            "record_id": "fixture-record-five-tfbs-sites",
        },
    ]
    assert select_notebook_baserender_default_record_id(record_ids, counts) == "fixture-record-five-tfbs-sites"
    assert select_notebook_baserender_default_record_id(record_ids, {}) == record_ids[0]
    assert has_notebook_baserender_record_options(record_ids)
    assert not has_notebook_baserender_record_options(["(no renderable records)"])


def test_baserender_record_choices_disambiguate_compact_identifier_collisions() -> None:
    record_ids = [
        "abcdefghijkl-MIDDLE-ONE-qrstuvwx",
        "abcdefghijkl-MIDDLE-TWO-qrstuvwx",
    ]

    choices = build_notebook_baserender_record_choices_with_counts(record_ids, {})

    assert len(choices) == 2
    assert len({choice["label"] for choice in choices}) == 2
    assert {choice["record_id"] for choice in choices} == set(record_ids)
    assert all(f"ID {choice['record_id']}" in choice["label"] for choice in choices)


def test_baserender_densegen_contract_uses_metadata_records_path_for_annotations(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    metadata_records_path = tmp_path / "densegen.parquet"
    record_id = "fixture-record-densegen-metadata"
    stale_detail = []
    authoritative_detail = [
        {"part_kind": "tfbs", "regulator": "baeR_TTTCTSCVHNA", "offset_raw": 5, "length": 6},
        {"part_kind": "fixed_element", "role": "upstream", "offset_raw": 0, "length": 6},
    ]
    pl.DataFrame(
        {
            "id": [record_id],
            "sequence": ["TTGACAAAAAAAAAAAAAAAATATAAT"],
            "densegen__used_tfbs_detail": [stale_detail],
        }
    ).write_parquet(records_path)
    pl.DataFrame(
        {
            "id": [record_id],
            "densegen__used_tfbs_detail": [authoritative_detail],
        }
    ).write_parquet(metadata_records_path)

    contract = build_notebook_baserender_contract(
        ["id", "sequence", "densegen__used_tfbs_detail"],
        records_path=str(records_path),
        metadata_records_path=str(metadata_records_path),
        metadata_schema_columns=["id", "densegen__used_tfbs_detail"],
    )

    assert contract["available"] is True
    assert contract["metadata_records_path"] == str(metadata_records_path)
    assert build_notebook_baserender_record_annotation_counts(records_path, contract, record_ids=[record_id]) == {
        record_id: 2
    }
    row = load_notebook_baserender_record_row(records_path, record_id, contract)
    assert row is not None
    assert row["sequence"] == "TTGACAAAAAAAAAAAAAAAATATAAT"
    assert len(row["densegen__used_tfbs_detail"]) == 2
    assert row["densegen__used_tfbs_detail"][0]["regulator"] == "baeR_TTTCTSCVHNA"
    assert row["densegen__used_tfbs_detail"][1]["role"] == "upstream"


def test_campaign_dropdown_label_prefers_display_target_metadata() -> None:
    row = build_notebook_campaign_summary_row(
        {
            "campaign": {
                "slug": "tfbs_baeR_count_fraction_matched_null_random_id_seed7",
                "name": "DenseGen TFBS learnability: BaeR count fraction (BaeR count / 3), matched-null oracle, seed 7",
                "metadata": {
                    "target": "baeR_count_fraction",
                    "target_label": "BaeR count fraction (BaeR count / 3)",
                    "target_dropdown_label": "BaeR count fraction (count / 3)",
                    "label_oracle_kind": "null",
                    "label_split_id": "random_id",
                    "label_family_id": "tf_family_count_fraction",
                    "seed": 7,
                },
            },
            "status": {"progress_status": "done"},
            "plot_manifests": [],
            "stale_artifacts": [],
            "warnings": [],
        }
    )

    assert row["label"] == ("BaeR count fraction (count / 3) | matched-null | random | s7 | done")
    assert "baeR_count_fraction" not in row["label"]
    assert "tf_family_count_fraction" not in row["label"]


def test_campaign_dropdown_label_disambiguates_slot_probe_scope() -> None:
    base = {
        "campaign": {
            "slug": "tfbs_cpxR_or_baeR_in_slot2_matched_null_random_id_seed7",
            "name": "DenseGen TFBS learnability: CpxR or BaeR in slot 2, matched-null oracle, seed 7",
            "metadata": {
                "target_dropdown_label": "CpxR or BaeR in slot 2",
                "label_oracle_kind": "null",
                "label_split_id": "random_id",
                "label_family_id": "tf_slot_family_presence",
                "seed": 7,
            },
        },
        "status": {"progress_status": "done"},
        "plot_manifests": [],
        "stale_artifacts": [],
        "warnings": [],
    }
    count_preserving = {
        **base,
        "campaign": {
            **base["campaign"],
            "metadata": {
                **base["campaign"]["metadata"],
                "null_version": "densegen_tfbs_learnability_slot_geometry_count_matched_null_v1",
            },
        },
    }
    count_fixed = {
        **base,
        "campaign": {
            **base["campaign"],
            "metadata": {
                **base["campaign"]["metadata"],
                "candidate_scope_policy_id": "tfbs_slot_position_target_count_eq_1_v1",
            },
        },
    }

    assert "count-preserving" in build_notebook_campaign_summary_row(count_preserving)["label"]
    assert "count-fixed" in build_notebook_campaign_summary_row(count_fixed)["label"]
    assert (
        build_notebook_campaign_summary_row(count_preserving)["label"]
        != build_notebook_campaign_summary_row(count_fixed)["label"]
    )


def test_collection_visual_description_explains_metric_and_interval() -> None:
    text = build_notebook_collection_visual_description(
        {
            "title": "BaeR count fraction lift",
            "caption": "Realized selected-label lift by round.",
            "metric_label": "Selected-label lift ratio",
            "metric_expression": "mean(selected label) / mean(candidate-pool label)",
            "premise": "Active selection should enrich sequence-matched metadata.",
            "math_note": "Enrichment is mean(y_selected) / mean(y_candidate_pool).",
            "design_note": "Campaigns share initial IDs; only the label table differs.",
            "claim_boundary": "Synthetic metadata learnability only.",
            "summary": "per_round",
            "interval_kind": "none",
            "interpretation_note": "This is a synthetic construction-label learnability surface.",
        }
    )

    assert "BaeR count fraction lift" in text
    assert "Premise: Active selection should enrich sequence-matched metadata." in text
    assert "mean(selected label) / mean(candidate-pool label)" in text
    assert "Math: Enrichment is mean(y_selected) / mean(y_candidate_pool)." in text
    assert "Design: Campaigns share initial IDs; only the label table differs." in text
    assert "Claim boundary: Synthetic metadata learnability only." in text
    assert "Spread: none for this materialized single-pair review." in text
    assert "synthetic construction-label learnability" in text

    replicate_text = build_notebook_collection_visual_description(
        {
            "title": "BaeR count fraction lift",
            "caption": "Realized selected-label lift by round.",
            "metric_label": "Selected-label lift ratio",
            "metric_expression": "mean(selected label) / mean(candidate-pool label)",
            "summary": "per_round",
            "interval_kind": "iqr",
            "interval": {
                "kind": "iqr",
                "unit": "seed replicate",
                "is_confidence_interval": False,
            },
        }
    )

    assert "Spread: IQR across seed replicate; not a statistical confidence interval" in replicate_text


def test_notebook_baserender_contract_detects_schema_without_generated_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "from dnadesign.baserender import" not in text

    unavailable = build_notebook_baserender_contract(["id", "sequence"], records_path="records.parquet")
    assert unavailable["available"] is False
    unavailable_rows = {
        str(row["field"]): str(row["value"]) for row in build_notebook_baserender_contract_rows(unavailable)
    }
    assert unavailable_rows["available"] == "false"
    assert unavailable_rows["contract"] == "dnadesign.baserender.record_render_contract.v1"

    contract = build_notebook_baserender_contract(
        ["id", "sequence", "densegen__used_tfbs_detail"],
        records_path="records.parquet",
    )
    assert contract["available"] is True
    assert contract["adapter_kind"] == "densegen_tfbs"
    assert contract["adapter_columns"]["annotations"] == "densegen__used_tfbs_detail"
    assert callable(render_notebook_baserender_record)
    assert (
        "densegen__used_tfbs_detail"
        not in Path("src/dnadesign/opal/src/analysis/notebook_components/baserender.py").read_text()
    )

    generic = build_notebook_baserender_contract(
        ["id", "sequence", "opal__baserender_features", "densegen__used_tfbs_detail"],
        records_path="records.parquet",
    )
    assert generic["adapter_kind"] == "generic_features"


def test_notebook_baserender_options_fail_fast_for_bad_available_contract(tmp_path: Path) -> None:
    contract = build_notebook_baserender_contract(
        ["id", "sequence", "opal__baserender_features"],
        records_path=str(tmp_path / "missing.parquet"),
    )

    with pytest.raises(Exception, match="missing.parquet|No such file|not found"):
        build_notebook_baserender_record_options(tmp_path / "missing.parquet", contract)


def test_notebook_baserender_record_options_include_empty_densegen_annotation_rows(tmp_path: Path) -> None:
    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["null", "empty", "good"]),
            "sequence": pa.array(["TTGACATATAAT", "TTGACATATAAT", "TTGACATATAAT"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    None,
                    [],
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))

    assert build_notebook_baserender_record_options(records_path, contract) == ["null", "empty", "good"]
    assert build_notebook_baserender_record_annotation_counts(records_path, contract) == {
        "null": 0,
        "empty": 0,
        "good": 1,
    }
    assert load_notebook_baserender_record_row(records_path, "null", contract)["id"] == "null"
    assert load_notebook_baserender_record_row(records_path, "empty", contract)["id"] == "empty"
    assert load_notebook_baserender_record_row(records_path, "good", contract)["id"] == "good"


def test_notebook_baserender_record_options_reject_noncanonical_densegen_annotation_rows(tmp_path: Path) -> None:
    noncanonical_feature_type = pa.list_(
        pa.struct(
            [
                ("tf", pa.string()),
                ("tfbs", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["noncanonical"]),
            "sequence": pa.array(["TTGACATATAAT"]),
            "densegen__used_tfbs_detail": pa.array(
                [[{"tf": "lexA", "tfbs": "TTGACA", "orientation": "fwd", "offset": 0}]],
                type=noncanonical_feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))

    with pytest.raises(ValueError, match="BaseRender candidate evidence is incomplete"):
        build_notebook_baserender_record_options(
            records_path,
            contract,
            record_ids=["noncanonical"],
        )


def test_notebook_baserender_record_options_reject_partial_selected_evidence(tmp_path: Path) -> None:
    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["good", "bad"]),
            "sequence": pa.array(["TTGACATATAAT", "TTGACATATAAT"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                    [{"regulator": None, "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))

    with pytest.raises(ValueError, match="incomplete.*bad"):
        build_notebook_baserender_record_options(
            records_path,
            contract,
            record_ids=["good", "bad"],
        )

    model = build_notebook_baserender_selector_model(
        records_path,
        contract,
        [
            {
                "record_id": record_id,
                "active_selection_view_id": "ethanol",
                "active_view_rank": index,
                "selection_memberships": [{"selection_view_id": "ethanol", "view_rank": index}],
                "observed_rounds": [],
            }
            for index, record_id in enumerate(("good", "bad"), start=1)
        ],
    )

    assert model["record_options"] == ["good"]
    assert model["unrenderable_record_ids"] == ["bad"]
    assert model["has_renderable_records"] is True


def test_notebook_baserender_record_options_reject_duplicate_record_rows(tmp_path: Path) -> None:
    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["duplicate", "duplicate"]),
            "sequence": pa.array(["TTGACATATAAT", "TTGACATATAAT"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))

    with pytest.raises(ValueError, match="duplicate BaseRender record id"):
        build_notebook_baserender_record_options(records_path, contract, record_ids=["duplicate"])


def test_notebook_baserender_record_options_reject_mixed_validity_duplicate_rows(tmp_path: Path) -> None:
    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["duplicate", "duplicate"]),
            "sequence": pa.array([None, "TTGACATATAAT"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    None,
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))

    with pytest.raises(ValueError, match="duplicate BaseRender record id"):
        build_notebook_baserender_record_options(records_path, contract, record_ids=["duplicate"])


def test_notebook_baserender_record_options_reject_duplicate_metadata_rows(tmp_path: Path) -> None:
    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    metadata_path = tmp_path / "metadata.parquet"
    records = pa.table({"id": pa.array(["duplicate"]), "sequence": pa.array(["TTGACATATAAT"])})
    metadata = pa.table(
        {
            "id": pa.array(["duplicate", "duplicate"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    [{"regulator": "lexA", "sequence": "GGGGGG", "orientation": "fwd", "offset": 0}],
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(records, records_path)
    pq.write_table(metadata, metadata_path)
    contract = build_notebook_baserender_contract(
        records.column_names,
        records_path=str(records_path),
        metadata_records_path=str(metadata_path),
        metadata_schema_columns=metadata.column_names,
    )

    with pytest.raises(ValueError, match="duplicate BaseRender record id"):
        build_notebook_baserender_record_options(records_path, contract, record_ids=["duplicate"])


def test_notebook_baserender_record_options_filter_to_selected_ids(tmp_path: Path) -> None:
    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["first", "second"]),
            "sequence": pa.array(["TTGACATATAAT", "TTGACATATAAT"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))

    assert build_notebook_baserender_record_options(
        records_path,
        contract,
        record_ids=["second", "first"],
    ) == ["second", "first"]
    assert build_notebook_baserender_record_options(records_path, contract, record_ids=[]) == [
        "(no renderable records)"
    ]


def test_notebook_baserender_render_passes_selection_context_to_public_sequence_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dnadesign.opal.src.analysis.notebook_components.baserender_render as baserender_render

    captured: dict[str, object] = {}

    def fake_render_sequence_panel_image(row, **kwargs):
        captured["row"] = dict(row)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            image=np.full((72, 240, 4), 255, dtype=np.uint8),
            diagnostics=SimpleNamespace(sequence_length_bp=4, feature_count=0),
        )

    fake_baserender = SimpleNamespace(render_sequence_panel_image=fake_render_sequence_panel_image)
    monkeypatch.setattr(baserender_render, "import_module", lambda _name: fake_baserender)

    payload = render_notebook_baserender_record(
        {"id": "record-abc", "sequence": "ACGT"},
        {
            "available": True,
            "adapter_kind": "densegen_tfbs",
            "adapter_columns": {"id": "id", "sequence": "sequence", "annotations": "densegen__used_tfbs_detail"},
            "adapter_policies": {"require_non_empty": False},
            "render_route": "sequence_panel",
        },
        title="Ethanol selection · competition rank 2 · candidate record-abc",
    )

    row = captured["row"]
    kwargs = captured["kwargs"]
    assert isinstance(row, dict)
    assert "__opal_baserender_record_title" not in row
    assert isinstance(kwargs, dict)
    assert "overlay_text" not in kwargs["adapter_columns"]
    assert kwargs["title"] == "Ethanol selection · competition rank 2 · candidate record-abc"
    assert kwargs["vertical_anchor"] == "center"
    assert kwargs["canvas_top_pad_px"] == 0
    assert payload["record_id"] == "record-abc"
    assert payload["caption"] == "DenseGen TFBS annotation · 4 bp · 0 annotated elements"


def test_notebook_baserender_render_preserves_vertical_canvas_while_fitting_width(tmp_path: Path) -> None:
    from PIL import Image

    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["promoter-record"]),
            "sequence": pa.array(["TTGACAAAAAAAATATAATCCCCCCCCCCTTGACAGGGGGGTATAATCCGGAATTCCGG"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    [
                        {"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0},
                        {"regulator": "cpxR", "sequence": "TATAAT", "orientation": "fwd", "offset": 13},
                        {"regulator": "baeR", "sequence": "TTGACA", "orientation": "fwd", "offset": 29},
                        {"regulator": "baeR", "sequence": "TATAAT", "orientation": "fwd", "offset": 42},
                    ]
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))
    row = load_notebook_baserender_record_row(records_path, "promoter-record", contract)

    payload = render_notebook_baserender_record(row, contract)

    image = Image.open(BytesIO(payload["image_bytes"])).convert("RGBA")
    arr = np.asarray(image)
    assert arr[:, :, 3].min() == 255
    assert tuple(arr[0, 0, :3].tolist()) == (255, 255, 255)
    rgb = arr[:, :, :3]
    near_black_fraction = float((rgb.max(axis=2) <= 24).mean())
    assert near_black_fraction < 0.01
    assert image.width >= 900
    assert image.height == int(contract["target_height_px"])
    content_mask = (rgb < 245).any(axis=2)
    ys, xs = np.where(content_mask)
    assert int(xs.min()) <= 40
    assert int(image.width - 1 - xs.max()) <= 40
    assert int(ys.min()) > 0
    assert int(ys.max()) < image.height - 1


def test_notebook_baserender_content_fit_normalizes_black_matte_to_white() -> None:
    from PIL import Image, ImageDraw

    from dnadesign.opal.src.analysis.notebook_components.baserender_render import _encode_content_fit_white_png

    source = Image.new("RGBA", (420, 140), (0, 0, 0, 255))
    draw = ImageDraw.Draw(source)
    draw.rounded_rectangle((96, 48, 324, 88), radius=10, fill=(68, 106, 140, 255))

    image = Image.open(BytesIO(_encode_content_fit_white_png(source))).convert("RGBA")
    arr = np.asarray(image)

    assert arr[:, :, 3].min() == 255
    assert tuple(arr[0, 0, :3].tolist()) == (255, 255, 255)
    assert tuple(arr[-1, -1, :3].tolist()) == (255, 255, 255)
    edge = np.concatenate((arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]))
    assert int(((edge < 20).all(axis=1)).sum()) == 0
    assert bool(((arr[:, :, 2] > arr[:, :, 0]) & (arr[:, :, 1] > 80)).any())
