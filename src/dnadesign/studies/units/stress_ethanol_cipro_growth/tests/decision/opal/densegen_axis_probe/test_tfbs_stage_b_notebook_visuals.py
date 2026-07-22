"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_notebook_visuals.py

Regression tests for TFBS stage b notebook visuals studies units stress.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .probe_modules import probe_module

_specs = probe_module("tfbs.stage_b.notebook_visuals.specs")
StageBNotebookVisualSpec = _specs.StageBNotebookVisualSpec
build_visual_spec_registry = _specs.build_visual_spec_registry
realized_visual_spec = _specs.realized_visual_spec
slot_visual_spec = _specs.slot_visual_spec


def test_stage_b_notebook_visual_specs_are_registry_backed() -> None:
    spec = realized_visual_spec("realized_label_lift_trajectory")

    assert spec.visual_id(label_name="lexA present") == "tfbs_stage_b_lexA_present_selected_label_lift_trajectory"
    assert spec.plot_filename(label_name="lexA present") == "lexA_present__selected_label_lift_trajectory.png"
    assert spec.plot_title(label_name="lexA present") == "LexA Present enrichment vs candidate pool"
    assert "oracle" not in spec.caption_text(label_name="lexA_present").lower()
    assert "Count-fraction labels use target TFBS count / 3" in spec.caption_text(label_name="lexA_present")
    assert "slot-position labels use y=1" in spec.caption_text(label_name="lexA_present")
    assert "bold line = mean" in spec.caption_text(label_name="lexA_present")
    assert "sample SD" in spec.caption_text(label_name="lexA_present")
    assert spec.tidy_csv_path(
        trajectory_csv_path=Path("trajectory.csv"),
        pair_summary_csv_path=Path("pair_summary.csv"),
    ) == Path("trajectory.csv")
    assert slot_visual_spec("slot_count_stratified_lift_summary").metric_name == (
        "positive_minus_null_count_stratified_lift_ratio"
    )
    assert slot_visual_spec("slot_count_stratified_lift_summary").plot_filename() == (
        "slot_count_stratified_lift_summary.png"
    )


def test_stage_b_notebook_visual_registry_fails_fast() -> None:
    duplicate = StageBNotebookVisualSpec(
        kind="duplicate_kind",
        visual_id_template="duplicate_kind",
        label="Duplicate",
        group_key="group",
        metric_name="metric",
        metric_label="Metric",
        metric_expression="metric",
        summary_name="summary",
        tidy_source="trajectory",
        caption="Duplicate fixture.",
        plot_filename_template="duplicate_kind.png",
        plot_title_template="Duplicate fixture",
        alt_text="Duplicate fixture plot.",
    )
    bad_template = StageBNotebookVisualSpec(
        kind="bad_template",
        visual_id_template="bad_template_{unknown}",
        label="Bad template",
        group_key="group",
        metric_name="metric",
        metric_label="Metric",
        metric_expression="metric",
        summary_name="summary",
        tidy_source="trajectory",
        caption="Bad fixture.",
        plot_filename_template="bad_template.png",
        plot_title_template="Bad template",
        alt_text="Bad template plot.",
    )

    with pytest.raises(ValueError, match="Unsupported Stage B realized review plot kind"):
        realized_visual_spec("legacy_plot_kind")
    with pytest.raises(ValueError, match="requires a nonempty label_name"):
        realized_visual_spec("positive_null_lift_summary").visual_id()
    with pytest.raises(ValueError, match="Duplicate test visual spec kind: duplicate_kind"):
        build_visual_spec_registry((duplicate, duplicate), surface="test")
    with pytest.raises(ValueError, match="unsupported template field"):
        build_visual_spec_registry((bad_template,), surface="test")
