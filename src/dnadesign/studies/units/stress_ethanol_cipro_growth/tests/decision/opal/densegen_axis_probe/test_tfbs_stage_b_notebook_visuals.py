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

    assert spec.visual_id(label_name="lexA present") == "tfbs_stage_b_lexA_present_realized_label_lift_trajectory"
    assert spec.tidy_csv_path(
        trajectory_csv_path=Path("trajectory.csv"),
        pair_summary_csv_path=Path("pair_summary.csv"),
    ) == Path("trajectory.csv")
    assert slot_visual_spec("slot_count_stratified_lift_summary").metric_name == (
        "positive_minus_null_count_stratified_lift_ratio"
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
    )

    with pytest.raises(ValueError, match="Unsupported Stage B realized review plot kind"):
        realized_visual_spec("legacy_plot_kind")
    with pytest.raises(ValueError, match="requires a nonempty label_name"):
        realized_visual_spec("positive_null_lift_summary").visual_id()
    with pytest.raises(ValueError, match="Duplicate test visual spec kind: duplicate_kind"):
        build_visual_spec_registry((duplicate, duplicate), surface="test")
