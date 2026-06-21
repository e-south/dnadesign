"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/workbench/test_teto_pwm_trim_rescue_deliverables.py

Tests for tetO PWM trim rescue deliverable IA.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from ..support.paths import repo_root_from


def test_teto_pwm_trim_rescue_deliverable_plan_maps_review_and_handoff_outputs() -> None:
    root = repo_root_from(__file__)
    study_dir = root / "docs" / "studies" / "retron_hairpin_design"
    design_set = yaml.safe_load(
        (study_dir / "workbench" / "design_sets" / "teto_pwm_trim_rescue_v1.yaml").read_text(encoding="utf-8")
    )
    plan_path = study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))

    assert plan["contract"] == "retron_hairpin_deliverable_plan_v1"
    assert plan["design_set_ref"] == (
        "docs/studies/retron_hairpin_design/workbench/design_sets/teto_pwm_trim_rescue_v1.yaml"
    )
    assert design_set["deliverable_plan_ref"] == (
        "docs/studies/retron_hairpin_design/workbench/deliverables/teto_pwm_trim_rescue_v1.yaml"
    )
    assert plan["compiler_spec_ref"] == design_set["compiler_spec_ref"]
    assert plan["output_policy"]["generated_artifacts"] == (
        "ignored_workbench_outputs_preferred_or_explicit_transient_output_root"
    )
    assert plan["output_policy"]["preferred_generated_root"].endswith("workbench/outputs/teto_pwm_trim_rescue_v1")
    assert "workbench/deliverables" in plan["output_policy"]["durable_records"]
    assert "GenBank exports" in plan["output_policy"]["do_not_commit_by_default"]
    assert plan["reader_boundary"]["status"] == "experiment_time_only"

    families = plan["artifact_families"]
    assert set(families) == {
        "clone_handoff_bundle",
        "future_reader_outcome_overlay",
        "msd_sequence_review_stills",
        "msd_sequence_review_video",
        "pwm_trim_review_panel",
        "review_package_manifest",
    }
    assert families["msd_sequence_review_stills"]["expected_count"] == design_set["expected_variant_count"] == 9
    assert families["clone_handoff_bundle"]["expected_count"] == 9
    assert families["pwm_trim_review_panel"]["status"] == "current_review_renderer_output"
    assert families["msd_sequence_review_video"]["status"] == "current_review_renderer_output"
    assert families["review_package_manifest"]["status"] == "current_review_renderer_output"
    assert families["clone_handoff_bundle"]["status"] == "current_materialize_output"
    assert families["future_reader_outcome_overlay"]["owner_surface"] == (
        "Reader SPOP bridge and future trim-outcome join"
    )

    pwm_panel_ids = {panel["payload_trim_id"] for panel in families["pwm_trim_review_panel"]["panels"]}
    assert pwm_panel_ids == set(design_set["payload_trims"])
    assert (
        "reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.png"
        in families["pwm_trim_review_panel"]["expected_files"]
    )
    assert (
        "reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4"
        in families["msd_sequence_review_video"]["expected_files"]
    )
    assert (
        "variants/<construct-id>__<msd-design-id>/sequences/forward.gb"
        in families["clone_handoff_bundle"]["per_design_files"]
    )
    assert (
        "variants/<construct-id>__<msd-design-id>/sequences/forward.fa"
        in families["clone_handoff_bundle"]["per_design_files"]
    )
    assert "reviews/review_manifest.json" in families["review_package_manifest"]["expected_files"]
