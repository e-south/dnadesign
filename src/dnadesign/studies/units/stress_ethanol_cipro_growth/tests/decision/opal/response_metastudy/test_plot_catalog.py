"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_plot_catalog.py

Tests for the response metric metastudy plot catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import plot_catalog


def test_plot_catalog_has_unique_semantic_deliverables() -> None:
    plot_ids = [spec.plot_id for spec in plot_catalog.PLOT_SPECS]
    filenames = [spec.filename for spec in plot_catalog.PLOT_SPECS]

    assert len(plot_ids) == len(set(plot_ids))
    assert len(filenames) == len(set(filenames))
    assert {"primary_decision", "metric_diagnostic", "screen_appendix"} <= {
        spec.tier for spec in plot_catalog.PLOT_SPECS
    }
    assert {
        "measured_response_examples",
        "sfxi_score_contours",
        "target_view_pareto_fronts",
        "denominator_sensitivity",
        "policy_comparison_panel_roles",
        "model_validation",
    } <= set(plot_ids)
    assert {
        "metric_compensation_comparison",
        "sfxi_comparison_stability",
        "sfxi_comparison_target_coverage",
    }.isdisjoint(plot_ids)
    tiers = {spec.plot_id: spec.tier for spec in plot_catalog.PLOT_SPECS}
    assert tiers["selected_setpoint_residuals"] == "screen_appendix"
    assert tiers["policy_comparison_panel_roles"] == "screen_appendix"
    assert {spec.plot_id for spec in plot_catalog.PLOT_SPECS if spec.tier == "primary_decision"} == {
        "measured_response_examples",
        "response_separation_stability",
        "label_model_screen",
        "greedy_support_evidence",
    }
    primary = [spec for spec in plot_catalog.PLOT_SPECS if spec.tier == "primary_decision"]
    assert [(spec.review_step, spec.plot_id) for spec in primary] == [
        (1, "measured_response_examples"),
        (2, "response_separation_stability"),
        (3, "label_model_screen"),
        (4, "greedy_support_evidence"),
    ]
    assert all(spec.review_step is None for spec in plot_catalog.PLOT_SPECS if spec.tier != "primary_decision")
    for spec in plot_catalog.PLOT_SPECS:
        assert spec.premise.strip()
        assert spec.decision_value.strip()
        assert spec.rationale.strip()
        assert spec.alt_text.strip()
        assert spec.non_claim_boundary.strip()
        assert spec.data_table.startswith("tables/")
        assert spec.title
        assert not spec.title.endswith(".")
        assert spec.title[0].isupper()
        reader_text = " ".join((spec.title, spec.decision_value, spec.alt_text))
        assert "effect_scaled" not in reader_text
        assert "policy_id" not in reader_text
    measured = next(spec for spec in plot_catalog.PLOT_SPECS if spec.plot_id == "measured_response_examples")
    assert measured.title == "The target mask changes which fixed Reader states define each RMF requirement"
    assert "same measured SpyP and sulAp summaries" in measured.decision_value
    model_screen = next(spec for spec in plot_catalog.PLOT_SPECS if spec.plot_id == "label_model_screen")
    assert "sequence features" in model_screen.title
    assert "configured campaign model, baseline, and fixed challengers remain separate" in model_screen.alt_text
    greedy = next(spec for spec in plot_catalog.PLOT_SPECS if spec.plot_id == "greedy_support_evidence")
    assert "configured campaign random forest" in greedy.alt_text
    assert "not selection authority" in greedy.alt_text


def test_plot_manifest_preserves_catalog_order_and_fields() -> None:
    paths = {spec.plot_id: Path("/tmp") / spec.filename for spec in plot_catalog.PLOT_SPECS}

    manifest = plot_catalog.build_plot_manifest(paths, root=Path("/tmp"))

    assert manifest["plot_id"].tolist() == [spec.plot_id for spec in plot_catalog.PLOT_SPECS]
    assert set(manifest.columns) == {
        "plot_id",
        "filename",
        "tier",
        "visual_type",
        "review_step",
        "title",
        "premise",
        "decision_value",
        "rationale",
        "alt_text",
        "non_claim_boundary",
        "data_table",
        "path",
    }
    assert manifest["path"].str.startswith("/").sum() == 0
