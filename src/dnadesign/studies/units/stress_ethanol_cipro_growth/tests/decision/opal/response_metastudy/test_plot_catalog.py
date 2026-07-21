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
from typing import get_args

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import (
    plot_catalog,
    plot_contracts,
)


def test_plot_catalog_has_unique_semantic_deliverables() -> None:
    plot_ids = [spec.plot_id for spec in plot_catalog.PLOT_SPECS]
    filenames = [spec.filename for spec in plot_catalog.PLOT_SPECS]

    assert len(plot_ids) == len(set(plot_ids))
    assert len(filenames) == len(set(filenames))
    assert {"primary_decision", "metric_diagnostic", "screen_appendix"} <= {
        spec.tier for spec in plot_catalog.PLOT_SPECS
    }
    assert {
        "historical_observed_sfxi_decomposition",
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
    section_order: dict[str, list[int]] = {}
    for spec in plot_catalog.PLOT_SPECS:
        assert spec.review_section in {
            "assay_and_labels",
            "historical_model_screens",
            "rmf_comparator",
            "sfxi_comparator",
        }
        assert spec.section_order > 0
        section_order.setdefault(spec.review_section, []).append(spec.section_order)
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
    for orders in section_order.values():
        assert sorted(orders) == list(range(1, len(orders) + 1))
    sections = {spec.plot_id: spec.review_section for spec in plot_catalog.PLOT_SPECS}
    orders = {spec.plot_id: spec.section_order for spec in plot_catalog.PLOT_SPECS}
    assert sections["response_separation_stability"] == "assay_and_labels"
    assert orders["response_separation_stability"] == 1
    assert sections["reader_event_intervals"] == "assay_and_labels"
    assert sections["repeated_design_agreement"] == "assay_and_labels"
    assert sections["label_model_screen"] == "historical_model_screens"
    assert sections["measured_response_examples"] == "rmf_comparator"
    assert sections["historical_observed_sfxi_decomposition"] == "sfxi_comparator"
    assert orders["historical_observed_sfxi_decomposition"] == 1
    assert orders["policy_guardrail_matrix"] == 2
    assert sections["sfxi_score_contours"] == "sfxi_comparator"
    observed_sfxi = next(
        spec for spec in plot_catalog.PLOT_SPECS if spec.plot_id == "historical_observed_sfxi_decomposition"
    )
    assert observed_sfxi.data_table == "tables/sfxi_round0_training_components.csv"
    assert "measured corpus" in observed_sfxi.title.lower()
    measured = next(spec for spec in plot_catalog.PLOT_SPECS if spec.plot_id == "measured_response_examples")
    assert measured.title == "The target mask changes which fixed Reader states define each RMF requirement"
    assert "same measured SpyP and sulAp summaries" in measured.decision_value
    model_screen = next(spec for spec in plot_catalog.PLOT_SPECS if spec.plot_id == "label_model_screen")
    assert "sequence features" in model_screen.title
    assert "configured campaign model, baseline, and fixed challengers remain separate" in model_screen.alt_text
    greedy = next(spec for spec in plot_catalog.PLOT_SPECS if spec.plot_id == "greedy_support_evidence")
    assert "configured campaign random forest" in greedy.alt_text
    assert "not selection authority" in greedy.alt_text


def test_review_section_type_matches_the_published_section_ontology() -> None:
    assert set(get_args(plot_contracts.ReviewSection)) == {
        "assay_and_labels",
        "historical_model_screens",
        "rmf_comparator",
        "sfxi_comparator",
    }


def test_plot_manifest_preserves_catalog_order_and_fields() -> None:
    paths = {spec.plot_id: Path("/tmp") / spec.filename for spec in plot_catalog.PLOT_SPECS}

    manifest = plot_catalog.build_plot_manifest(paths, root=Path("/tmp"))

    assert manifest["plot_id"].tolist() == [spec.plot_id for spec in plot_catalog.PLOT_SPECS]
    assert set(manifest.columns) == {
        "plot_id",
        "filename",
        "tier",
        "review_section",
        "section_order",
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
