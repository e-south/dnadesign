"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_review_notebook.py

Tests for the generated single-viewport metastudy notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy import build_review_summary
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting.notebook import (
    write_review_notebook,
)


def test_review_notebook_has_one_deliverable_selector_and_one_viewport(tmp_path: Path) -> None:
    path = write_review_notebook(tmp_path)
    source = path.read_text(encoding="utf-8")
    normalized_source = " ".join(source.split())

    assert 'app = marimo.App(width="medium")' in source
    assert source.count("mo.ui.dropdown(") == 2
    assert "mo.ui.radio(" not in source
    assert source.count("mo.image(") == 1
    assert "data_table" in source
    assert "non_claim_boundary" in source
    assert "response_metastudy.build_review_summary(bundle_manifest)" in source
    assert 'bundle_manifest["recommendation"]' not in source
    assert "**Prospective hill climb:** {review_summary.prospective_hill_climb}" in source
    assert "review_summary.primary_assay_summary" in source
    assert "publication.verify_bundle_artifacts(bundle_root)" in source
    assert 'label="Evidence tier"' in source
    assert 'label="Figure"' in source
    assert "options=tier_options" in source
    assert "value=next(iter(tier_options))" in source
    assert 'plot_catalog["tier"].eq(tier.value)' in source
    assert "row.title" in source
    assert "How the target mask changes the score" in source
    assert "log2[(YFP / CFP)_design,i(t)]" in source
    assert "log2[(YFP / OD600)_design,i(t)]" in source
    assert "Ethanol | ethanol; ethanol + ciprofloxacin" in source
    assert "S_RMF = min(q_response, q_on, q_off)" in source
    assert "**RMF selection rule**" in source
    assert "Positive values clear the configured requirement" in normalized_source
    assert "Higher is better" not in source
    assert "top-K selects" in normalized_source
    assert "a strong component cannot compensate for a failed one" in normalized_source
    assert "wrap=True" in source
    assert 'width="100%"' in source
    assert '"max-width": "100%"' in source
    assert '"max-height": "68vh"' in source
    assert '"object-fit": "contain"' in source


def test_review_notebook_initializes_dropdown_with_an_option_name(tmp_path: Path) -> None:
    path = write_review_notebook(tmp_path)
    source = path.read_text(encoding="utf-8")

    assert "value=next(iter(deliverable_options))" in source
    assert 'value=plot_catalog.iloc[0]["plot_id"]' not in source


def test_review_notebook_numbers_and_sorts_primary_evidence(tmp_path: Path) -> None:
    source = write_review_notebook(tmp_path).read_text(encoding="utf-8")

    assert 'plot_catalog["tier"].eq(tier.value)' in source
    assert 'tier_catalog.sort_values("review_step", kind="mergesort")' in source
    assert 'f"{int(row.review_step)}. {row.title}"' in source
    assert "Primary evidence figures require an explicit review step" in source


def test_review_summary_is_rmf_specific_and_fail_fast() -> None:
    manifest = {
        "response_metric_screen": {
            "status": "screen_complete_not_promoted",
            "primary_reduction_candidate": "event_logmean_6_12h_post",
            "label_count": 35,
            "reader_event_experiment_count": 8,
            "prospective_hill_climb_demonstrated": False,
            "best_fixed_model_screen": {"weakest_target_view_response_separation_spearman": 0.15},
            "response_screen_protocol": {
                "reductions": [
                    {
                        "id": "event_logmean_6_12h_post",
                        "screen_role": "primary",
                        "method": "geometric_time_mean",
                        "window_start_event_h": 6.0,
                        "window_end_event_h": 12.0,
                    }
                ]
            },
        }
    }

    summary = build_review_summary(manifest)

    assert summary.decision == "RMF remains inactive for production selection"
    assert "0.15" in summary.basis
    assert summary.evidence_base == "35 observed labels across 8 Reader experiments"


def test_review_summary_rejects_unpromoted_screen_with_hill_climb_claim() -> None:
    manifest = {
        "response_metric_screen": {
            "status": "screen_complete_not_promoted",
            "primary_reduction_candidate": "event_logmean_6_12h_post",
            "label_count": 35,
            "reader_event_experiment_count": 8,
            "prospective_hill_climb_demonstrated": True,
            "best_fixed_model_screen": {"weakest_target_view_response_separation_spearman": 0.15},
            "response_screen_protocol": {
                "reductions": [
                    {
                        "id": "event_logmean_6_12h_post",
                        "screen_role": "primary",
                        "method": "geometric_time_mean",
                        "window_start_event_h": 6.0,
                        "window_end_event_h": 12.0,
                    }
                ]
            },
        }
    }

    with pytest.raises(ValueError, match="cannot claim a prospective hill climb"):
        build_review_summary(manifest)
