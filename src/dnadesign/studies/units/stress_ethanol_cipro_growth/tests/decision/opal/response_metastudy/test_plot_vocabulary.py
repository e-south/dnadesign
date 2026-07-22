"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_plot_vocabulary.py

Tests for response metastudy publication plot vocabulary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting import (
    plot_vocabulary,
)


def test_plot_vocabulary_names_every_visible_domain_value() -> None:
    assert plot_vocabulary.target_view_label("ethanol") == "Ethanol"
    assert plot_vocabulary.target_view_label("ciprofloxacin") == "Ciprofloxacin"
    assert plot_vocabulary.target_view_label("and") == "AND"
    assert plot_vocabulary.target_view_label("or") == "OR (pressure test)"
    assert plot_vocabulary.policy_label("sfxi_beta1_gamma1") == "Canonical SFXI beta=1 gamma=1"
    assert plot_vocabulary.compact_policy_label("sfxi_beta1_gamma1") == "Canonical SFXI"
    assert plot_vocabulary.compact_policy_label("tradeoff_logic0p95") == "Logic weight 0.95"
    assert plot_vocabulary.panel_role_label("canonical_sfxi_shared_overlap") == "Canonical SFXI shared overlap"
    assert plot_vocabulary.model_metric_label("v01") == "Logic\nCiprofloxacin (v01)"
    assert plot_vocabulary.model_metric_label("y11_star") == "Fluorescence\nBoth stresses (y*11)"
    assert plot_vocabulary.representation_label("event_logmean_6_12h_post") == "6-12 h\nlog mean"
    assert plot_vocabulary.representation_label("event_logmean_0_6h_post") == "0-6 h\nlog mean"
    assert plot_vocabulary.representation_label("event_logmean_0_12h_post") == "0-12 h\nlog mean"
    assert plot_vocabulary.reader_experiment_label("20260706_sfxi_sensor-panel-m9-glu-secg") == (
        "2026-07-06 | SECG sensor panel"
    )


@pytest.mark.parametrize(
    ("labeler", "value"),
    [
        (plot_vocabulary.target_view_label, "unknown"),
        (plot_vocabulary.policy_label, "unknown"),
        (plot_vocabulary.compact_policy_label, "unknown"),
        (plot_vocabulary.panel_role_label, "unknown"),
        (plot_vocabulary.model_metric_label, "unknown"),
        (plot_vocabulary.representation_label, "unknown"),
        (plot_vocabulary.representation_role, "unknown"),
        (plot_vocabulary.reader_experiment_label, "unknown"),
    ],
)
def test_plot_vocabulary_fails_fast_on_unmapped_values(labeler, value: str) -> None:
    with pytest.raises(ValueError, match="has no publication label"):
        labeler(value)
