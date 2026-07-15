"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_greedy_support.py

Tests for grouped evidence about greedy selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    greedy_support,
)


def test_greedy_support_reports_exact_uncertainty_without_assigning_slots() -> None:
    screen = pd.DataFrame(
        [
            {
                "representation_id": "primary",
                "model_id": "campaign_random_forest",
                "model_role": "campaign_model",
                "promotion_eligible": True,
                "all_target_view_metrics_finite": True,
                "weakest_required_ordering_spearman": 0.05,
                "median_channel_spearman": 0.2,
            },
            {
                "representation_id": "primary",
                "model_id": "pls4",
                "model_role": "fixed_challenger",
                "promotion_eligible": True,
                "all_target_view_metrics_finite": True,
                "weakest_required_ordering_spearman": 0.15,
                "median_channel_spearman": 0.4,
            },
        ]
    )
    rows: list[dict[str, object]] = []
    for model_id, per_view in (
        ("campaign_random_forest", (("ciprofloxacin", 2), ("and", 3), ("ethanol", 1))),
        ("pls4", (("ciprofloxacin", 7), ("and", 4), ("ethanol", 3))),
    ):
        for selection_view_id, successes in per_view:
            for index in range(8):
                rows.append(
                    {
                        "representation_id": "primary",
                        "model_id": model_id,
                        "selection_view_id": selection_view_id,
                        "reader_experiment_id": f"exp-{index}",
                        "selection_defined": True,
                        "selected_true_percentile": 0.8 if index < successes else 0.2,
                        "beats_group_median": index < successes,
                    }
                )

    evidence = greedy_support.build_greedy_support_evidence(
        screen,
        pd.DataFrame(rows),
        primary_reduction_id="primary",
        model_role="fixed_challenger",
    ).set_index("selection_view_id")

    assert evidence.at["ciprofloxacin", "groups_beating_median"] == 7
    assert evidence.at["ciprofloxacin", "fraction_ci_low"] < 0.5
    assert evidence.at["ciprofloxacin", "evidence_posture"] == "descriptive_above_half"
    assert set(evidence["allocation_boundary"]) == {"descriptive_only_no_slot_assignment"}
    assert set(evidence["evidence_basis"]) == {"best_fixed_challenger"}
    assert "greedy_slots" not in evidence.columns

    campaign = greedy_support.build_greedy_support_evidence(
        screen,
        pd.DataFrame(rows),
        primary_reduction_id="primary",
        model_role="campaign_model",
    ).set_index("selection_view_id")
    assert campaign.at["ciprofloxacin", "groups_beating_median"] == 2
    assert set(campaign["model_id"]) == {"campaign_random_forest"}
    assert set(campaign["evidence_basis"]) == {"configured_campaign_model"}
