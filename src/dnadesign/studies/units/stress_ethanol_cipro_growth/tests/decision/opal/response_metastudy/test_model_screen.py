"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_model_screen.py

Tests for the grouped response-label model screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy import evaluation
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    StressTargetView,
)


def test_factorial_contrast_preserves_all_response_separations() -> None:
    response_magnitude = np.asarray(
        [
            [1.0, 3.0, 2.0, 5.0, -1.0, 0.5, -0.5, 1.0],
            [2.0, 1.0, 4.0, 3.0, -0.2, 0.1, 0.4, 0.8],
        ]
    )
    decoded = evaluation.decode_to_response_magnitude(
        evaluation.response_magnitude_to_factorial_contrast7(response_magnitude),
        decoder="factorial_contrast7",
    )

    for target in ((0, 1, 0, 1), (0, 0, 1, 1), (0, 0, 0, 1), (0, 1, 1, 1)):
        on = np.asarray(target, dtype=bool)
        expected = response_magnitude[:, :4][:, on].min(axis=1) - response_magnitude[:, :4][:, ~on].max(axis=1)
        actual = decoded[:, :4][:, on].min(axis=1) - decoded[:, :4][:, ~on].max(axis=1)
        assert np.allclose(actual, expected)
    assert np.allclose(decoded[:, 4:], response_magnitude[:, 4:])


def test_only_declared_reduction_and_its_contrast_are_promotion_eligible() -> None:
    response_magnitude = np.arange(24, dtype=float).reshape(3, 8)
    ids = ["a", "b", "c"]
    base = pd.DataFrame(
        response_magnitude,
        columns=["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"],
    )
    base.insert(0, "id", ids)
    primary = base.assign(reduction_id="primary")
    sensitivity = base.assign(reduction_id="sensitivity")

    representations = evaluation.build_label_representations(
        ids=ids,
        snapshot_y=response_magnitude,
        response_summaries=pd.concat((primary, sensitivity), ignore_index=True),
        primary_reduction_id="primary",
        promotion_reduction_ids=frozenset({"primary"}),
    )

    eligibility = {value.id: value.promotion_eligible for value in representations}
    assert eligibility == {
        "snapshot_vec8": False,
        "primary": True,
        "primary__factorial_contrast7": True,
        "sensitivity": False,
    }


def test_grouped_model_screen_returns_margin_and_enrichment_evidence() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 4))
    response = np.column_stack((x[:, 0], x[:, 0] + x[:, 1], x[:, 2], x[:, 1] + x[:, 2]))
    brightness = np.column_stack((x[:, 3] - 1.0, x[:, 3] + 0.5, x[:, 3] - 0.5, x[:, 3] + 1.0))
    response_magnitude = np.column_stack((response, brightness))
    ids = [f"id-{index}" for index in range(len(x))]
    summaries = pd.DataFrame(response_magnitude, columns=["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"])
    summaries.insert(0, "reduction_id", "event_logmean_6_12h_post")
    summaries.insert(0, "id", ids)
    representations = evaluation.build_label_representations(
        ids=ids,
        snapshot_y=response_magnitude,
        response_summaries=summaries,
        primary_reduction_id="event_logmean_6_12h_post",
        promotion_reduction_ids=frozenset({"event_logmean_6_12h_post"}),
    )
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))

    groups = np.repeat(["g1", "g2", "g3", "g4"], 6)
    summary, group_metrics, enrichment = evaluation.screen_label_models(
        x,
        groups=groups,
        candidate_ids=ids,
        representations=representations,
        target_views=(ethanol,),
        uncertainty_rows=_uncertainty_rows(ids, groups, selection_view_id="ethanol"),
        scale_quantile=0.9,
        bootstrap_samples=100,
        random_forest_params={"n_estimators": 10, "random_state": 7, "n_jobs": 1},
        model_specs=(evaluation.ModelScreenSpec(id="pls2", kind="pls", components=2),),
    )

    assert set(summary["representation_id"]) == {
        "snapshot_vec8",
        "event_logmean_6_12h_post",
        "event_logmean_6_12h_post__factorial_contrast7",
    }
    assert (summary["model_id"] == "pls2").all()
    assert summary["ethanol__response_separation_spearman"].notna().all()
    assert np.allclose(
        summary["weakest_required_ordering_spearman"],
        summary[["weakest_target_view_response_separation_spearman", "weakest_target_view_feasibility_spearman"]].min(
            axis=1
        ),
    )
    assert not group_metrics.empty
    assert not enrichment.empty
    assert enrichment["selection_defined"].all()
    assert enrichment["selected_true_percentile"].between(0.0, 1.0).all()


def test_retrospective_enrichment_rejects_tied_mean_predictions() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(18, 3))
    response_magnitude = rng.normal(size=(18, 8))
    ids = [f"id-{index}" for index in range(len(x))]
    summaries = pd.DataFrame(response_magnitude, columns=["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"])
    summaries.insert(0, "reduction_id", "event_logmean_6_12h_post")
    summaries.insert(0, "id", ids)
    representations = evaluation.build_label_representations(
        ids=ids,
        snapshot_y=response_magnitude,
        response_summaries=summaries,
        primary_reduction_id="event_logmean_6_12h_post",
        promotion_reduction_ids=frozenset({"event_logmean_6_12h_post"}),
    )
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))

    groups = np.repeat(["g1", "g2", "g3"], 6)
    _, _, enrichment = evaluation.screen_label_models(
        x,
        groups=groups,
        candidate_ids=ids,
        representations=(representations[1],),
        target_views=(ethanol,),
        uncertainty_rows=_uncertainty_rows(ids, groups, selection_view_id="ethanol"),
        scale_quantile=0.9,
        bootstrap_samples=100,
        random_forest_params={"n_estimators": 10, "random_state": 7, "n_jobs": 1},
        model_specs=(evaluation.ModelScreenSpec(id="mean", kind="mean"),),
    )

    assert not enrichment["selection_defined"].any()
    assert enrichment["selected_true_percentile"].isna().all()


def _uncertainty_rows(ids: list[str], groups: np.ndarray, *, selection_view_id: str) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "id": ids,
            "selection_view_id": [selection_view_id] * len(ids),
            "reader_experiment_id": groups,
        }
    )
    for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling"):
        rows[f"{component}__combined_sd"] = np.linspace(0.1, 0.3, len(ids))
    return rows
