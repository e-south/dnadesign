"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_metric_behavior.py

Tests for response metric metastudy metric-behavior probes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    PolicySpec,
    SfxiEvidenceFrame,
    SfxiSourceProvenance,
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    metric_behavior,
    pressure_rows,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation.scoring import (
    score_sfxi_evidence,
)


def test_denominator_sensitivity_recomputes_effect_scaled_topk() -> None:
    policy = PolicySpec(
        id="sfxi_beta1_gamma1",
        label="Canonical SFXI",
        kind="multiplicative",
        beta=1.0,
        gamma=1.0,
    )
    sfxi_evidence = (_sfxi_evidence_frame(),)
    scored = {policy.id: {"ethanol": score_sfxi_evidence(sfxi_evidence[0], policy)}}

    sensitivity = metric_behavior.build_denominator_sensitivity(
        (policy,),
        sfxi_evidence,
        scored,
        factors=(0.5, 1.0, 2.0),
        top_k=2,
    )

    assert sensitivity["denominator_factor"].tolist() == [0.5, 1.0, 2.0]
    assert sensitivity["effective_topk"].tolist() == [2, 2, 2]
    assert (
        sensitivity.loc[sensitivity["denominator_factor"] == 2.0, "median_effect_scaled"].iloc[0]
        < sensitivity.loc[
            sensitivity["denominator_factor"] == 1.0,
            "median_effect_scaled",
        ].iloc[0]
    )
    assert set(sensitivity["interpretation_boundary"]) == {"recomputed_from_predicted_effect_raw"}


def test_effect_dominance_is_indeterminate_when_correlations_are_not_finite() -> None:
    pairwise = pd.DataFrame(
        [
            {
                "policy_id": "sfxi_beta1_gamma1",
                "metric": "within_selection_view",
                "selection_view_a": "ethanol",
                "selection_view_b": component,
                "pearson": np.nan,
            }
            for component in ("logic_fidelity", "effect_scaled")
        ]
    )

    rows = pressure_rows.effect_dominance_rows(pairwise)

    assert rows[0]["status"] == "indeterminate"
    assert "not finite" in rows[0]["interpretation"]


def _sfxi_evidence_frame() -> SfxiEvidenceFrame:
    target_view = StressTargetView(
        id="ethanol",
        label="Synthetic",
        target_mask=(0.0, 1.0, 0.0, 1.0),
    )
    y_hat = pd.Series(
        [
            [0.05, 0.9, 0.05, 0.9, 0.0, 2.0, 0.0, 2.0],
            [0.2, 0.75, 0.2, 0.75, 0.0, 1.0, 0.0, 1.0],
            [0.8, 0.2, 0.8, 0.2, 0.0, 3.0, 0.0, 3.0],
        ]
    )
    predictions = pd.DataFrame(
        {
            "id": ["strong", "moderate", "wrong-bright"],
            "sequence": ["A", "C", "G"],
            "pred__y_hat_model": y_hat,
            "pred__score_selected": [0.0, 0.0, 0.0],
            "sel__rank_competition": [1, 2, 3],
            "sel__is_selected": [True, False, False],
            "obj__logic_fidelity": [0.0, 0.0, 0.0],
            "obj__effect_scaled": [0.0, 0.0, 0.0],
        }
    )
    return SfxiEvidenceFrame(
        source=SfxiSourceProvenance(
            source_id="test-sfxi-ethanol",
            source_campaign_slug="test-sfxi-ethanol",
            expected_run_id="r0",
            target_view_id="ethanol",
        ),
        target_view=target_view,
        predictions=predictions,
        y_hat=pd.DataFrame(y_hat.tolist()).to_numpy(dtype=float),
        denom=8.0,
        run_id="r0",
    )
