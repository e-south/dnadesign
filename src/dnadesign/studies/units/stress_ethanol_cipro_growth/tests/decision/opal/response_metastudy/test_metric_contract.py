"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_metric_contract.py

Intrinsic SFXI and candidate-support contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    SfxiEvidenceFrame,
    SfxiSourceProvenance,
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    metric_contract,
    support,
)

TARGET_VIEWS = (
    StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0)),
    StressTargetView("ciprofloxacin", "Ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
    StressTargetView("and", "AND", (0.0, 0.0, 0.0, 1.0)),
)


def test_intrinsic_metric_contract_distinguishes_setpoints_and_exposes_boundary() -> None:
    checks = metric_contract.build_metric_contract_tests(TARGET_VIEWS).set_index("check_id")

    for target_view in TARGET_VIEWS:
        assert checks.loc[f"setpoint_identity_{target_view.id}", "status"] == "pass"
    assert checks.loc["positive_exponent_scale_rank_invariance", "status"] == "pass"
    assert checks.loc["off_state_absolute_intensity_boundary", "status"] == "attention"
    assert checks.loc["rmf_noncompensation", "status"] == "pass"
    assert checks.loc["rmf_state_permutation_equivariance", "status"] == "pass"


def test_rmf_cardinality_screen_exposes_hard_extrema_noise_bias() -> None:
    screen = metric_contract.build_rmf_cardinality_pressure(
        state_counts=(2, 4, 8),
        draws=2_000,
        seed=7,
    )

    assert set(screen["mask_topology"]) == {"one ON", "balanced", "one OFF"}
    assert (screen["response_separation_bias"] <= 0.0).all()
    multi_on = screen[screen["on_count"].gt(1)]
    multi_off = screen[screen["off_count"].gt(1)]
    assert (multi_on["on_magnitude_floor_bias"] < 0.0).all()
    assert (multi_off["off_magnitude_ceiling_bias"] > 0.0).all()
    assert screen.loc[screen["on_count"].eq(1), "on_magnitude_floor_bias"].abs().max() < 0.02
    assert screen.loc[screen["off_count"].eq(1), "off_magnitude_ceiling_bias"].abs().max() < 0.02
    one_on = screen[screen["mask_topology"].eq("one ON")].set_index("state_count")
    assert one_on.loc[8, "off_magnitude_ceiling_bias"] > one_on.loc[2, "off_magnitude_ceiling_bias"]
    one_off = screen[screen["mask_topology"].eq("one OFF")].set_index("state_count")
    assert one_off.loc[8, "on_magnitude_floor_bias"] < one_off.loc[4, "on_magnitude_floor_bias"]


def test_setpoint_support_counts_each_target_view_independently() -> None:
    sfxi_evidence = tuple(
        SfxiEvidenceFrame(
            source=SfxiSourceProvenance(
                source_id=f"test-{target_view.id}",
                source_campaign_slug=f"test-sfxi-{target_view.id}",
                expected_run_id="r0",
                target_view_id=target_view.id,
            ),
            target_view=target_view,
            predictions=pd.DataFrame(),
            y_hat=np.empty((0, 8)),
            denom=1.0,
            run_id="r0",
        )
        for target_view in TARGET_VIEWS[:2]
    )
    scores = {
        "ethanol": pd.DataFrame({"logic_fidelity": [0.2, 0.5, 0.8]}),
        "ciprofloxacin": pd.DataFrame({"logic_fidelity": [0.1, 0.3, 0.4]}),
    }

    result = support.build_setpoint_support(sfxi_evidence, scores, thresholds=(0.25, 0.45))

    at_guardrail = result[result["logic_threshold"] == 0.45].set_index("selection_view_id")
    assert at_guardrail.loc["ethanol", "candidate_count"] == 2
    assert at_guardrail.loc["ciprofloxacin", "candidate_count"] == 0
