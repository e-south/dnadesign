from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.api.sfxi import (
    SFXI_API_VERSION,
    SFXI_REFERENCE_OVERLAY_PREFIX,
    SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION,
    SFXIScoringConfig,
    score_vec8,
    to_sfxi_reference_overlay_records,
)


def test_score_vec8_public_api_uses_objective_channel_names() -> None:
    vec8 = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cfg = SFXIScoringConfig(
        setpoint_vector=(0.0, 0.0, 0.0, 1.0),
        scaling_percentile=95,
        scaling_min_n=1,
        scaling_eps=1.0e-8,
        logic_exponent_beta=1.0,
        intensity_exponent_gamma=1.0,
        intensity_log2_offset_delta=0.0,
    )

    result = score_vec8(vec8, cfg, scaling_vec8=vec8[0:1, :])

    assert result.api_version == SFXI_API_VERSION
    assert result.objective_name == "sfxi_v1"
    assert result.denom_used == 2.0
    assert np.allclose(result.logic_fidelity, np.array([1.0, 1.0]))
    assert np.allclose(result.effect_raw, np.array([2.0, 1.0]))
    assert np.allclose(result.effect_scaled, np.array([1.0, 0.5]))
    assert np.allclose(result.sfxi, np.array([1.0, 0.5]))

    row = result.to_records()[0]
    assert {"logic_fidelity", "effect_raw", "effect_scaled", "sfxi"} <= set(row)
    assert "f_logic" not in row
    assert "e_scaled" not in row
    assert "score" not in row


def test_sfxi_reference_overlay_records_are_namespaced() -> None:
    vec8 = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    result = score_vec8(
        vec8,
        SFXIScoringConfig(setpoint_vector=(0.0, 0.0, 0.0, 1.0), scaling_min_n=1),
        scaling_vec8=vec8[0:1, :],
    )

    rows = to_sfxi_reference_overlay_records(
        result,
        batch_id="batch-0",
        campaign_id="stress_ethanol",
        design_id=["design-a", "design-b"],
        source_id=["usr-a", "usr-b"],
    )

    assert len(rows) == 2
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}schema_version"] == SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}metric_id"] == "sfxi_v1/sfxi"
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}metric_value"] == rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}sfxi"]
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}logic_fidelity"] == 1.0
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}effect_scaled"] == 1.0
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}batch_id"] == "batch-0"
    assert rows[1][f"{SFXI_REFERENCE_OVERLAY_PREFIX}campaign_id"] == "stress_ethanol"
    assert rows[1][f"{SFXI_REFERENCE_OVERLAY_PREFIX}design_id"] == "design-b"
    assert rows[1][f"{SFXI_REFERENCE_OVERLAY_PREFIX}source_id"] == "usr-b"
    assert "sfxi" not in rows[0]


def test_sfxi_reference_overlay_rejects_misaligned_ids() -> None:
    result = score_vec8(
        np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=float),
        SFXIScoringConfig(setpoint_vector=(0.0, 0.0, 0.0, 1.0), scaling_min_n=1),
    )

    with pytest.raises(ValueError, match="design_id length"):
        to_sfxi_reference_overlay_records(result, design_id=["a", "b"])
