"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/api/test_sfxi_api.py

Regression tests for SFXI API OPAL API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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
    validate_sfxi_reference_overlay_records,
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
        setpoint_name="and",
        collection_id="reader_sfxi_pdual10_latest",
        batch_id="batch-0",
        campaign_id="stress_ethanol",
        reference_instance_id=["design-a", "design-b"],
        sequence_source_id=["usr-a", "usr-b"],
    )

    assert len(rows) == 2
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}metric_id"] == "sfxi_v1/and/sfxi"
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}metric_value"] == rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}sfxi"]
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}logic_fidelity"] == 1.0
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}effect_scaled"] == 1.0
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}collection_id"] == "reader_sfxi_pdual10_latest"
    assert rows[0][f"{SFXI_REFERENCE_OVERLAY_PREFIX}batch_id"] == "batch-0"
    assert rows[1][f"{SFXI_REFERENCE_OVERLAY_PREFIX}campaign_id"] == "stress_ethanol"
    assert rows[1][f"{SFXI_REFERENCE_OVERLAY_PREFIX}reference_instance_id"] == "design-b"
    assert rows[1][f"{SFXI_REFERENCE_OVERLAY_PREFIX}sequence_source_id"] == "usr-b"
    assert f"{SFXI_REFERENCE_OVERLAY_PREFIX}schema_version" not in rows[0]
    assert "sfxi" not in rows[0]
    summary = validate_sfxi_reference_overlay_records(
        rows,
        expected_setpoint_vector=(0.0, 0.0, 0.0, 1.0),
        metric_id="sfxi_v1/and/sfxi",
    )
    assert summary == {
        "schema_version": SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION,
        "namespace": "sfxi_ref",
        "row_count": 2,
        "metric_ids": ["sfxi_v1/and/sfxi"],
    }


def test_sfxi_reference_overlay_rejects_misaligned_ids() -> None:
    result = score_vec8(
        np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=float),
        SFXIScoringConfig(setpoint_vector=(0.0, 0.0, 0.0, 1.0), scaling_min_n=1),
    )

    with pytest.raises(ValueError, match="reference_instance_id length"):
        to_sfxi_reference_overlay_records(result, design_id=["a", "b"])


def test_sfxi_reference_overlay_validation_rejects_setpoint_drift() -> None:
    result = score_vec8(
        np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=float),
        SFXIScoringConfig(setpoint_vector=(0.0, 0.0, 0.0, 1.0), scaling_min_n=1),
    )
    rows = to_sfxi_reference_overlay_records(result, setpoint_name="and")

    with pytest.raises(ValueError, match="setpoint_vector"):
        validate_sfxi_reference_overlay_records(rows, expected_setpoint_vector=(0.0, 1.0, 0.0, 1.0))
