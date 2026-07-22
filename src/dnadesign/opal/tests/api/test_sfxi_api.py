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
    score_vec8_with_denom,
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


def test_score_vec8_with_denom_reuses_persisted_run_scale() -> None:
    vec8 = np.array(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]],
        dtype=float,
    )

    result = score_vec8_with_denom(
        vec8,
        SFXIScoringConfig(setpoint_vector=(0.0, 0.0, 0.0, 1.0)),
        denom=8.0,
    )

    assert result.denom_used == 8.0
    assert result.effect_raw[0] == 4.0
    assert result.effect_scaled[0] == 0.5
    assert result.sfxi[0] == 0.5


def test_score_vec8_rejects_intensity_values_that_overflow_linear_recovery() -> None:
    vec8 = np.array(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2048.0]],
        dtype=float,
    )

    with pytest.raises(ValueError, match="exceed stable score range"):
        score_vec8(
            vec8,
            SFXIScoringConfig(setpoint_vector=(0.0, 0.0, 0.0, 1.0), scaling_min_n=1),
        )


def test_score_vec8_is_monotone_for_setpoint_weighted_right_and_bright() -> None:
    cfg = SFXIScoringConfig(
        setpoint_vector=(0.0, 0.0, 0.0, 1.0),
        scaling_percentile=95,
        scaling_min_n=5,
        intensity_log2_offset_delta=0.0,
    )
    scaling_vec8 = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0],
        ],
        dtype=float,
    )

    desired_brightness = np.array(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, y11] for y11 in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]],
        dtype=float,
    )
    desired_result = score_vec8(desired_brightness, cfg, scaling_vec8=scaling_vec8)
    assert np.all(np.diff(desired_result.effect_raw) > 0.0)
    assert np.all(np.diff(desired_result.sfxi) >= 0.0)

    wrong_state_brightness = np.array(
        [[0.0, 0.0, 0.0, 1.0, 0.0, y10, 0.0, 2.0] for y10 in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]],
        dtype=float,
    )
    wrong_state_result = score_vec8(wrong_state_brightness, cfg, scaling_vec8=scaling_vec8)
    assert np.allclose(wrong_state_result.effect_raw, wrong_state_result.effect_raw[0])
    assert np.allclose(wrong_state_result.sfxi, wrong_state_result.sfxi[0])

    wrong_logic = np.array(
        [[0.0, v10, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0] for v10 in [0.0, 0.25, 0.5, 0.75, 1.0]],
        dtype=float,
    )
    wrong_logic_result = score_vec8(wrong_logic, cfg, scaling_vec8=scaling_vec8)
    assert np.all(np.diff(wrong_logic_result.logic_fidelity) < 0.0)
    assert np.all(np.diff(wrong_logic_result.sfxi) < 0.0)


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
