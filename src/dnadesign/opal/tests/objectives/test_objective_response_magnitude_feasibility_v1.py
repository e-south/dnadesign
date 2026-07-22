"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/objectives/test_objective_response_magnitude_feasibility_v1.py

Contract and adversarial tests for Response-Magnitude Feasibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal import (
    ResponseMagnitudeFeasibilityComponents,
    calibrate_response_magnitude_feasibility,
    score_response_magnitude_feasibility,
)
from dnadesign.opal.src.objectives.response_magnitude_feasibility_v1 import (
    response_magnitude_feasibility_components,
    response_magnitude_feasibility_v1,
)


def _params(target_mask: list[int]) -> dict[str, object]:
    return {
        "state_ids": ["00", "10", "01", "11"] if len(target_mask) == 4 else [f"s{i}" for i in range(len(target_mask))],
        "target_mask": target_mask,
        "calibration": {
            "response_separation_min": 0.0,
            "on_magnitude_min": 0.0,
            "off_magnitude_max": 0.0,
            "response_separation_scale": 1.0,
            "on_magnitude_scale": 1.0,
            "off_magnitude_scale": 1.0,
        },
    }


@pytest.mark.parametrize(
    ("target_mask", "response"),
    [
        ([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]),
        ([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]),
        ([0, 0, 0, 1], [0.0, 0.0, 0.0, 1.0]),
        ([0, 1, 1, 1], [0.0, 1.0, 1.0, 1.0]),
    ],
)
def test_components_support_representative_binary_masks(
    target_mask: list[int],
    response: list[float],
) -> None:
    magnitude = [0.5 if state_is_on else -1.0 for state_is_on in target_mask]
    prediction = np.asarray([response + magnitude], dtype=float)

    components = response_magnitude_feasibility_components(prediction, target_mask=target_mask)

    assert components.response_separation.tolist() == pytest.approx([1.0])
    assert components.on_magnitude_floor.tolist() == pytest.approx([0.5])
    assert components.off_magnitude_ceiling.tolist() == pytest.approx([-1.0])


def test_components_generalize_to_an_explicit_three_state_contract() -> None:
    prediction = np.asarray([[0.0, 2.0, 1.0, -0.5, 0.8, 0.2]], dtype=float)

    components = response_magnitude_feasibility_components(
        prediction,
        target_mask=[0, 1, 1],
    )

    assert components.response_separation.tolist() == pytest.approx([1.0])
    assert components.on_magnitude_floor.tolist() == pytest.approx([0.2])
    assert components.off_magnitude_ceiling.tolist() == pytest.approx([-0.5])


def test_components_are_equivariant_to_joint_state_permutation() -> None:
    values = np.asarray([[0.0, 2.0, 1.0, -0.5, 0.8, 0.2]], dtype=float)
    permuted = values[:, [2, 0, 1, 5, 3, 4]]

    direct = response_magnitude_feasibility_components(values, target_mask=[0, 1, 1])
    reordered = response_magnitude_feasibility_components(permuted, target_mask=[1, 0, 1])

    assert reordered.response_separation.tolist() == pytest.approx(direct.response_separation.tolist())
    assert reordered.on_magnitude_floor.tolist() == pytest.approx(direct.on_magnitude_floor.tolist())
    assert reordered.off_magnitude_ceiling.tolist() == pytest.approx(direct.off_magnitude_ceiling.tolist())


@pytest.mark.parametrize(
    "target_mask",
    [[int(bit) for bit in f"{value:04b}"] for value in range(1, 15)],
)
def test_feasibility_score_is_monotonic_in_every_declared_direction(target_mask: list[int]) -> None:
    rng = np.random.default_rng(20260712)
    values = rng.normal(size=(64, 8))
    calibration = _params(target_mask)["calibration"]
    baseline = score_response_magnitude_feasibility(
        values,
        target_mask=target_mask,
        calibration=calibration,
    ).feasibility_margin

    for state_index, state_is_on in enumerate(target_mask):
        for column_index in (state_index, state_index + len(target_mask)):
            improved = values.copy()
            improved[:, column_index] += 0.25 if state_is_on else -0.25
            rescored = score_response_magnitude_feasibility(
                improved,
                target_mask=target_mask,
                calibration=calibration,
            ).feasibility_margin
            assert np.all(rescored >= baseline - 1e-12)


@pytest.mark.parametrize(
    ("target_mask", "expected"),
    [
        ([0, 1, 0, 1], (1.0, 0.5, 0.2, -0.2)),
        ([0, 0, 1, 1], (-1.0, 0.2, 0.5, -1.0)),
        ([0, 0, 0, 1], (1.0, 0.8, 0.5, -0.5)),
        ([0, 1, 1, 1], (1.0, 0.2, -1.0, 0.2)),
    ],
)
def test_documented_observation_changes_only_through_target_partition(
    target_mask: list[int],
    expected: tuple[float, float, float, float],
) -> None:
    observation = np.asarray([[0.0, 2.0, 1.0, 3.0, -1.0, 0.5, 0.2, 0.8]])

    result = score_response_magnitude_feasibility(
        observation,
        target_mask=target_mask,
        calibration=_params(target_mask)["calibration"],
    )

    response_separation, on_magnitude_floor, off_magnitude_ceiling, feasibility = expected
    assert result.components.response_separation.tolist() == pytest.approx([response_separation])
    assert result.components.on_magnitude_floor.tolist() == pytest.approx([on_magnitude_floor])
    assert result.components.off_magnitude_ceiling.tolist() == pytest.approx([off_magnitude_ceiling])
    assert result.feasibility_margin.tolist() == pytest.approx([feasibility])


def test_objective_exposes_non_compensatory_components_and_maximin_score() -> None:
    predictions = np.asarray(
        [
            # Correct input-A response and adequate target-state magnitude.
            [0.0, 2.0, 0.2, 1.8, -1.0, 0.5, -0.8, 0.7],
            # Correct response, but too dim in one target-ON state.
            [0.0, 2.0, 0.2, 1.8, -1.0, -0.4, -0.8, -0.2],
            # Bright in target-ON states, but input-B-shaped response.
            [0.0, 0.2, 2.0, 1.8, -1.0, 0.8, 0.7, 0.9],
            # Correct response and ON magnitude, but excessive OFF magnitude.
            [0.0, 2.0, 0.2, 1.8, 0.6, 0.7, 0.8, 0.9],
        ],
        dtype=float,
    )

    result = response_magnitude_feasibility_v1(
        y_pred=predictions,
        params=_params([0, 1, 0, 1]),
        ctx=None,
        train_view=None,
        y_pred_std=None,
    )

    assert result.scores_by_name["response_separation"].tolist() == pytest.approx([1.6, 1.6, -1.8, 1.6])
    assert result.scores_by_name["on_magnitude_floor"].tolist() == pytest.approx([0.5, -0.4, 0.8, 0.7])
    assert result.scores_by_name["off_magnitude_ceiling"].tolist() == pytest.approx([-0.8, -0.8, 0.7, 0.8])
    assert result.scores_by_name["feasibility_margin"].tolist() == pytest.approx([0.5, -0.4, -1.8, -0.8])
    assert result.modes_by_name == {
        "feasibility_margin": "maximize",
        "response_separation": "maximize",
        "on_magnitude_floor": "maximize",
        "off_magnitude_ceiling": "minimize",
    }
    assert result.uncertainty_by_name == {}


def test_flat_response_has_zero_response_separation_without_minmax_rescaling() -> None:
    prediction = np.asarray([[1.0, 1.0, 1.0, 1.0, -1.0, 0.5, -1.0, 0.5]], dtype=float)

    components = response_magnitude_feasibility_components(prediction, target_mask=[0, 1, 0, 1])

    assert components.response_separation.tolist() == pytest.approx([0.0])


@pytest.mark.parametrize("target_mask", [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0.5, 0, 1]])
def test_target_mask_must_be_binary_and_contain_on_and_off_states(target_mask: list[float]) -> None:
    prediction = np.zeros((1, 8), dtype=float)

    with pytest.raises(ValueError, match="target_mask"):
        response_magnitude_feasibility_components(prediction, target_mask=target_mask)


@pytest.mark.parametrize(
    "prediction",
    [
        np.zeros((1, 7), dtype=float),
        np.zeros((1, 9), dtype=float),
        np.asarray([[0.0, 1.0, 0.0, 1.0, 0.0, np.nan, 0.0, 1.0]], dtype=float),
    ],
)
def test_prediction_contract_is_exact_raw8_and_finite(prediction: np.ndarray) -> None:
    with pytest.raises(ValueError, match="input"):
        response_magnitude_feasibility_components(prediction, target_mask=[0, 1, 0, 1])


def test_public_math_api_matches_objective_channels() -> None:
    predictions = np.asarray(
        [
            [0.0, 2.0, 0.2, 1.8, -1.0, 0.5, -0.8, 0.7],
            [0.0, 0.2, 2.0, 1.8, -1.0, 0.8, 0.7, 0.9],
        ],
        dtype=float,
    )
    params = _params([0, 1, 0, 1])
    result = response_magnitude_feasibility_v1(
        y_pred=predictions,
        params=params,
        ctx=None,
        train_view=None,
        y_pred_std=None,
    )

    public = score_response_magnitude_feasibility(
        predictions,
        target_mask=params["target_mask"],
        calibration=params["calibration"],
    )

    assert public.feasibility_margin.tolist() == pytest.approx(result.scores_by_name["feasibility_margin"].tolist())
    assert public.components.response_separation.tolist() == pytest.approx(
        result.scores_by_name["response_separation"].tolist()
    )


def test_public_calibration_normalizes_sequence_components_to_arrays() -> None:
    components = ResponseMagnitudeFeasibilityComponents(
        response_separation=[1.0],
        on_magnitude_floor=[0.5],
        off_magnitude_ceiling=[-0.25],
    )

    result = calibrate_response_magnitude_feasibility(
        components,
        calibration=_params([0, 1, 0, 1])["calibration"],
    )

    assert result.feasibility_margin.tolist() == pytest.approx([0.25])
    assert result.components.response_separation.tolist() == pytest.approx([1.0])


def test_public_calibration_rejects_non_vector_components() -> None:
    components = ResponseMagnitudeFeasibilityComponents(
        response_separation=np.ones((1, 1)),
        on_magnitude_floor=np.ones(1),
        off_magnitude_ceiling=np.ones(1),
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        calibrate_response_magnitude_feasibility(
            components,
            calibration=_params([0, 1, 0, 1])["calibration"],
        )


def test_objective_requires_explicit_calibration() -> None:
    prediction = np.zeros((1, 8), dtype=float)

    with pytest.raises(ValueError, match="calibration"):
        response_magnitude_feasibility_v1(
            y_pred=prediction,
            params={"target_mask": [0, 1, 0, 1]},
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )


@pytest.mark.parametrize(
    "state_ids",
    [["00", "10", "01"], ["00", "10", "10", "11"], ["00", "", "01", "11"]],
)
def test_objective_rejects_misaligned_or_ambiguous_state_ids(state_ids: list[str]) -> None:
    prediction = np.zeros((1, 8), dtype=float)
    params = _params([0, 1, 0, 1])
    params["state_ids"] = state_ids

    with pytest.raises(ValueError, match="state_ids"):
        response_magnitude_feasibility_v1(
            y_pred=prediction,
            params=params,
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )


def test_objective_rejects_nonpositive_calibration_scale() -> None:
    prediction = np.zeros((1, 8), dtype=float)
    params = _params([0, 1, 0, 1])
    assert isinstance(params["calibration"], dict)
    params["calibration"]["response_separation_scale"] = 0.0

    with pytest.raises(ValueError, match="response_separation_scale"):
        response_magnitude_feasibility_v1(
            y_pred=prediction,
            params=params,
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )
