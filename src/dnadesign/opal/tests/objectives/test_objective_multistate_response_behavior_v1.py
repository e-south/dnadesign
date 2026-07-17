"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/objectives/test_objective_multistate_response_behavior_v1.py

Contract and adversarial tests for Multistate Response Behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal import (
    MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION,
    MultistateResponseBehaviorScore,
    multistate_response_behavior_clearances,
    score_multistate_response_behavior,
)
from dnadesign.opal.src.config.plugin_schemas import validate_params
from dnadesign.opal.src.objectives.multistate_response_behavior_v1 import (
    multistate_response_behavior_v1,
)
from dnadesign.opal.src.registries.objectives import (
    get_objective_declared_channels,
    get_objective_family,
    get_objective_observed_replay_contract,
)


def _state_ids(state_count: int) -> list[str]:
    return [f"s{index}" for index in range(state_count)]


def _normalization() -> dict[str, float]:
    return {"response_scale": 2.0, "fluorescence_scale": 4.0}


def _params(target_mask: list[int]) -> dict[str, object]:
    return {
        "state_ids": _state_ids(len(target_mask)),
        "target_mask": target_mask,
        "normalization": _normalization(),
    }


def _score(values: np.ndarray, target_mask: list[int]) -> MultistateResponseBehaviorScore:
    return score_multistate_response_behavior(
        values,
        state_ids=_state_ids(len(target_mask)),
        target_mask=target_mask,
        normalization=_normalization(),
    )


def test_public_api_version_is_explicit() -> None:
    assert MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION == "1"


def test_documented_clearances_and_family_balanced_score() -> None:
    values = np.asarray([[0.0, 4.0, 2.0, -4.0, 4.0, -8.0]], dtype=float)

    result = _score(values, [0, 1, 0])

    np.testing.assert_allclose(result.clearances.response, [[2.0, 1.0]])
    np.testing.assert_allclose(result.clearances.on_expression, [[1.0]])
    np.testing.assert_allclose(result.clearances.off_suppression, [[1.0, 2.0]])
    assert result.coordinate_labels == (
        "response:s1>s0",
        "response:s1>s2",
        "on_expression:s1",
        "off_suppression:s0",
        "off_suppression:s2",
    )
    expected_response = -np.log(np.mean(np.exp(-np.asarray([2.0, 1.0]))))
    expected_off = -np.log(np.mean(np.exp(-np.asarray([1.0, 2.0]))))
    expected = -np.log(np.mean(np.exp(-np.asarray([expected_response, 1.0, expected_off]))))
    assert result.response_family_score.tolist() == pytest.approx([expected_response])
    assert result.on_expression_family_score.tolist() == pytest.approx([1.0])
    assert result.off_suppression_family_score.tolist() == pytest.approx([expected_off])
    assert result.behavior_score.tolist() == pytest.approx([expected])
    assert result.hard_bottleneck_clearance.tolist() == pytest.approx([1.0])
    assert result.limiting_coordinate_label == ("response:s1>s2",)
    assert result.all_reference_directions_met.tolist() == [True]
    np.testing.assert_allclose(np.sum(result.coordinate_weights, axis=1), np.ones(1))


@pytest.mark.parametrize("state_count", range(2, 17))
def test_every_desired_coordinate_improvement_strictly_increases_score(state_count: int) -> None:
    rng = np.random.default_rng(20260717 + state_count)
    target_mask = [int(index % 3 == 0) for index in range(state_count)]
    if all(target_mask) or not any(target_mask):  # pragma: no cover - defensive for changed mask recipe
        raise AssertionError("test mask recipe must contain ON and OFF states")
    values = rng.normal(size=(9, 2 * state_count))
    baseline = _score(values, target_mask).behavior_score

    for state_index, state_is_on in enumerate(target_mask):
        response_improved = values.copy()
        response_improved[:, state_index] += 0.125 if state_is_on else -0.125
        assert np.all(_score(response_improved, target_mask).behavior_score > baseline)

        fluorescence_improved = values.copy()
        fluorescence_improved[:, state_count + state_index] += 0.125 if state_is_on else -0.125
        assert np.all(_score(fluorescence_improved, target_mask).behavior_score > baseline)


def test_pareto_dominance_is_preserved() -> None:
    rng = np.random.default_rng(2101)
    target_mask = [0, 1, 0, 1, 0]
    dominated = rng.normal(size=(24, 10))
    dominant = dominated.copy()
    state_count = len(target_mask)
    for index, state_is_on in enumerate(target_mask):
        direction = 1.0 if state_is_on else -1.0
        dominant[:, index] += direction * 0.2
        dominant[:, state_count + index] += direction * 0.3

    assert np.all(_score(dominant, target_mask).behavior_score > _score(dominated, target_mask).behavior_score)


def test_joint_state_permutation_is_equivariant() -> None:
    rng = np.random.default_rng(1954)
    values = rng.normal(size=(17, 12))
    state_ids = ["basal", "a", "b", "ab", "c", "ac"]
    target_mask = [0, 1, 0, 1, 0, 1]
    permutation = np.asarray([4, 1, 5, 0, 3, 2], dtype=int)
    permuted = np.concatenate((values[:, permutation], values[:, 6 + permutation]), axis=1)

    direct = score_multistate_response_behavior(
        values,
        state_ids=state_ids,
        target_mask=target_mask,
        normalization=_normalization(),
    )
    reordered = score_multistate_response_behavior(
        permuted,
        state_ids=[state_ids[index] for index in permutation],
        target_mask=[target_mask[index] for index in permutation],
        normalization=_normalization(),
    )

    np.testing.assert_allclose(reordered.behavior_score, direct.behavior_score)
    np.testing.assert_allclose(reordered.hard_bottleneck_clearance, direct.hard_bottleneck_clearance)
    np.testing.assert_allclose(reordered.response_family_score, direct.response_family_score)
    np.testing.assert_allclose(reordered.on_expression_family_score, direct.on_expression_family_score)
    np.testing.assert_allclose(reordered.off_suppression_family_score, direct.off_suppression_family_score)


def test_family_means_prevent_coordinate_cardinality_from_reweighting_a_family() -> None:
    two_state = np.asarray([[0.0, 2.0, -4.0, 4.0]], dtype=float)
    repeated_off_state = np.asarray([[0.0, 0.0, 0.0, 2.0, -4.0, -4.0, -4.0, 4.0]], dtype=float)

    baseline = score_multistate_response_behavior(
        two_state,
        state_ids=["off", "on"],
        target_mask=[0, 1],
        normalization={"response_scale": 1.0, "fluorescence_scale": 1.0},
    )
    expanded = score_multistate_response_behavior(
        repeated_off_state,
        state_ids=["off-a", "off-b", "off-c", "on"],
        target_mask=[0, 0, 0, 1],
        normalization={"response_scale": 1.0, "fluorescence_scale": 1.0},
    )

    assert expanded.behavior_score.tolist() == pytest.approx(baseline.behavior_score.tolist())
    assert expanded.response_family_score.tolist() == pytest.approx(baseline.response_family_score.tolist())
    assert expanded.off_suppression_family_score.tolist() == pytest.approx(
        baseline.off_suppression_family_score.tolist()
    )


def test_equal_clearances_give_each_family_one_third_total_weight() -> None:
    result = score_multistate_response_behavior(
        np.zeros((1, 8), dtype=float),
        state_ids=["off-a", "on", "off-b", "off-c"],
        target_mask=[0, 1, 0, 0],
        normalization={"response_scale": 1.0, "fluorescence_scale": 1.0},
    )
    response_count = result.clearances.response.shape[1]
    on_count = result.clearances.on_expression.shape[1]

    assert np.sum(result.coordinate_weights[:, :response_count], axis=1).tolist() == pytest.approx([1.0 / 3.0])
    assert np.sum(
        result.coordinate_weights[:, response_count : response_count + on_count], axis=1
    ).tolist() == pytest.approx([1.0 / 3.0])
    assert np.sum(result.coordinate_weights[:, response_count + on_count :], axis=1).tolist() == pytest.approx(
        [1.0 / 3.0]
    )


def test_extreme_finite_values_remain_finite_and_weighted() -> None:
    values = np.asarray(
        [
            [-1.0e300, 1.0e300, -1.0e300, 1.0e300],
            [1.0e300, -1.0e300, 1.0e300, -1.0e300],
        ],
        dtype=float,
    )

    result = score_multistate_response_behavior(
        values,
        state_ids=["off", "on"],
        target_mask=[0, 1],
        normalization={"response_scale": 1.0e-100, "fluorescence_scale": 1.0e-100},
    )

    for array in (
        result.coordinate_clearances,
        result.coordinate_weights,
        result.behavior_score,
        result.hard_bottleneck_clearance,
        result.response_family_score,
        result.on_expression_family_score,
        result.off_suppression_family_score,
    ):
        assert np.all(np.isfinite(array))
    np.testing.assert_allclose(np.sum(result.coordinate_weights, axis=1), np.ones(2))


def test_smooth_compensation_above_hard_bottleneck_is_bounded() -> None:
    values = np.asarray([[0.0, 1.0, -1.0, 1.0]], dtype=float)
    result = score_multistate_response_behavior(
        values,
        state_ids=["off", "on"],
        target_mask=[0, 1],
        normalization={"response_scale": 1.0, "fluorescence_scale": 1.0},
    )
    minimum_prior_weight = 1.0 / 3.0

    gap = result.behavior_score - result.hard_bottleneck_clearance
    assert np.all(gap >= 0.0)
    assert np.all(gap <= -np.log(minimum_prior_weight) + 1.0e-12)


def test_positive_behavior_score_is_not_a_feasibility_claim() -> None:
    values = np.asarray([[0.0, -0.1, -0.2, 10.0]], dtype=float)

    result = score_multistate_response_behavior(
        values,
        state_ids=["off", "on"],
        target_mask=[0, 1],
        normalization={"response_scale": 1.0, "fluorescence_scale": 1.0},
    )

    assert result.behavior_score[0] > 0.0
    assert result.hard_bottleneck_clearance[0] < 0.0
    assert result.all_reference_directions_met.tolist() == [False]


@pytest.mark.parametrize(
    ("target_mask", "match"),
    [
        ([0, 0], "ON and one OFF"),
        ([1, 1], "ON and one OFF"),
        ([0, 0.5], "binary"),
        ([False, True], "not boolean aliases"),
    ],
)
def test_target_mask_is_strict_binary_partition(target_mask: list[object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        score_multistate_response_behavior(
            np.zeros((1, 4), dtype=float),
            state_ids=["a", "b"],
            target_mask=target_mask,
            normalization=_normalization(),
        )


def test_target_mask_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        score_multistate_response_behavior(
            np.zeros((1, 4), dtype=float),
            state_ids=["off", "on"],
            target_mask=[[0, 1]],  # type: ignore[list-item]
            normalization=_normalization(),
        )


@pytest.mark.parametrize(
    "normalization",
    [
        {},
        {"response_scale": 1.0, "fluorescence_scale": 1.0, "threshold": 0.0},
        {"response_scale": True, "fluorescence_scale": 1.0},
        {"response_scale": 0.0, "fluorescence_scale": 1.0},
        {"response_scale": 1.0, "fluorescence_scale": np.inf},
    ],
)
def test_normalization_requires_exact_positive_finite_scales(normalization: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="normalization"):
        score_multistate_response_behavior(
            np.zeros((1, 4), dtype=float),
            state_ids=["a", "b"],
            target_mask=[0, 1],
            normalization=normalization,
        )


@pytest.mark.parametrize(
    "state_ids",
    [["a"], ["a", "a"], ["a", ""], ["a", 2]],
)
def test_state_ids_are_aligned_unique_and_nonempty(state_ids: list[object]) -> None:
    with pytest.raises(ValueError, match="state_ids"):
        score_multistate_response_behavior(
            np.zeros((1, 4), dtype=float),
            state_ids=state_ids,
            target_mask=[0, 1],
            normalization=_normalization(),
        )


@pytest.mark.parametrize(
    "values",
    [np.zeros((1, 3)), np.zeros((1, 5)), np.full((1, 4), np.nan), np.zeros((0, 4))],
)
def test_input_contract_is_exact_2k_nonempty_and_finite(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="input"):
        score_multistate_response_behavior(
            values,
            state_ids=["a", "b"],
            target_mask=[0, 1],
            normalization=_normalization(),
        )


def test_plugin_exposes_only_ledger_safe_scalar_channels_and_diagnostics() -> None:
    values = np.asarray(
        [
            [0.0, 2.0, 1.0, -1.0, 2.0, -1.0],
            [1.0, 0.0, 2.0, 1.0, -1.0, 2.0],
        ],
        dtype=float,
    )
    params = _params([0, 1, 0])

    result = multistate_response_behavior_v1(
        y_pred=values,
        params=params,
        ctx=None,
        train_view=None,
        y_pred_std=np.ones_like(values),
    )

    assert set(result.scores_by_name) == {"behavior_score"}
    assert set(result.modes_by_name) == {"behavior_score"}
    assert set(result.modes_by_name.values()) == {"maximize"}
    assert result.uncertainty_by_name == {}
    for name in (
        "hard_bottleneck_clearance",
        "response_family_score",
        "on_expression_family_score",
        "off_suppression_family_score",
        "all_reference_directions_met",
        "limiting_coordinate_index",
        "limiting_coordinate_weight",
    ):
        diagnostic = np.asarray(result.diagnostics[name])
        assert diagnostic.ndim == 1
        assert len(diagnostic) == len(values)
        assert np.issubdtype(diagnostic.dtype, np.number)
    assert result.diagnostics["uncertainty_emitted"] is False
    assert "feasible" not in result.diagnostics


def test_plugin_registry_contract_and_config_schema_are_explicit() -> None:
    assert get_objective_family("multistate_response_behavior_v1") == "multistate_response_behavior"
    assert get_objective_observed_replay_contract("multistate_response_behavior_v1") == "pointwise_params_v1"
    declared = get_objective_declared_channels("multistate_response_behavior_v1")
    assert declared["score"] == ("behavior_score",)
    assert set(declared["score_modes"].values()) == {"maximize"}
    assert declared["uncertainty"] == ()

    assert validate_params("objective", "multistate_response_behavior_v1", _params([0, 1])) == _params([0, 1])
    with pytest.raises(Exception, match="extra"):
        validate_params(
            "objective",
            "multistate_response_behavior_v1",
            {**_params([0, 1]), "temperature": 2.0},
        )
    with pytest.raises(Exception, match="boolean"):
        validate_params(
            "objective",
            "multistate_response_behavior_v1",
            {
                **_params([0, 1]),
                "normalization": {"response_scale": True, "fluorescence_scale": 1.0},
            },
        )


def test_clearance_builder_is_public_and_matches_scoring_result() -> None:
    values = np.asarray([[0.0, 1.0, 2.0, -1.0, 2.0, -3.0]], dtype=float)
    kwargs = {
        "state_ids": ["off", "on", "also-off"],
        "target_mask": [0, 1, 0],
        "normalization": _normalization(),
    }

    clearances = multistate_response_behavior_clearances(values, **kwargs)
    scored = score_multistate_response_behavior(values, **kwargs)

    np.testing.assert_allclose(clearances.coordinate_clearances, scored.coordinate_clearances)
    assert clearances.coordinate_labels == scored.coordinate_labels
