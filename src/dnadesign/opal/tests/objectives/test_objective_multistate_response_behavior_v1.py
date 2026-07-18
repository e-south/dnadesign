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
    return {"response_scale": 2.0, "signal_scale": 4.0}


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


def _family_level_vector(
    target_mask: list[int],
    *,
    response: float,
    on_signal: float,
    off_suppression: float,
) -> np.ndarray:
    """Build one row with a constant normalized level inside each family."""

    target = np.asarray(target_mask, dtype=bool)
    response_values = np.where(target, response / 2.0, -response / 2.0)
    signal_values = np.where(target, on_signal, -off_suppression)
    return np.concatenate((response_values, signal_values))[None, :]


def _ideal_prototype(target_mask: list[int]) -> np.ndarray:
    return _family_level_vector(
        target_mask,
        response=4.0,
        on_signal=2.0,
        off_suppression=2.0,
    )


def test_public_api_version_is_explicit() -> None:
    assert MULTISTATE_RESPONSE_BEHAVIOR_API_VERSION == "1"


def test_documented_clearances_and_family_balanced_score() -> None:
    values = np.asarray([[0.0, 4.0, 2.0, -4.0, 4.0, -8.0]], dtype=float)

    result = _score(values, [0, 1, 0])

    np.testing.assert_allclose(result.clearances.response, [[2.0, 1.0]])
    np.testing.assert_allclose(result.clearances.on_signal, [[1.0]])
    np.testing.assert_allclose(result.clearances.off_signal_suppression, [[1.0, 2.0]])
    assert result.coordinate_labels == (
        "response:s1>s0",
        "response:s1>s2",
        "on_signal:s1",
        "off_signal_suppression:s0",
        "off_signal_suppression:s2",
    )
    expected_response = -np.log(np.mean(np.exp(-np.asarray([2.0, 1.0]))))
    expected_off = -np.log(np.mean(np.exp(-np.asarray([1.0, 2.0]))))
    expected = -np.log(np.mean(np.exp(-np.asarray([expected_response, 1.0, expected_off]))))
    assert result.response_family_score.tolist() == pytest.approx([expected_response])
    assert result.on_signal_family_score.tolist() == pytest.approx([1.0])
    assert result.off_signal_suppression_family_score.tolist() == pytest.approx([expected_off])
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

        signal_improved = values.copy()
        signal_improved[:, state_count + state_index] += 0.125 if state_is_on else -0.125
        assert np.all(_score(signal_improved, target_mask).behavior_score > baseline)


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
    np.testing.assert_allclose(reordered.on_signal_family_score, direct.on_signal_family_score)
    np.testing.assert_allclose(
        reordered.off_signal_suppression_family_score,
        direct.off_signal_suppression_family_score,
    )


def test_family_means_prevent_coordinate_cardinality_from_reweighting_a_family() -> None:
    two_state = np.asarray([[0.0, 2.0, -4.0, 4.0]], dtype=float)
    repeated_off_state = np.asarray([[0.0, 0.0, 0.0, 2.0, -4.0, -4.0, -4.0, 4.0]], dtype=float)

    baseline = score_multistate_response_behavior(
        two_state,
        state_ids=["off", "on"],
        target_mask=[0, 1],
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )
    expanded = score_multistate_response_behavior(
        repeated_off_state,
        state_ids=["off-a", "off-b", "off-c", "on"],
        target_mask=[0, 0, 0, 1],
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )

    assert expanded.behavior_score.tolist() == pytest.approx(baseline.behavior_score.tolist())
    assert expanded.response_family_score.tolist() == pytest.approx(baseline.response_family_score.tolist())
    assert expanded.off_signal_suppression_family_score.tolist() == pytest.approx(
        baseline.off_signal_suppression_family_score.tolist()
    )


def test_equal_clearances_give_each_family_one_third_total_weight() -> None:
    result = score_multistate_response_behavior(
        np.zeros((1, 8), dtype=float),
        state_ids=["off-a", "on", "off-b", "off-c"],
        target_mask=[0, 1, 0, 0],
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )
    response_count = result.clearances.response.shape[1]
    on_count = result.clearances.on_signal.shape[1]

    assert np.sum(result.coordinate_weights[:, :response_count], axis=1).tolist() == pytest.approx([1.0 / 3.0])
    assert np.sum(
        result.coordinate_weights[:, response_count : response_count + on_count], axis=1
    ).tolist() == pytest.approx([1.0 / 3.0])
    assert np.sum(result.coordinate_weights[:, response_count + on_count :], axis=1).tolist() == pytest.approx(
        [1.0 / 3.0]
    )


@pytest.mark.parametrize("target_mask", ([0, 1, 0, 1], [0, 0, 0, 1]))
def test_asymmetric_masks_preserve_equal_family_standing(target_mask: list[int]) -> None:
    scores: list[float] = []
    for levels in ((-1.0, 2.0, 2.0), (2.0, -1.0, 2.0), (2.0, 2.0, -1.0)):
        result = score_multistate_response_behavior(
            _family_level_vector(
                target_mask,
                response=levels[0],
                on_signal=levels[1],
                off_suppression=levels[2],
            ),
            state_ids=["00", "10", "01", "11"],
            target_mask=target_mask,
            normalization={"response_scale": 1.0, "signal_scale": 1.0},
        )
        family_scores = (
            float(result.response_family_score[0]),
            float(result.on_signal_family_score[0]),
            float(result.off_signal_suppression_family_score[0]),
        )
        assert family_scores == pytest.approx(levels)
        assert float(result.hard_bottleneck_clearance[0]) == pytest.approx(-1.0)
        scores.append(float(result.behavior_score[0]))

    assert scores == pytest.approx([scores[0]] * 3)


def test_stress_view_prototypes_do_not_collapse_to_one_shared_axis() -> None:
    masks = {
        "ethanol": [0, 1, 0, 1],
        "ciprofloxacin": [0, 0, 1, 1],
        "and": [0, 0, 0, 1],
    }
    state_ids = ["00", "10", "01", "11"]

    for prototype_id, prototype_mask in masks.items():
        scores = {
            view_id: float(
                score_multistate_response_behavior(
                    _ideal_prototype(prototype_mask),
                    state_ids=state_ids,
                    target_mask=view_mask,
                    normalization={"response_scale": 1.0, "signal_scale": 1.0},
                ).behavior_score[0]
            )
            for view_id, view_mask in masks.items()
        }
        assert max(scores, key=scores.get) == prototype_id  # type: ignore[arg-type]
        assert list(scores.values()).count(scores[prototype_id]) == 1


@pytest.mark.parametrize("target_mask", ([0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]))
def test_stress_views_penalize_each_distinct_behavior_failure(target_mask: list[int]) -> None:
    state_ids = ["00", "10", "01", "11"]
    baseline_values = _ideal_prototype(target_mask)
    baseline = score_multistate_response_behavior(
        baseline_values,
        state_ids=state_ids,
        target_mask=target_mask,
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )
    state_count = len(target_mask)
    on_index = target_mask.index(1)
    off_index = target_mask.index(0)

    perturbations = {
        "response": (on_index, -1.0),
        "on_signal": (state_count + on_index, -1.0),
        "off_signal_suppression": (state_count + off_index, 1.0),
    }
    family_fields = {
        "response": "response_family_score",
        "on_signal": "on_signal_family_score",
        "off_signal_suppression": "off_signal_suppression_family_score",
    }
    for affected_family, (column, delta) in perturbations.items():
        perturbed_values = baseline_values.copy()
        perturbed_values[0, column] += delta
        perturbed = score_multistate_response_behavior(
            perturbed_values,
            state_ids=state_ids,
            target_mask=target_mask,
            normalization={"response_scale": 1.0, "signal_scale": 1.0},
        )

        assert perturbed.behavior_score[0] < baseline.behavior_score[0]
        for family, field in family_fields.items():
            before = float(getattr(baseline, field)[0])
            after = float(getattr(perturbed, field)[0])
            if family == affected_family:
                assert after < before
            else:
                assert after == pytest.approx(before)


@pytest.mark.parametrize(
    ("target_mask", "family_sizes"),
    [
        ([0, 1, 0, 1], (4, 2, 2)),
        ([0, 0, 1, 1], (4, 2, 2)),
        ([0, 0, 0, 1], (3, 1, 3)),
    ],
)
def test_current_stress_masks_expose_analytic_coordinate_priors_and_compensation_bounds(
    target_mask: list[int],
    family_sizes: tuple[int, int, int],
) -> None:
    result = score_multistate_response_behavior(
        _family_level_vector(
            target_mask,
            response=-1.0,
            on_signal=2.0,
            off_suppression=2.0,
        ),
        state_ids=["00", "10", "01", "11"],
        target_mask=target_mask,
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )
    expected_priors = np.concatenate([np.full(count, 1.0 / (3.0 * count)) for count in family_sizes])

    np.testing.assert_allclose(result.coordinate_prior_weights, expected_priors)
    assert result.compensation_gap[0] >= 0.0
    assert result.compensation_gap[0] <= result.maximum_compensation_gap[0] + 1.0e-12
    limiting_prior = expected_priors[result.limiting_coordinate_index[0]]
    assert result.maximum_compensation_gap[0] == pytest.approx(-np.log(limiting_prior))


@pytest.mark.parametrize("target_mask", ([0, 1, 0, 1], [0, 0, 0, 1]))
def test_normalization_covariance_preserves_scores_and_diagnostics(target_mask: list[int]) -> None:
    rng = np.random.default_rng(20260718)
    values = rng.normal(size=(7, 8))
    baseline = score_multistate_response_behavior(
        values,
        state_ids=["00", "10", "01", "11"],
        target_mask=target_mask,
        normalization={"response_scale": 2.0, "signal_scale": 3.0},
    )
    scaled = values.copy()
    scaled[:, :4] *= 7.0
    scaled[:, 4:] *= 0.25
    transformed = score_multistate_response_behavior(
        scaled,
        state_ids=["00", "10", "01", "11"],
        target_mask=target_mask,
        normalization={"response_scale": 14.0, "signal_scale": 0.75},
    )

    for field in (
        "coordinate_clearances",
        "coordinate_weights",
        "behavior_score",
        "hard_bottleneck_clearance",
        "compensation_gap",
        "maximum_compensation_gap",
    ):
        np.testing.assert_allclose(getattr(transformed, field), getattr(baseline, field))
    assert transformed.limiting_coordinate_label == baseline.limiting_coordinate_label


def test_uniform_state_replication_is_invariant_but_selective_duplication_is_not() -> None:
    baseline_values = np.asarray([[-1.0, 0.0, 2.0, 0.5, -1.0, -2.0]], dtype=float)
    baseline = score_multistate_response_behavior(
        baseline_values,
        state_ids=["off-a", "off-b", "on"],
        target_mask=[0, 0, 1],
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )
    uniformly_replicated = score_multistate_response_behavior(
        np.asarray(
            [[-1.0, -1.0, 0.0, 0.0, 2.0, 2.0, 0.5, 0.5, -1.0, -1.0, -2.0, -2.0]],
            dtype=float,
        ),
        state_ids=["off-a-1", "off-a-2", "off-b-1", "off-b-2", "on-1", "on-2"],
        target_mask=[0, 0, 0, 0, 1, 1],
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )
    selectively_duplicated = score_multistate_response_behavior(
        np.asarray([[-1.0, -1.0, 0.0, 2.0, 0.5, 0.5, -1.0, -2.0]], dtype=float),
        state_ids=["off-a-1", "off-a-2", "off-b", "on"],
        target_mask=[0, 0, 0, 1],
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )

    assert uniformly_replicated.behavior_score.tolist() == pytest.approx(baseline.behavior_score.tolist())
    assert selectively_duplicated.behavior_score.tolist() != pytest.approx(baseline.behavior_score.tolist())


@pytest.mark.parametrize("state_count", (2, 4, 8, 16))
@pytest.mark.parametrize("weak_family", ("response", "on_signal", "off_signal_suppression"))
def test_cardinality_pressure_exposes_weak_coordinate_dilution(
    state_count: int,
    weak_family: str,
) -> None:
    on_counts = sorted({1, state_count // 2, state_count - 1})
    for on_count in on_counts:
        target_mask = [1] * on_count + [0] * (state_count - on_count)
        on_indices = np.flatnonzero(target_mask)
        off_indices = np.flatnonzero(np.logical_not(target_mask))
        response = np.where(np.asarray(target_mask, dtype=bool), 10.0, -10.0)
        signal = np.where(np.asarray(target_mask, dtype=bool), 10.0, -10.0)
        if weak_family == "response":
            response[on_indices[0]] = 0.0
            response[off_indices[0]] = 1.0
            expected_family_size = len(on_indices) * len(off_indices)
        elif weak_family == "on_signal":
            signal[on_indices[0]] = -1.0
            expected_family_size = len(on_indices)
        else:
            signal[off_indices[0]] = 1.0
            expected_family_size = len(off_indices)

        result = score_multistate_response_behavior(
            np.concatenate((response, signal))[None, :],
            state_ids=_state_ids(state_count),
            target_mask=target_mask,
            normalization={"response_scale": 1.0, "signal_scale": 1.0},
        )
        limiting_index = int(result.limiting_coordinate_index[0])
        limiting_prior = float(result.coordinate_prior_weights[limiting_index])

        assert result.hard_bottleneck_clearance.tolist() == pytest.approx([-1.0])
        assert limiting_prior == pytest.approx(1.0 / (3.0 * expected_family_size))
        assert result.maximum_compensation_gap.tolist() == pytest.approx([np.log(3.0 * expected_family_size)])
        assert 0.0 <= result.compensation_gap[0] <= result.maximum_compensation_gap[0] + 1.0e-12


@pytest.mark.parametrize(
    ("target_mask", "on_coordinate_prior"),
    [([0, 1, 0, 1], 1.0 / 6.0), ([0, 0, 0, 1], 1.0 / 3.0)],
)
def test_one_arbitrarily_favorable_coordinate_has_a_finite_score_gain(
    target_mask: list[int],
    on_coordinate_prior: float,
) -> None:
    neutral = np.zeros((1, 8), dtype=float)
    superstar = neutral.copy()
    superstar[0, 4 + target_mask.index(1)] = 1.0e6
    result = score_multistate_response_behavior(
        superstar,
        state_ids=["00", "10", "01", "11"],
        target_mask=target_mask,
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )

    assert result.behavior_score[0] == pytest.approx(-np.log(1.0 - on_coordinate_prior))


def test_one_family_superstar_does_not_outrank_balanced_behavior() -> None:
    target_mask = [0, 1, 0, 1]
    balanced = score_multistate_response_behavior(
        _family_level_vector(target_mask, response=1.0, on_signal=1.0, off_suppression=1.0),
        state_ids=["00", "10", "01", "11"],
        target_mask=target_mask,
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )
    bright_but_leaky = score_multistate_response_behavior(
        _family_level_vector(target_mask, response=1.0, on_signal=100.0, off_suppression=-1.0),
        state_ids=["00", "10", "01", "11"],
        target_mask=target_mask,
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
    )

    assert balanced.behavior_score[0] == pytest.approx(1.0)
    assert bright_but_leaky.behavior_score[0] < 0.0
    assert bright_but_leaky.behavior_score[0] < balanced.behavior_score[0]


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
        normalization={"response_scale": 1.0e-100, "signal_scale": 1.0e-100},
    )

    for array in (
        result.coordinate_clearances,
        result.coordinate_weights,
        result.behavior_score,
        result.hard_bottleneck_clearance,
        result.response_family_score,
        result.on_signal_family_score,
        result.off_signal_suppression_family_score,
    ):
        assert np.all(np.isfinite(array))
    np.testing.assert_allclose(np.sum(result.coordinate_weights, axis=1), np.ones(2))


@pytest.mark.parametrize("scale", (1.0e-300, 1.0, 1.0e300))
def test_extreme_desired_direction_moves_never_decrease_score(scale: float) -> None:
    target_mask = [0, 1, 0, 1]
    state_ids = ["00", "10", "01", "11"]
    values = np.asarray(
        [[-1.0e300, -1.0e100, 1.0e100, 1.0e300, -1.0e300, 0.0, 1.0e100, 1.0e300]],
        dtype=float,
    )
    normalization = {"response_scale": scale, "signal_scale": scale}
    baseline = score_multistate_response_behavior(
        values,
        state_ids=state_ids,
        target_mask=target_mask,
        normalization=normalization,
    ).behavior_score

    for state_index, state_is_on in enumerate(target_mask):
        direction = np.inf if state_is_on else -np.inf
        for column in (state_index, len(target_mask) + state_index):
            improved = values.copy()
            improved[0, column] = np.nextafter(improved[0, column], direction)
            score = score_multistate_response_behavior(
                improved,
                state_ids=state_ids,
                target_mask=target_mask,
                normalization=normalization,
            ).behavior_score
            assert np.all(score >= baseline)


def test_smooth_compensation_above_hard_bottleneck_is_bounded() -> None:
    values = np.asarray([[0.0, 1.0, -1.0, 1.0]], dtype=float)
    result = score_multistate_response_behavior(
        values,
        state_ids=["off", "on"],
        target_mask=[0, 1],
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
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
        normalization={"response_scale": 1.0, "signal_scale": 1.0},
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
        {"response_scale": 1.0, "signal_scale": 1.0, "threshold": 0.0},
        {"response_scale": True, "signal_scale": 1.0},
        {"response_scale": 0.0, "signal_scale": 1.0},
        {"response_scale": 1.0, "signal_scale": np.inf},
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


@pytest.mark.parametrize("state_ids", [["a", " a"], ["a", "b "], ["a", "   "]])
def test_state_ids_reject_whitespace_normalization(state_ids: list[str]) -> None:
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
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
        "compensation_gap",
        "maximum_compensation_gap",
        "response_family_score",
        "on_signal_family_score",
        "off_signal_suppression_family_score",
        "all_reference_directions_met",
        "limiting_coordinate_index",
        "limiting_coordinate_prior_weight",
        "limiting_coordinate_bottleneck_weight",
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
                "normalization": {"response_scale": True, "signal_scale": 1.0},
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
