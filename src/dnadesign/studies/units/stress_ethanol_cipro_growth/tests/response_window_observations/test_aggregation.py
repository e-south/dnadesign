"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_aggregation.py

Scientific-contract tests for candidate-level response-window observation aggregation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import (
    aggregation,
)

VALUE_COLUMNS = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


def test_selected_repeat_uses_only_the_explicit_label_source() -> None:
    measurements = _measurements(
        ("candidate-a", "design-a", "experiment-a", 0.0),
        ("candidate-a", "design-a", "experiment-b", 10.0),
    )
    draws = _draws(measurements, offsets=(0.0, 0.25, -0.25))

    preview = aggregation.aggregate_response_window_observations(
        measurements,
        draws,
        policy=_policy(bootstrap_samples=200),
        repeat_decisions=_decisions(
            (
                "candidate-a",
                "design-a",
                ("experiment-a", "experiment-b"),
                "label_source_selected",
                "experiment-b",
            )
        ),
    )

    assert preview.blockers == ()
    assert preview.observations.loc[0, list(VALUE_COLUMNS)].tolist() == [10.0] * len(VALUE_COLUMNS)
    assert preview.observations.loc[0, "label_source_method"] == "explicit_repeat_selection"
    assert preview.observations.loc[0, "reader_experiment_count"] == 2
    assert preview.observations.loc[0, "label_source_reader_experiment_id"] == "experiment-b"
    assert preview.contributions.groupby("reader_experiment_id")["selected_as_label_source"].first().to_dict() == {
        "experiment-a": False,
        "experiment-b": True,
    }
    assert preview.contributions.groupby("reader_experiment_id")["included_in_label"].first().to_dict() == {
        "experiment-a": False,
        "experiment-b": True,
    }


def test_repeat_source_selection_is_explicit_not_chronology_inferred() -> None:
    measurements = _measurements(
        ("candidate-a", "design-a", "experiment-a", 0.0),
        ("candidate-a", "design-a", "experiment-b", 3.0),
        ("candidate-a", "design-a", "experiment-c", 100.0),
    )
    draws = _draws(measurements, offsets=(0.0,))

    preview = aggregation.aggregate_response_window_observations(
        measurements,
        draws,
        policy=_policy(bootstrap_samples=100),
        repeat_decisions=_decisions(
            (
                "candidate-a",
                "design-a",
                ("experiment-a", "experiment-b", "experiment-c"),
                "label_source_selected",
                "experiment-b",
            )
        ),
    )

    assert preview.observations.loc[0, list(VALUE_COLUMNS)].tolist() == [3.0] * len(VALUE_COLUMNS)
    assert preview.observations.loc[0, "label_source_method"] == "explicit_repeat_selection"
    assert preview.observations.loc[0, "label_source_reader_experiment_id"] == "experiment-b"


def test_selected_source_bootstrap_is_deterministic_and_keeps_joint_reader_draws() -> None:
    measurements = _measurements(
        ("candidate-a", "design-a", "experiment-a", 0.0),
        ("candidate-a", "design-a", "experiment-b", 10.0),
    )
    draws = _joint_draw_fixture(measurements)
    kwargs = {
        "policy": _policy(bootstrap_samples=250, random_seed=17),
        "repeat_decisions": _decisions(
            (
                "candidate-a",
                "design-a",
                ("experiment-a", "experiment-b"),
                "label_source_selected",
                "experiment-b",
            )
        ),
    }

    first = aggregation.aggregate_response_window_observations(measurements, draws, **kwargs)
    second = aggregation.aggregate_response_window_observations(measurements, draws, **kwargs)

    pd.testing.assert_frame_equal(first.uncertainty, second.uncertainty)
    pd.testing.assert_frame_equal(first.bootstrap_draws, second.bootstrap_draws)
    assert np.allclose(
        first.bootstrap_draws["r10"] - first.bootstrap_draws["r00"],
        first.bootstrap_draws["b10"] - first.bootstrap_draws["b00"],
    )


def test_unresolved_repeat_is_a_label_truth_blocker_not_an_implicit_average() -> None:
    measurements = _measurements(
        ("candidate-a", "design-a", "experiment-a", 0.0),
        ("candidate-a", "design-a", "experiment-b", 10.0),
        ("candidate-b", "design-b", "experiment-a", 2.0),
    )
    draws = _draws(measurements, offsets=(0.0,))

    preview = aggregation.aggregate_response_window_observations(
        measurements,
        draws,
        policy=_policy(bootstrap_samples=100),
        repeat_decisions=_decisions(
            ("candidate-a", "design-a", ("experiment-a", "experiment-b"), "review_required", None)
        ),
    )

    assert preview.observations["candidate_id"].tolist() == ["candidate-b"]
    assert preview.blockers == ("candidate-a: repeated experiments require an explicit label-source decision",)
    assert (
        preview.contributions.loc[preview.contributions["candidate_id"].eq("candidate-a"), "included_in_label"]
        .eq(False)
        .all()
    )


def test_repeat_policy_requires_exact_candidate_and_design_accounting() -> None:
    measurements = _measurements(
        ("candidate-a", "design-a", "experiment-a", 0.0),
        ("candidate-a", "design-a", "experiment-b", 1.0),
    )
    draws = _draws(measurements, offsets=(0.0,))

    with pytest.raises(aggregation.ResponseWindowAggregationError, match="missing repeated candidates"):
        aggregation.aggregate_response_window_observations(
            measurements,
            draws,
            policy=_policy(bootstrap_samples=100),
            repeat_decisions=pd.DataFrame(columns=aggregation.DECISION_COLUMNS),
        )

    with pytest.raises(aggregation.ResponseWindowAggregationError, match="design aliases disagree"):
        aggregation.aggregate_response_window_observations(
            measurements,
            draws,
            policy=_policy(bootstrap_samples=100),
            repeat_decisions=_decisions(
                (
                    "candidate-a",
                    "wrong-design",
                    ("experiment-a", "experiment-b"),
                    "label_source_selected",
                    "experiment-b",
                )
            ),
        )

    with pytest.raises(aggregation.ResponseWindowAggregationError, match="experiment identities disagree"):
        aggregation.aggregate_response_window_observations(
            measurements,
            draws,
            policy=_policy(bootstrap_samples=100),
            repeat_decisions=_decisions(
                (
                    "candidate-a",
                    "design-a",
                    ("experiment-a", "experiment-new"),
                    "label_source_selected",
                    "experiment-new",
                )
            ),
        )


def test_bootstrap_draws_must_match_exact_design_experiment_contributions() -> None:
    measurements = _measurements(
        ("candidate-a", "design-a", "experiment-a", 0.0),
        ("candidate-a", "design-a", "experiment-b", 1.0),
    )
    draws = _draws(measurements, offsets=(0.0,))
    draws.loc[draws["reader_experiment_id"].eq("experiment-b"), "design_id"] = "wrong-design"

    with pytest.raises(aggregation.ResponseWindowAggregationError, match="coverage disagrees"):
        aggregation.aggregate_response_window_observations(
            measurements,
            draws,
            policy=_policy(bootstrap_samples=100),
            repeat_decisions=_decisions(
                (
                    "candidate-a",
                    "design-a",
                    ("experiment-a", "experiment-b"),
                    "label_source_selected",
                    "experiment-b",
                )
            ),
        )


def test_reduction_and_event_time_sensitivity_remain_separate() -> None:
    measurements = _measurements(("candidate-a", "design-a", "experiment-a", 2.0))

    preview = aggregation.aggregate_response_window_observations(
        measurements,
        _draws(measurements, offsets=(0.0,)),
        policy=_policy(bootstrap_samples=100),
        repeat_decisions=pd.DataFrame(columns=aggregation.DECISION_COLUMNS),
    )

    assert set(preview.reduction_sensitivity["reduction_id"]) == {"primary", "event_sensitivity"}
    assert set(preview.event_time_sensitivity["component"]) == set(VALUE_COLUMNS)


def test_bounded_primary_component_is_excluded_without_imputation() -> None:
    measurements = _measurements(("candidate-a", "design-a", "experiment-a", 2.0))
    primary = measurements["reduction_id"].eq("primary")
    measurements.loc[primary, "r01_bound_kind"] = "lower"
    measurements.loc[primary, "r01_has_instrument_overflow"] = True

    preview = aggregation.aggregate_response_window_observations(
        measurements,
        _draws(measurements, offsets=(0.0,)),
        policy=_policy(bootstrap_samples=100),
        repeat_decisions=pd.DataFrame(columns=aggregation.DECISION_COLUMNS),
    )

    assert preview.observations.empty
    assert preview.blockers == ()
    contribution = preview.contributions.iloc[0]
    assert contribution["selected_as_label_source"]
    assert not contribution["included_in_label"]
    assert contribution["label_exclusion_reason"] == "nonexact_primary_component"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="r00_bound_kind"), "missing censor provenance"),
        (
            lambda frame: frame.assign(r00_has_policy_clipping="False"),
            "r00_has_policy_clipping.*must contain booleans",
        ),
        (lambda frame: frame.assign(r00_bound_kind="lower"), "r00.*disagrees with its bound kind"),
        (lambda frame: frame.assign(r00_bound_kind="estimated"), "unsupported values"),
    ],
)
def test_measurement_censor_provenance_fails_closed(mutate, message: str) -> None:
    measurements = mutate(_measurements(("candidate-a", "design-a", "experiment-a", 2.0)))

    with pytest.raises(aggregation.ResponseWindowAggregationError, match=message):
        aggregation.aggregate_response_window_observations(
            measurements,
            _draws(_measurements(("candidate-a", "design-a", "experiment-a", 2.0)), offsets=(0.0,)),
            policy=_policy(bootstrap_samples=100),
            repeat_decisions=pd.DataFrame(columns=aggregation.DECISION_COLUMNS),
        )


def _policy(
    *,
    bootstrap_samples: int,
    random_seed: int = 7,
) -> aggregation.ResponseWindowAggregationPolicy:
    return aggregation.ResponseWindowAggregationPolicy(
        policy_id="explicit_label_source_v1",
        primary_reduction_id="primary",
        bootstrap_samples=bootstrap_samples,
        confidence_level=0.90,
        random_seed=random_seed,
        minimum_reader_draws_per_experiment=1,
    )


def _decisions(*rows: tuple[str, str, tuple[str, ...], str, str | None]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "candidate_id": candidate_id,
                "reader_design_ids": [design_id],
                "reader_experiment_ids": list(experiment_ids),
                "label_source_reader_experiment_id": selected_experiment_id,
                "status": status,
                "classification": "unresolved" if status == "review_required" else "source_agreement_accepted",
                "evidence_artifact": None if status == "review_required" else "repeat-review.json",
                "evidence_sha256": None if status == "review_required" else "a" * 64,
                "adjudicated_by": None if status == "review_required" else "study-reviewer",
                "adjudicated_at": None if status == "review_required" else "2026-07-15T12:00:00+00:00",
                "reason": ("label_source_review_pending" if status == "review_required" else "label_source_selected"),
            }
            for candidate_id, design_id, experiment_ids, status, selected_experiment_id in rows
        ]
    )


def _measurements(*rows: tuple[str, str, str, float]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for candidate_id, design_id, experiment_id, value in rows:
        for reduction_id, delta in (("primary", 0.0), ("event_sensitivity", 0.5)):
            records.append(
                {
                    "candidate_id": candidate_id,
                    "design_id": design_id,
                    "reader_experiment_id": experiment_id,
                    "reduction_id": reduction_id,
                    "reduction_role": "primary" if reduction_id == "primary" else "sensitivity",
                    **{column: value + delta for column in VALUE_COLUMNS},
                    **{f"{column}_event_half_range": 0.1 for column in VALUE_COLUMNS},
                    **_exact_censor_provenance(),
                }
            )
    return pd.DataFrame.from_records(records)


def _exact_censor_provenance() -> dict[str, object]:
    return {
        f"{component}_{suffix}": False if suffix != "bound_kind" else "exact"
        for component in VALUE_COLUMNS
        for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
    }


def _draws(measurements: pd.DataFrame, *, offsets: tuple[float, ...]) -> pd.DataFrame:
    primary = measurements.loc[measurements["reduction_id"].eq("primary")]
    records: list[dict[str, object]] = []
    for row in primary.itertuples(index=False):
        for draw_index, offset in enumerate(offsets):
            records.append(
                {
                    "candidate_id": row.candidate_id,
                    "design_id": row.design_id,
                    "reader_experiment_id": row.reader_experiment_id,
                    "reduction_id": "primary",
                    "draw_index": draw_index,
                    **{column: float(getattr(row, column)) + offset for column in VALUE_COLUMNS},
                }
            )
    return pd.DataFrame.from_records(records)


def _joint_draw_fixture(measurements: pd.DataFrame) -> pd.DataFrame:
    primary = measurements.loc[measurements["reduction_id"].eq("primary")]
    records: list[dict[str, object]] = []
    for row in primary.itertuples(index=False):
        for draw_index, offset in enumerate((-1.0, 1.0)):
            values = {column: float(getattr(row, column)) for column in VALUE_COLUMNS}
            values["r00"] += offset
            values["r10"] += offset * 2.0
            values["b00"] += offset
            values["b10"] += offset * 2.0
            records.append(
                {
                    "candidate_id": row.candidate_id,
                    "design_id": row.design_id,
                    "reader_experiment_id": row.reader_experiment_id,
                    "reduction_id": "primary",
                    "draw_index": draw_index,
                    **values,
                }
            )
    return pd.DataFrame.from_records(records)
