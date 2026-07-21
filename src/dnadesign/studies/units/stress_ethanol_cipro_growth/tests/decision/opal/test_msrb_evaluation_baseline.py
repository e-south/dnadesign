"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/test_msrb_evaluation_baseline.py

Contract tests for the frozen round-0 MSRB evaluation baseline.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.multistate_response_behavior import (
    evaluation_baseline,
    evaluation_baseline_artifacts,
    prospective_evaluation,
)

BASELINE_PATH = evaluation_baseline.BASELINE_PATH
CAMPAIGN_SLUG = evaluation_baseline.CAMPAIGN_SLUG
RUN_ID = evaluation_baseline.RUN_ID
SCHEMA_ID = evaluation_baseline.SCHEMA_ID
MsrbEvaluationBaselineError = evaluation_baseline.MsrbEvaluationBaselineError
load_msrb_evaluation_baseline = evaluation_baseline.load_msrb_evaluation_baseline

REPO_ROOT = Path(__file__).resolve().parents[8]
PROTOCOL_PATH = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/protocol.yaml"
)


def _payload() -> dict[str, Any]:
    raw = yaml.safe_load((REPO_ROOT / BASELINE_PATH).read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "evaluation_baseline.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_round0_baseline_verifies_exact_frozen_sources_and_allocations() -> None:
    baseline = load_msrb_evaluation_baseline(REPO_ROOT)

    assert baseline.schema_id == SCHEMA_ID
    assert baseline.campaign_slug == CAMPAIGN_SLUG
    assert baseline.run_id == RUN_ID
    assert baseline.round_index == 0
    assert baseline.campaign_config.path.as_posix() == (
        "src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml"
    )
    assert baseline.selection_allocation_api_version == "1"
    assert baseline.selection_replay.score_count == 464_355
    assert baseline.selection_replay.max_abs_score_difference == 0.0
    assert baseline.selection_replay.allocated_count == 18
    assert baseline.prediction_ledger.row_count == 154_785
    assert baseline.selection_batch.row_count == 18
    assert baseline.labels_used.row_count == 27
    assert tuple(row.study_alias for row in baseline.allocations) == tuple(
        f"SECG-{ordinal:03d}" for ordinal in range(19, 37)
    )
    assert len({row.candidate_id for row in baseline.allocations}) == 18
    assert len({row.sequence_sha256 for row in baseline.allocations}) == 18
    assert Counter(row.selection_view for row in baseline.allocations) == {
        "ethanol": 6,
        "ciprofloxacin": 6,
        "and": 6,
    }
    assert len(baseline.comparison_candidate_ids) == 27
    assert baseline.comparison_role == "historical_model_free_six_candidate_baseline"
    assert baseline.comparison_method == "exhaustive_unordered_subsets_without_replacement"
    assert baseline.comparison_subset_size == 6
    assert baseline.comparison_subset_count == 296_010
    assert baseline.physical_random_control is False
    assert baseline.comparison_statement == (
        "Every six-candidate subset of the 27 prior observed labels defines a deterministic historical reference "
        "distribution for best and median observed MSRB by view. This is not a randomized or physically measured "
        "control cohort."
    )
    assert baseline.acquisition_efficacy_claim == "not_supported"
    assert baseline.hill_climb_claim == "not_supported"
    assert baseline.synthesis_authorization == "prohibited"
    assert baseline.claim_limit_statement == (
        "This baseline does not support acquisition-efficacy or hill-climb claims and does not authorize synthesis."
    )
    rank_endpoint = next(
        row for row in _payload()["evaluation"]["endpoints"] if row["id"] == "within_view_rank_preservation"
    )
    assert rank_endpoint["method"] == "spearman_tie_aware_average_ranks_with_undefined_constant_case"


def test_protocol_points_to_exact_baseline_without_granting_synthesis() -> None:
    protocol = yaml.safe_load((REPO_ROOT / PROTOCOL_PATH).read_text(encoding="utf-8"))
    baseline_path = REPO_ROOT / BASELINE_PATH

    assert protocol["evidence"]["acquisition_baseline_contract"] == {
        "schema_id": SCHEMA_ID,
        "path": BASELINE_PATH.as_posix(),
        "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    }
    assert protocol["evidence"]["synthesis_authorization"] == "prohibited"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("campaign", "slug", "secg_rmf_greedy", "campaign slug"),
        ("campaign", "run_id", "wrong-run", "run ID"),
        ("campaign", "round_index", 1, "round index"),
    ),
)
def test_baseline_rejects_wrong_campaign_identity(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _payload()
    payload[section][field] = value

    with pytest.raises(MsrbEvaluationBaselineError, match=message):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, payload))


def test_baseline_rejects_artifact_digest_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["artifacts"]["prediction_ledger"]["sha256"] = "0" * 64

    with pytest.raises(MsrbEvaluationBaselineError, match="prediction_ledger SHA-256 mismatch"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, payload))


def test_baseline_rejects_campaign_config_or_allocator_version_drift(tmp_path: Path) -> None:
    digest_payload = _payload()
    digest_payload["campaign"]["config"]["sha256"] = "0" * 64
    with pytest.raises(MsrbEvaluationBaselineError, match="campaign.config SHA-256 mismatch"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, digest_payload))

    version_payload = _payload()
    version_payload["campaign"]["selection_allocation_api_version"] = "2"
    with pytest.raises(MsrbEvaluationBaselineError, match="selection_allocation_api_version"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, version_payload))


@pytest.mark.parametrize(
    ("field", "source_index", "message"),
    (
        ("candidate_id", 1, "candidate IDs must be unique"),
        ("sequence_sha256", 1, "sequence digests must be unique"),
        ("study_alias", 1, "study aliases must be unique"),
    ),
)
def test_baseline_rejects_duplicate_allocation_identity(
    tmp_path: Path,
    field: str,
    source_index: int,
    message: str,
) -> None:
    payload = _payload()
    payload["allocations"][0][field] = payload["allocations"][source_index][field]

    with pytest.raises(MsrbEvaluationBaselineError, match=message):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, payload))


def test_baseline_rejects_wrong_selection_count_and_quota(tmp_path: Path) -> None:
    count_payload = _payload()
    count_payload["artifacts"]["selection_batch"]["row_count"] = 17
    with pytest.raises(MsrbEvaluationBaselineError, match="selection_batch row count mismatch"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, count_payload))

    quota_payload = _payload()
    quota_payload["allocations"][-1]["selection_view"] = "ethanol"
    with pytest.raises(MsrbEvaluationBaselineError, match="allocation quotas"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, quota_payload))


def test_baseline_rejects_unknown_study_alias(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["allocations"][0]["study_alias"] = "SECG-999"

    with pytest.raises(MsrbEvaluationBaselineError, match="unknown study alias"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, payload))


def test_baseline_rejects_labels_comparison_set_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["comparison_set"]["candidate_ids"][0] = "not-a-label"

    with pytest.raises(MsrbEvaluationBaselineError, match="comparison candidate IDs do not match labels_used"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, payload))


def test_baseline_nonredundancy_rejects_prior_candidate_or_sequence() -> None:
    with pytest.raises(MsrbEvaluationBaselineError, match="candidate IDs overlap"):
        evaluation_baseline_artifacts.require_selection_disjoint_from_labels(
            selected_candidate_ids={"selected-a"},
            selected_sequences={"AAAA"},
            observed_candidate_ids={"selected-a"},
            observed_sequences={"CCCC"},
        )
    with pytest.raises(MsrbEvaluationBaselineError, match="sequences overlap"):
        evaluation_baseline_artifacts.require_selection_disjoint_from_labels(
            selected_candidate_ids={"selected-a"},
            selected_sequences={"AAAA"},
            observed_candidate_ids={"observed-a"},
            observed_sequences={"AAAA"},
        )


def test_baseline_rejects_mutable_or_incomplete_historical_generator(tmp_path: Path) -> None:
    method_payload = _payload()
    method_payload["comparison_set"]["generator"]["method"] = "random_sample"
    with pytest.raises(MsrbEvaluationBaselineError, match="comparison_set.generator.method"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, method_payload))

    count_payload = _payload()
    count_payload["comparison_set"]["generator"]["subset_count"] = 10_000
    with pytest.raises(MsrbEvaluationBaselineError, match="subset_count"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, count_payload))


def test_baseline_freezes_exact_prospective_evaluation_conventions(tmp_path: Path) -> None:
    payload = _payload()

    conventions = payload["evaluation"]["conventions"]
    assert conventions["missing_or_nonfinite"] == "error"
    assert conventions["score_direction"] == "higher_is_better"
    assert conventions["raw_y8"] == {
        "coordinate_order": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"],
        "candidate_weighting": "equal",
        "coordinate_weighting": "equal",
        "coordinate_mae": "mean_absolute_error_across_candidates",
        "coordinate_rmse": "root_mean_squared_error_across_candidates",
        "pooled_mae": "mean_absolute_error_across_all_candidates_and_coordinates",
        "pooled_rmse": "root_mean_squared_error_across_all_candidates_and_coordinates",
    }
    assert conventions["spearman"] == {
        "rank_ties": "average_rank",
        "tie_equality": "exact_numeric_equality",
        "correlation": "pearson_correlation_of_average_ranks",
        "constant_input": "undefined_null",
    }
    assert conventions["median"] == {
        "even_sample": "arithmetic_midpoint_of_two_central_sorted_values",
    }
    assert conventions["exhaustive_subset_percentile"] == {
        "enumeration": "each_unordered_subset_exactly_once",
        "ties": "midrank",
        "tie_equality": "exact_numeric_equality",
        "formula": "100 * (count(reference < observed) + 0.5 * count(reference == observed)) / subset_count",
        "range": [0, 100],
    }

    payload["evaluation"]["conventions"]["spearman"]["constant_input"] = "zero"
    with pytest.raises(MsrbEvaluationBaselineError, match="evaluation conventions"):
        load_msrb_evaluation_baseline(REPO_ROOT, baseline_path=_write_payload(tmp_path, payload))


def test_raw_y8_error_summary_uses_equal_candidate_and_coordinate_weighting() -> None:
    predicted = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    observed = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    summary = prospective_evaluation.raw_y8_error_summary(predicted, observed)

    assert summary.coordinate_order == ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")
    assert summary.coordinate_mae == pytest.approx((1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert summary.coordinate_rmse == pytest.approx((5**0.5 / 2**0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert summary.pooled_mae == pytest.approx(3.0 / 16.0)
    assert summary.pooled_rmse == pytest.approx((5.0 / 16.0) ** 0.5)


@pytest.mark.parametrize(
    ("predicted", "observed"),
    (
        ([[0.0] * 8], [[float("nan")] * 8]),
        ([[0.0] * 8], [[float("inf")] * 8]),
        ([[0.0] * 8], [[0.0] * 7]),
        ([], []),
    ),
)
def test_raw_y8_error_summary_fails_closed_on_invalid_values(
    predicted: list[list[float]],
    observed: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        prospective_evaluation.raw_y8_error_summary(predicted, observed)


def test_spearman_uses_average_ranks_for_ties_and_null_for_constant_input() -> None:
    assert prospective_evaluation.spearman_average_rank([1.0, 1.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(
        3.0**0.5 / 2.0
    )
    assert prospective_evaluation.spearman_average_rank([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert prospective_evaluation.spearman_average_rank([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None


def test_spearman_fails_closed_on_missing_nonfinite_or_incompatible_values() -> None:
    with pytest.raises(ValueError):
        prospective_evaluation.spearman_average_rank([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(ValueError):
        prospective_evaluation.spearman_average_rank([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        prospective_evaluation.spearman_average_rank([1.0], [1.0])


def test_midpoint_median_and_exhaustive_subset_midrank_percentile_are_hand_computable() -> None:
    assert prospective_evaluation.midpoint_median([10.0, 2.0, 6.0, 4.0]) == pytest.approx(5.0)
    assert prospective_evaluation.midpoint_median([9.0, 1.0, 5.0]) == pytest.approx(5.0)

    reference = [1.0, 2.0, 2.0, 4.0]
    assert prospective_evaluation.midrank_percentile(2.0, reference) == pytest.approx(50.0)
    assert prospective_evaluation.midrank_percentile(0.0, reference) == pytest.approx(0.0)
    assert prospective_evaluation.midrank_percentile(5.0, reference) == pytest.approx(100.0)


def test_median_and_midrank_percentile_fail_closed_on_missing_or_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        prospective_evaluation.midpoint_median([])
    with pytest.raises(ValueError):
        prospective_evaluation.midpoint_median([1.0, float("nan")])
    with pytest.raises(ValueError):
        prospective_evaluation.midrank_percentile(float("inf"), [1.0, 2.0])
    with pytest.raises(ValueError):
        prospective_evaluation.midrank_percentile(1.0, [])
