"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/evaluation_baseline_contracts.py

Types and fixed identities for the round-0 MSRB evaluation baseline.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCHEMA_ID = "stress_ethanol_cipro_growth.msrb_evaluation_baseline.v1"
SCHEMA_VERSION = "1"
BASELINE_PATH = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
    "multistate_response_behavior/evaluation_baseline.yaml"
)
CAMPAIGN_CONFIG_PATH = Path("src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml")
CAMPAIGN_SLUG = "secg_msrb_greedy"
PROTOCOL_ID = "secg_msrb_learning_probe_v1"
OBJECTIVE_ID = "multistate_response_behavior_v1"
RUN_ID = "r0-2026-07-19T22:21:41+00:00-01784499701298508000-24e5927eb1ce4d0daf013dc0c352c584"

ROOT_FIELDS = {
    "schema_id",
    "schema_version",
    "baseline_id",
    "study_id",
    "campaign",
    "alias_registry",
    "artifacts",
    "allocations",
    "comparison_set",
    "evaluation",
    "claim_limits",
}
VIEW_PRIORITY = ("ethanol", "ciprofloxacin", "and")
EXPECTED_QUOTAS = {view_id: 6 for view_id in VIEW_PRIORITY}
EXPECTED_ENDPOINTS = (
    (
        "raw_y8_prediction_fidelity",
        "mae_and_rmse_by_coordinate_plus_pooled",
        "candidate",
        "frozen_prediction_vs_first_subsequent_exact_observation",
    ),
    (
        "within_view_rank_preservation",
        "spearman_tie_aware_average_ranks_with_undefined_constant_case",
        "selection_view",
        "all_18_allocated_candidates_rescored_in_each_view",
    ),
    (
        "within_batch_view_specificity",
        "median_observed_score_allocated_six_minus_other_twelve",
        "selection_view",
        "same_18_candidate_batch",
    ),
    (
        "prior_observed_corpus_context",
        "allocated_six_best_and_median_percentile_against_all_historical_six_subsets",
        "selection_view",
        "allocated_six_vs_exhaustive_subsets_of_exact_27_prior_observed_labels",
    ),
)
EXPECTED_EVALUATION_CONVENTIONS = {
    "missing_or_nonfinite": "error",
    "score_direction": "higher_is_better",
    "raw_y8": {
        "coordinate_order": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"],
        "candidate_weighting": "equal",
        "coordinate_weighting": "equal",
        "coordinate_mae": "mean_absolute_error_across_candidates",
        "coordinate_rmse": "root_mean_squared_error_across_candidates",
        "pooled_mae": "mean_absolute_error_across_all_candidates_and_coordinates",
        "pooled_rmse": "root_mean_squared_error_across_all_candidates_and_coordinates",
    },
    "spearman": {
        "rank_ties": "average_rank",
        "tie_equality": "exact_numeric_equality",
        "correlation": "pearson_correlation_of_average_ranks",
        "constant_input": "undefined_null",
    },
    "median": {
        "even_sample": "arithmetic_midpoint_of_two_central_sorted_values",
    },
    "exhaustive_subset_percentile": {
        "enumeration": "each_unordered_subset_exactly_once",
        "ties": "midrank",
        "tie_equality": "exact_numeric_equality",
        "formula": "100 * (count(reference < observed) + 0.5 * count(reference == observed)) / subset_count",
        "range": [0, 100],
    },
}
EXPECTED_CANDIDATE_OUTPUTS = (
    "study_alias",
    "candidate_id",
    "allocated_view",
    "allocation_slot",
    "predicted_y8",
    "observed_y8",
    "predicted_and_observed_family_scores",
    "predicted_and_observed_behavior_score",
    "predicted_and_observed_hard_bottleneck",
    "predicted_and_observed_all_reference_directions_met",
    "predicted_and_observed_within_view_rank",
)
COMPARISON_STATEMENT = (
    "Every six-candidate subset of the 27 prior observed labels defines a deterministic historical reference "
    "distribution for best and median observed MSRB by view. This is not a randomized or physically measured "
    "control cohort."
)
COMPARISON_ROLE = "historical_model_free_six_candidate_baseline"
COMPARISON_METHOD = "exhaustive_unordered_subsets_without_replacement"
COMPARISON_SUBSET_SIZE = 6
COMPARISON_SUBSET_COUNT = 296_010
CLAIM_LIMIT_STATEMENT = (
    "This baseline does not support acquisition-efficacy or hill-climb claims and does not authorize synthesis."
)


class MsrbEvaluationBaselineError(ValueError):
    """Raised when the persisted baseline or one of its sources drifts."""


@dataclass(frozen=True)
class FrozenArtifact:
    path: Path
    sha256: str
    row_count: int


@dataclass(frozen=True)
class FrozenFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class SelectionReplayEvidence:
    score_count: int
    max_abs_score_difference: float
    allocated_count: int


@dataclass(frozen=True)
class FrozenAllocation:
    study_alias: str
    candidate_id: str
    sequence_sha256: str
    selection_view: str
    allocation_slot: int


@dataclass(frozen=True)
class ParsedBaseline:
    campaign_config: FrozenFile
    selection_allocation_api_version: str
    prediction_ledger: FrozenArtifact
    selection_batch: FrozenArtifact
    labels_used: FrozenArtifact
    allocations: tuple[FrozenAllocation, ...]
    alias_registry_path: str
    comparison_candidate_ids: tuple[str, ...]
    comparison_method: str
    comparison_subset_size: int
    comparison_subset_count: int
    endpoint_ids: tuple[str, ...]
    acquisition_efficacy_claim: str
    hill_climb_claim: str
    synthesis_authorization: str
    claim_limit_statement: str


@dataclass(frozen=True)
class MsrbEvaluationBaseline:
    schema_id: str
    baseline_id: str
    campaign_slug: str
    run_id: str
    round_index: int
    campaign_config: FrozenFile
    selection_allocation_api_version: str
    selection_replay: SelectionReplayEvidence
    prediction_ledger: FrozenArtifact
    selection_batch: FrozenArtifact
    labels_used: FrozenArtifact
    allocations: tuple[FrozenAllocation, ...]
    comparison_role: str
    comparison_candidate_ids: tuple[str, ...]
    comparison_method: str
    comparison_subset_size: int
    comparison_subset_count: int
    physical_random_control: bool
    comparison_statement: str
    endpoint_ids: tuple[str, ...]
    acquisition_efficacy_claim: str
    hill_climb_claim: str
    synthesis_authorization: str
    claim_limit_statement: str
    source_path: Path
    source_sha256: str
