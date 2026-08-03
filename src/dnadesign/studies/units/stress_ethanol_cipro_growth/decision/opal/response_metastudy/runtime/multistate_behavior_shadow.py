"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_shadow.py

Verified source loading for the stress-study multistate behavior shadow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...source_evidence import response_window_round0_source_evidence_root
from ..core.contracts import MetastudyPaths
from ..evaluation import (
    VerifiedBehaviorCohortReceipt,
    behavior_cohort_unit_ids_sha256,
    behavior_normalization_source_rows_sha256,
    build_multistate_behavior_normalization_record,
    derive_multistate_behavior_normalization,
    load_multistate_behavior_protocol,
)
from ..evaluation.multistate_behavior_comparison import HardBehaviorComparison
from ..evaluation.multistate_behavior_normalization import MultistateBehaviorNormalizationEvidence
from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from ..evaluation.multistate_behavior_shadow import MultistateBehaviorShadowEvidence
from .calibration_preview import build_calibration_cohort
from .historical import load_historical_source_files
from .loading import assert_campaign_response_reduction, load_stress_campaign_contract
from .multistate_behavior_censor import build_behavior_censor_exclusions
from .multistate_behavior_completion import MultistateBehaviorCompletionEvidence
from .multistate_behavior_prediction import load_verified_behavior_prediction_run
from .multistate_behavior_reference import ReferenceSignalIdentityReceipt
from .multistate_behavior_shadow_scoring import build_behavior_shadow_scoring_assembly
from .multistate_behavior_sources import load_verified_behavior_sources
from .publication import sha256_file


@dataclass(frozen=True)
class VerifiedMultistateBehaviorShadow:
    normalization: MultistateBehaviorNormalizationEvidence
    normalization_record: dict[str, object]
    evidence: MultistateBehaviorShadowEvidence
    hard_comparison: HardBehaviorComparison
    censor_exclusions: pd.DataFrame
    completion: MultistateBehaviorCompletionEvidence
    reference_identity: ReferenceSignalIdentityReceipt
    source: dict[str, object]


def load_verified_multistate_behavior_shadow(
    *,
    repo_root: Path,
    reader_bundle_root: Path,
    candidate_bindings_root: Path,
    prediction_run_id: str,
) -> VerifiedMultistateBehaviorShadow:
    """Load exhaustive verified Reader evidence and one digest-bound prediction run."""

    root = Path(repo_root).resolve()
    historical_bundle_root = Path(reader_bundle_root).resolve()
    if not prediction_run_id.strip() or prediction_run_id != prediction_run_id.strip():
        raise ValueError("prediction_run_id must be one explicit nonempty run identity.")
    paths = MetastudyPaths(
        repo_root=root,
        reader_root=historical_bundle_root,
        reader_experiment_root=historical_bundle_root,
        out_dir=root / ".unused-behavior-shadow",
        campaign_root=response_window_round0_source_evidence_root(root).resolve(),
    )
    campaign = load_stress_campaign_contract(paths)
    protocol_path = (
        root
        / "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config"
        / "multistate_response_behavior_shadow_v2.yaml"
    )
    protocol = load_multistate_behavior_protocol(protocol_path)
    protocol.assert_target_views(campaign.target_views)
    historical_sources = load_historical_source_files(
        root,
        protocol=protocol.source_equivalence,
    )
    sources = load_verified_behavior_sources(
        reader_bundle_root=historical_bundle_root,
        reader_request_path=historical_sources.reader_request,
        candidate_bindings_root=Path(candidate_bindings_root).resolve(),
        prior_observation_policy_path=historical_sources.observation_policy,
        protocol=protocol,
    )
    primary_reduction_id = sources.prior_observation_policy.aggregation.primary_reduction_id
    assert_campaign_response_reduction(campaign, primary_reduction_id=primary_reduction_id)
    if primary_reduction_id != protocol.primary_reduction_id:
        raise ValueError("verified Reader primary reduction disagrees with the behavior protocol.")
    cohort = build_calibration_cohort(
        sources.resolved.measurements,
        sources.resolved.bootstrap_draws,
        primary_reduction_id=primary_reduction_id,
    )
    source_digests = {
        "reader_bundle_manifest_sha256": sources.reader_manifest_sha256,
        "reader_request_sha256": sha256_file(historical_sources.reader_request),
        "candidate_bindings_manifest_sha256": sources.candidate_bindings_manifest_sha256,
        "observation_policy_sha256": sources.prior_observation_policy.config_sha256,
    }
    receipt = VerifiedBehaviorCohortReceipt(
        cohort_id=protocol.normalization.cohort_id,
        primary_reduction_id=primary_reduction_id,
        unit_count=cohort.unit_count,
        candidate_count=cohort.candidate_count,
        reader_experiment_count=cohort.reader_experiment_count,
        excluded_nonexact_unit_count=cohort.excluded_nonexact_unit_count,
        reader_bundle_manifest_sha256=sources.reader_manifest_sha256,
        candidate_bindings_manifest_sha256=sources.candidate_bindings_manifest_sha256,
        unit_ids_sha256=behavior_cohort_unit_ids_sha256(cohort.labels),
        source_rows_sha256=behavior_normalization_source_rows_sha256(
            cohort.labels,
            cohort.draws,
            protocol=protocol,
        ),
    )
    normalization = derive_multistate_behavior_normalization(
        cohort.labels,
        cohort.draws,
        protocol=protocol,
        target_views=campaign.target_views,
        verified_cohort_receipt=receipt,
    )
    normalization_record = build_multistate_behavior_normalization_record(
        normalization,
        source_artifact_digests=source_digests,
    )
    prediction_run = load_verified_behavior_prediction_run(
        campaign_dir=paths.campaign_root / campaign.slug,
        candidate_records_path=campaign.candidate_records_path,
        prediction_run_id=prediction_run_id,
        state_ids=protocol.state_ids,
        target_masks=protocol.target_masks,
        comparator_calibration_by_view=campaign.rmf_calibration_by_view,
        comparator_objective_name=protocol.comparator_objective_name,
        comparator_channel=protocol.comparator_score_channel,
        comparator_direction=protocol.comparator_direction,
        model_name="random_forest",
        model_params=campaign.model_params,
        raw_top_k=protocol.prediction_raw_top_k,
    )
    scoring = build_behavior_shadow_scoring_assembly(
        cohort=cohort,
        normalization=normalization,
        prediction_run=prediction_run,
        campaign=campaign,
        protocol=protocol,
        candidate_records_path=campaign.candidate_records_path,
        current_measurements=sources.resolved.measurements,
        reader_bundle_manifest_sha256=sources.reader_manifest_sha256,
        source_observation_bundle_root=_source_observation_bundle_root(root, protocol=protocol),
    )
    exclusions = build_behavior_censor_exclusions(
        sources.resolved.measurements,
        primary_reduction_id=primary_reduction_id,
        state_ids=protocol.state_ids,
    )
    excluded_units = exclusions[["candidate_id", "reader_experiment_id"]].drop_duplicates()
    if len(excluded_units) != cohort.excluded_nonexact_unit_count:
        raise ValueError(
            "behavior censor review does not account for every nonexact primary unit: "
            f"receipt={cohort.excluded_nonexact_unit_count}, table={len(excluded_units)}."
        )
    return VerifiedMultistateBehaviorShadow(
        normalization=normalization,
        normalization_record=normalization_record,
        evidence=scoring.evidence,
        hard_comparison=scoring.comparison,
        censor_exclusions=exclusions,
        completion=scoring.completion,
        reference_identity=sources.reference_identity,
        source={
            "prediction": prediction_run.source,
            "reader_bundle_manifest_sha256": _canonical_digest(sources.reader_manifest_sha256),
            "candidate_bindings_manifest_sha256": _canonical_digest(sources.candidate_bindings_manifest_sha256),
            "reader_request_sha256": _canonical_digest(sha256_file(historical_sources.reader_request)),
            "observation_policy_sha256": _canonical_digest(sources.prior_observation_policy.config_sha256),
        },
    )


def _canonical_digest(value: str) -> str:
    digest = str(value).removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("source digest must be a lowercase SHA-256 value.")
    return f"sha256:{digest}"


def _source_observation_bundle_root(
    repo_root: Path,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> Path:
    relative = Path(protocol.source_equivalence.prior_observation_bundle_repo_path)
    resolved = (repo_root / relative).resolve()
    if not resolved.is_relative_to(repo_root) or not resolved.is_dir():
        raise ValueError("protocol-declared source observation bundle is missing or escapes the repository.")
    return resolved


__all__ = [
    "VerifiedMultistateBehaviorShadow",
    "build_behavior_censor_exclusions",
    "load_verified_multistate_behavior_shadow",
]
