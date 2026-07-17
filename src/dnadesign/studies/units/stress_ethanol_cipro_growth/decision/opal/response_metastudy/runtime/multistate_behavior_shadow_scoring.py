"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_shadow_scoring.py

Assemble corrected-RMF and behavior scores from one verified raw-Y matrix.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.contracts import StressCampaignContract
from ..evaluation.multistate_behavior_comparison import HardBehaviorComparison, compare_hard_and_behavior_scores
from ..evaluation.multistate_behavior_normalization import MultistateBehaviorNormalizationEvidence
from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from ..evaluation.multistate_behavior_rmf_replay import (
    bind_current_rmf_calibration,
    build_current_rmf_prediction_scores,
)
from ..evaluation.multistate_behavior_shadow import (
    MultistateBehaviorShadowEvidence,
    build_multistate_behavior_shadow_evidence,
)
from ..evaluation.response_uncertainty import estimate_response_calibration_from_reader_draws
from .calibration_preview import ResponseCalibrationCohort
from .multistate_behavior_completion import (
    MultistateBehaviorCompletionEvidence,
    build_multistate_behavior_completion_evidence,
)
from .multistate_behavior_prediction import VerifiedBehaviorPredictionRun


@dataclass(frozen=True)
class BehaviorShadowScoringAssembly:
    """Scored evidence, objective comparison, and completion-gate analyses."""

    evidence: MultistateBehaviorShadowEvidence
    comparison: HardBehaviorComparison
    completion: MultistateBehaviorCompletionEvidence


def build_behavior_shadow_scoring_assembly(
    *,
    cohort: ResponseCalibrationCohort,
    normalization: MultistateBehaviorNormalizationEvidence,
    prediction_run: VerifiedBehaviorPredictionRun,
    campaign: StressCampaignContract,
    protocol: MultistateBehaviorShadowProtocol,
    candidate_records_path: Path,
    current_measurements: pd.DataFrame,
    reader_bundle_manifest_sha256: str,
    source_observation_bundle_root: Path,
) -> BehaviorShadowScoringAssembly:
    """Score behavior and corrected RMF, then build the prespecified study gate."""

    evidence = build_multistate_behavior_shadow_evidence(
        observed=cohort.labels,
        bootstrap_draws=cohort.draws,
        predictions=prediction_run.predictions,
        protocol=protocol,
        normalization=normalization,
        target_views=campaign.target_views,
    )
    rmf_uncertainty = estimate_response_calibration_from_reader_draws(
        cohort.labels,
        cohort.draws,
        target_views=campaign.target_views,
        scale_quantile=protocol.completion_gate.normalization_primary_quantile,
        expected_bootstrap_samples=normalization.bootstrap_samples,
    )
    replayed_rmf_scores = build_current_rmf_prediction_scores(
        predictions=prediction_run.predictions,
        calibration=rmf_uncertainty.calibration,
        protocol=protocol,
        target_views=campaign.target_views,
    )
    comparison = compare_hard_and_behavior_scores(
        replayed_rmf_scores,
        evidence.prediction_scores,
        top_k=protocol.prediction_raw_top_k,
        hard_score_semantics=(
            f"{protocol.comparator_objective_name}.{protocol.comparator_score_channel}.{protocol.comparator_direction}"
        ),
    )
    replay_calibration = bind_current_rmf_calibration(
        rmf_uncertainty.calibration,
        reader_bundle_manifest_sha256=reader_bundle_manifest_sha256,
        normalization_source_rows_sha256=normalization.source_rows_sha256,
    )
    completion = build_multistate_behavior_completion_evidence(
        normalization=normalization,
        predictions=prediction_run.predictions,
        observed_scores=evidence.observed_scores,
        hard_behavior_detail=comparison.detail,
        candidate_records_path=candidate_records_path,
        campaign_config_path=campaign.config_path,
        current_measurements=current_measurements,
        source_observation_bundle_root=source_observation_bundle_root,
        rmf_uncertainty_rows=rmf_uncertainty.rows,
        rmf_replay_calibration=replay_calibration,
        target_views=campaign.target_views,
        protocol=protocol,
        model_params=campaign.model_params,
        prediction_run_id=str(prediction_run.source["run_id"]),
        prediction_source_sha256=str(prediction_run.source["ledger_sha256"]),
    )
    return BehaviorShadowScoringAssembly(
        evidence=evidence,
        comparison=comparison,
        completion=completion,
    )


__all__ = ["BehaviorShadowScoringAssembly", "build_behavior_shadow_scoring_assembly"]
