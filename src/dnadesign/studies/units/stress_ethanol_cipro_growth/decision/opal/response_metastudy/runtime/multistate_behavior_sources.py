"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_sources.py

Historical Reader and immutable label-policy sources for frozen behavior replay.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    load_promoter_candidate_bindings,
    verify_promoter_candidate_bindings,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.historical import (
    HistoricalReaderResponseBundleV5,
    load_historical_reader_response_bundle_v5,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.sources import (
    ResolvedReaderCandidateEvidence,
    resolve_reader_candidate_evidence,
)

from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .historical import HistoricalObservationPolicyV2, load_historical_observation_policy_v2
from .multistate_behavior_reference import (
    ReferenceSignalIdentityReceipt,
    verify_reference_relative_descriptive_resampling_identity,
)
from .publication import sha256_file


@dataclass(frozen=True)
class VerifiedBehaviorSources:
    reader: HistoricalReaderResponseBundleV5
    resolved: ResolvedReaderCandidateEvidence
    prior_observation_policy: HistoricalObservationPolicyV2
    reference_identity: ReferenceSignalIdentityReceipt
    reader_manifest_sha256: str
    candidate_bindings_manifest_sha256: str


def load_verified_behavior_sources(
    *,
    reader_bundle_root: Path,
    reader_request_path: Path,
    candidate_bindings_root: Path,
    prior_observation_policy_path: Path,
    protocol: MultistateBehaviorShadowProtocol,
) -> VerifiedBehaviorSources:
    """Bind corrected bootstrap evidence to independently verifiable prior label policy."""

    source_protocol = protocol.source_equivalence
    policy = load_historical_observation_policy_v2(
        prior_observation_policy_path,
        expected_sha256=source_protocol.prior_observation_policy_sha256,
        expected_reader_bundle_sha256=source_protocol.prior_observation_reader_bundle_sha256,
        expected_candidate_bindings_sha256=source_protocol.prior_candidate_bindings_manifest_sha256,
        expected_approval_sha256=source_protocol.prior_observation_approval_sha256,
        expected_primary_reduction_id=protocol.primary_reduction_id,
    )
    reader = load_historical_reader_response_bundle_v5(
        reader_bundle_root,
        expected_request_path=reader_request_path,
    )
    reader_sha = sha256_file(reader.manifest_path)
    if reader_sha != source_protocol.current_reader_bundle_sha256:
        raise ValueError("Reader bundle does not match the corrected source-equivalence version.")
    binding_root = Path(candidate_bindings_root).resolve()
    binding_verification = verify_promoter_candidate_bindings(binding_root, allowed_root=binding_root)
    binding_sha = sha256_file(binding_verification.manifest_json)
    if binding_sha != source_protocol.prior_candidate_bindings_manifest_sha256:
        raise ValueError("corrected Reader evaluation changed the approved candidate-binding universe.")
    bindings = load_promoter_candidate_bindings(binding_root, allowed_root=binding_root)
    resolved = resolve_reader_candidate_evidence(
        reader,
        binding_rows=bindings,
        unbound_reader_designs=policy.unbound_reader_designs,
    )
    if "is_reference" not in resolved.measurements.columns:
        raise ValueError("resolved Reader candidate evidence lacks the reference-exclusion field.")
    if resolved.measurements["is_reference"].astype(bool).any():
        raise ValueError("pDual-10 reference rows must not enter the candidate normalization cohort.")
    reference = verify_reference_relative_descriptive_resampling_identity(
        reader.designs,
        reader.descriptive_resampling_draws,
        primary_reduction_id=protocol.primary_reduction_id,
        state_ids=protocol.state_ids,
    )
    return VerifiedBehaviorSources(
        reader=reader,
        resolved=resolved,
        prior_observation_policy=policy,
        reference_identity=reference,
        reader_manifest_sha256=reader_sha,
        candidate_bindings_manifest_sha256=binding_sha,
    )


__all__ = ["VerifiedBehaviorSources", "load_verified_behavior_sources"]
