"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/evidence/test_audits.py

Tests profile audit closure, comparability, and evidence eligibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    EndpointReduction,
    TimeWindowReduction,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    EvidenceReadiness,
    MetastudyContractError,
    ProfileEvidence,
    evaluate_sensitivity,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    build_profile_audit_artifact as build_synthetic_profile_audit_artifact,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    evaluate_metastudy as evaluate_metastudy_with_attempts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    sensitivity_coverage as sensitivity_coverage_contracts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    _build_derivation_closed_profile_audit as build_profile_audit_artifact,
)

from .._builders import (
    HIGH_ANCHOR,
    KINETIC_IDS,
    LOW_ANCHOR,
    _attempts,
    _digest,
    _evidence,
    _profile,
    _ready,
    evaluate_metastudy,
)
from ._builders import (
    _complete_sensitivity_evidence,
    _sensitivity_coverages,
)


@pytest.mark.parametrize(
    ("audit_changes", "blocker"),
    [
        ({"required_observation_count": 0}, "required_observation_count_zero"),
        ({"overflow_observation_count": 1}, "observation_overflow_detected"),
        ({"clipped_observation_count": 1}, "observation_clipping_detected"),
    ],
)
def test_observation_quality_audit_blocks_zero_overflow_and_clipping(audit_changes, blocker: str) -> None:
    evidence = list(_evidence())
    audit = evidence[0].audit
    evidence[0] = replace(
        evidence[0],
        audit=build_profile_audit_artifact(
            evidence[0].profile,
            method_id=audit.method_id,
            within_acquisition_observation_range=audit.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=audit.reference_within_acquisition_observation_range,
            required_observation_count=audit_changes.get(
                "required_observation_count", audit.required_observation_count
            ),
            overflow_observation_count=audit_changes.get(
                "overflow_observation_count", audit.overflow_observation_count
            ),
            clipped_observation_count=audit_changes.get("clipped_observation_count", audit.clipped_observation_count),
        ),
    )

    decision = evaluate_metastudy(evidence, readiness=_ready())

    assert decision.status == "blocked"
    assert blocker in decision.evaluations[0].blockers


def test_profile_audit_rejects_mutation_and_cross_profile_rebinding() -> None:
    evidence = _evidence()
    with pytest.raises(MetastudyContractError, match="artifact digest mismatch"):
        replace(
            evidence[0],
            audit=replace(evidence[0].audit, within_acquisition_observation_range=9.0),
        )
    with pytest.raises(MetastudyContractError, match="source identity digest mismatch"):
        replace(evidence[1], audit=evidence[0].audit)


def test_public_audit_builder_cannot_claim_canonical_derivation() -> None:
    profile = _evidence()[0].profile

    with pytest.raises(ValueError, match="canonical audits are derived only"):
        build_synthetic_profile_audit_artifact(
            profile,
            method_id="canonical_profile_observation_audit_v1",
            within_acquisition_observation_range=0.1,
            reference_within_acquisition_observation_range=0.2,
            required_observation_count=1,
            overflow_observation_count=0,
            clipped_observation_count=0,
        )

    synthetic = build_synthetic_profile_audit_artifact(
        profile,
        method_id="synthetic_profile_audit_v1",
        within_acquisition_observation_range=0.1,
        reference_within_acquisition_observation_range=0.2,
        required_observation_count=1,
        overflow_observation_count=0,
        clipped_observation_count=0,
    )
    evidence = list(_evidence())
    evidence[0] = ProfileEvidence(profile=profile, audit=synthetic)
    with pytest.raises(MetastudyContractError, match="derivation-closed canonical"):
        evaluate_metastudy(evidence, readiness=_ready())


def test_full_profile_digest_prevents_rebinding_after_profile_mutation() -> None:
    evidence = _evidence()[0]
    changed_profile = replace(evidence.profile, profile_id="forged-profile-id")

    with pytest.raises(MetastudyContractError, match="full profile digest mismatch"):
        ProfileEvidence(profile=changed_profile, audit=evidence.audit)


def test_synthetic_readiness_cannot_enter_evaluation() -> None:
    readiness = EvidenceReadiness(
        selected_experiment_count=8,
        ready_experiment_count=8,
        ready_experiment_ids=KINETIC_IDS,
        blocked_experiment_ids=(),
        receipt_digest=_digest("f"),
    )

    with pytest.raises(MetastudyContractError, match="readiness_from_receipt"):
        evaluate_metastudy(_evidence(), readiness=readiness)


def test_cross_window_roster_drift_fails_closed() -> None:
    evidence = list(_evidence())
    evidence.pop()

    with pytest.raises(MetastudyContractError, match="candidate coordinate closure differs"):
        evaluate_metastudy(evidence, readiness=_ready())


def test_cross_window_reader_provenance_drift_fails_closed() -> None:
    evidence = list(_evidence())
    changed_profile = _profile(
        experiment_index=8,
        subject_id=HIGH_ANCHOR,
        window=DEFAULT_PROTOCOL.candidate_windows_h[-1],
        separation=36.0,
        response=1.008,
        revision_digest=_digest("f"),
    )
    prior_audit = evidence[-1].audit
    evidence[-1] = ProfileEvidence(
        profile=changed_profile,
        audit=build_profile_audit_artifact(
            changed_profile,
            method_id=prior_audit.method_id,
            within_acquisition_observation_range=prior_audit.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=(prior_audit.reference_within_acquisition_observation_range),
            required_observation_count=prior_audit.required_observation_count,
            overflow_observation_count=prior_audit.overflow_observation_count,
            clipped_observation_count=prior_audit.clipped_observation_count,
        ),
    )

    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        evaluate_metastudy(evidence, readiness=_ready())


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("reader_protocol_id", "plate_reader/another_protocol"),
        ("reader_record_kind", "another_kind"),
        ("reader_record_path", "artifacts/another.parquet"),
    ],
)
def test_primary_profile_requires_complete_attempt_reader_identity(
    field_name: str,
    changed_value: str,
) -> None:
    evidence = _evidence()
    attempts = list(_attempts(evidence))
    identity = attempts[0].reader_record_identity
    assert identity is not None
    attempts[0] = replace(
        attempts[0],
        reader_record_identity=replace(identity, **{field_name: changed_value}),
    )

    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        evaluate_metastudy_with_attempts(evidence, readiness=_ready(), attempts=attempts)


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("reader_protocol_id", "plate_reader/another_protocol"),
        ("reader_record_kind", "another_kind"),
        ("reader_record_path", "artifacts/another.parquet"),
    ],
)
def test_sensitivity_profile_requires_complete_coverage_reader_identity(
    field_name: str,
    changed_value: str,
) -> None:
    primary = _evidence()
    attempts = _attempts(primary)
    sensitivity = _complete_sensitivity_evidence(primary)
    coverage = _sensitivity_coverages(sensitivity, attempts)[0]
    changed_coverage = replace(
        coverage,
        reader_record_identity=replace(
            coverage.reader_record_identity,
            **{field_name: changed_value},
        ),
    )
    experiment_evidence = tuple(
        row for row in sensitivity if row.profile.provenance.reader_experiment_id == coverage.experiment_id
    )

    with pytest.raises(MetastudyContractError, match="sensitivity profile provenance differs"):
        sensitivity_coverage_contracts.validate_sensitivity_coverage(
            changed_coverage,
            evidence=experiment_evidence,
        )


def test_sensitivity_evidence_is_typed_and_never_selectable() -> None:
    primary = _profile(
        experiment_index=1,
        subject_id=LOW_ANCHOR,
        window=(4.0, 8.0),
        separation=40.0,
        response=0.5,
        doses=(5.0, 50.0, 500.0),
    )
    endpoint = replace(primary, reduction=EndpointReduction(recorded_time_h=8.0))
    centered = replace(
        primary,
        profile_id="centered-sensitivity-profile",
        reduction=TimeWindowReduction(
            recorded_start_time_h=8.0,
            recorded_end_time_h=10.0,
            summary_statistic="median",
            ratio_reduction_order="ratio_then_reduce",
        ),
    )
    template = _evidence(doses=(5.0, 50.0, 500.0))[0]
    audit = template.audit

    def bind(profile):
        return ProfileEvidence(
            profile=profile,
            audit=build_profile_audit_artifact(
                profile,
                method_id=audit.method_id,
                within_acquisition_observation_range=audit.within_acquisition_observation_range,
                reference_within_acquisition_observation_range=audit.reference_within_acquisition_observation_range,
                required_observation_count=audit.required_observation_count,
                overflow_observation_count=audit.overflow_observation_count,
                clipped_observation_count=audit.clipped_observation_count,
            ),
        )

    results = evaluate_sensitivity(
        (
            bind(endpoint),
            bind(centered),
        )
    )

    assert {row.kind for row in results} == {"dose", "endpoint", "centered_window"}
    assert all(row.selectable is False for row in results)


def test_mismatched_profile_comparability_fails_closed() -> None:
    evidence = list(_evidence())
    changed_profile = _profile(
        experiment_index=1,
        subject_id=LOW_ANCHOR,
        window=(4.0, 8.0),
        separation=40.0,
        response=0.5,
        observation_policy_id="rt_lnrna_observation_policy_v2",
        doses=(500.0,),
    )
    prior_audit = evidence[0].audit
    evidence[0] = ProfileEvidence(
        profile=changed_profile,
        audit=build_profile_audit_artifact(
            changed_profile,
            method_id=prior_audit.method_id,
            within_acquisition_observation_range=prior_audit.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=(prior_audit.reference_within_acquisition_observation_range),
            required_observation_count=prior_audit.required_observation_count,
            overflow_observation_count=prior_audit.overflow_observation_count,
            clipped_observation_count=prior_audit.clipped_observation_count,
        ),
    )

    with pytest.raises(MetastudyContractError, match="observation policy|comparability"):
        evaluate_metastudy(
            evidence,
            readiness=_ready(),
        )


def test_endpoint_profiles_cannot_enter_primary_selection() -> None:
    evidence = list(_evidence())
    endpoint = replace(evidence[0].profile, reduction=replace(evidence[0].profile.reduction))
    object.__setattr__(endpoint, "reduction", EndpointReduction(recorded_time_h=8.0))
    prior = evidence[0].audit
    evidence[0] = ProfileEvidence(
        profile=endpoint,
        audit=build_profile_audit_artifact(
            endpoint,
            within_acquisition_observation_range=prior.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=prior.reference_within_acquisition_observation_range,
            required_observation_count=prior.required_observation_count,
            overflow_observation_count=prior.overflow_observation_count,
            clipped_observation_count=prior.clipped_observation_count,
        ),
    )

    with pytest.raises(MetastudyContractError, match="only time-window profiles"):
        evaluate_metastudy(
            evidence,
            readiness=_ready(),
        )
