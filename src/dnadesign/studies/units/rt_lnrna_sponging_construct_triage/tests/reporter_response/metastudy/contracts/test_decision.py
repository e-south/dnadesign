"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/contracts/test_decision.py

Tests decision, attempt, and serialization contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    EvidenceReadiness,
    MaterializationBlocker,
    MaterializationOmission,
    MetastudyContractError,
    MetastudyDecision,
    decision_from_readiness,
    decision_to_dict,
    publish_metastudy,
    validate_decision_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    evaluate_metastudy as evaluate_metastudy_with_attempts,
)

from .._builders import (
    KINETIC_IDS,
    _attempts,
    _digest,
    _evidence,
    _ready,
    evaluate_metastudy,
)


def test_mutated_payload_is_rejected_before_publication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.publication import (
        service,
    )

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    payload = decision_to_dict(decision)
    payload["selected_reduction"] = {"recorded_start_time_h": 4.0, "recorded_end_time_h": 8.0}
    with pytest.raises(MetastudyContractError, match="blocked decision"):
        validate_decision_payload(payload)

    monkeypatch.setattr(service, "decision_to_dict", lambda _decision: payload)
    destination = tmp_path / "missing-parent" / "must-not-exist"
    with pytest.raises(MetastudyContractError, match="blocked decision"):
        publish_metastudy(decision, destination)
    assert not destination.parent.exists()


def test_selected_decision_cannot_be_reconstructed_from_copied_fields() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())

    with pytest.raises(MetastudyContractError, match="canonical evaluation"):
        MetastudyDecision(
            contract_id=selected.contract_id,
            protocol_id=selected.protocol_id,
            condition_ontology_digest=selected.condition_ontology_digest,
            status=selected.status,
            selection_use=selected.selection_use,
            evidence_grade=selected.evidence_grade,
            selected_reduction=selected.selected_reduction,
            blockers=selected.blockers,
            limitations=selected.limitations,
            policy_digest=selected.policy_digest,
            evidence_digest=selected.evidence_digest,
            readiness=selected.readiness,
            evaluations=selected.evaluations,
            materialization_attempts=selected.materialization_attempts,
        )


def test_selected_decision_serialization_binds_attempts_but_is_not_a_publication() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    payload = decision_to_dict(selected)

    validate_decision_payload(payload)
    assert len(payload["materialization_attempts"]) == 8

    fabricated = dict(payload)
    fabricated["materialization_attempts"] = payload["materialization_attempts"][:-1]
    with pytest.raises(MetastudyContractError, match="canonical materialization-attempt order"):
        validate_decision_payload(fabricated)


@pytest.mark.parametrize(
    ("limitations", "message"),
    [
        (["duplicate", "duplicate"], "must not contain duplicates"),
        ([""], "non-empty trimmed strings"),
        ([1], "non-empty trimmed strings"),
    ],
)
def test_selected_decision_payload_rejects_noncanonical_limitations(
    limitations: list[object],
    message: str,
) -> None:
    payload = decision_to_dict(evaluate_metastudy(_evidence(), readiness=_ready()))
    payload["limitations"] = limitations

    with pytest.raises(MetastudyContractError, match=message):
        validate_decision_payload(payload)


def test_decision_policy_identity_binds_the_actual_condition_ontology() -> None:
    readiness = EvidenceReadiness._from_validated_receipt(
        selected_experiment_count=8,
        ready_experiment_count=0,
        ready_experiment_ids=(),
        blocked_experiment_ids=KINETIC_IDS,
        receipt_digest=_digest("8"),
    )
    protocol = replace(DEFAULT_PROTOCOL, condition_ontology_digest=_digest("6"))
    payload = decision_to_dict(decision_from_readiness(readiness, protocol=protocol))

    assert payload["condition_ontology_digest"] == protocol.condition_ontology_digest
    validate_decision_payload(payload)
    payload["condition_ontology_digest"] = DEFAULT_PROTOCOL.condition_ontology_digest
    with pytest.raises(MetastudyContractError, match="policy_digest"):
        validate_decision_payload(payload)


def test_materialization_attempt_rejects_noncanonical_or_duplicate_profile_digests() -> None:
    attempt = _attempts(_evidence())[0]

    with pytest.raises(MetastudyContractError, match="canonical digest order"):
        replace(attempt, candidate_profile_digests=tuple(reversed(attempt.candidate_profile_digests)))

    duplicate = (attempt.candidate_profile_digests[0],) * attempt.candidate_profile_count
    with pytest.raises(MetastudyContractError, match="must be unique"):
        replace(attempt, candidate_profile_digests=duplicate)


def test_omission_only_blocked_attempt_requires_complete_coordinate_closure() -> None:
    attempt = _attempts(_evidence())[0]
    incomplete = (
        MaterializationOmission(
            code="condition_or_channel_observations_incomplete",
            subject_id=attempt.expected_subject_ids[0],
            reduction_id="window-4-8h",
        ),
    )
    blocked_fields = {
        "status": "blocked",
        "candidate_profile_count": 0,
        "candidate_profile_digests": (),
        "candidate_omissions": incomplete,
    }

    with pytest.raises(MetastudyContractError, match="complete expected coordinate closure"):
        replace(attempt, blockers=(), **blocked_fields)

    complete_omissions = tuple(
        sorted(
            (
                MaterializationOmission(
                    code="condition_or_channel_observations_incomplete",
                    subject_id=subject_id,
                    reduction_id=f"window-{start:g}-{end:g}h",
                )
                for subject_id in attempt.expected_subject_ids
                for start, end in DEFAULT_PROTOCOL.candidate_windows_h
            ),
            key=lambda row: (row.subject_id, row.reduction_id, row.code),
        )
    )
    closed = replace(
        attempt,
        blockers=(),
        **{**blocked_fields, "candidate_omissions": complete_omissions},
    )
    assert closed.status == "blocked"

    fatal = replace(
        attempt,
        blockers=(MaterializationBlocker("reader_artifact_unreadable"),),
        **blocked_fields,
    )
    assert fatal.status == "blocked"


@pytest.mark.parametrize(
    "malformed_reduction",
    ([], [6.0], [5.0, 9.0], ["6", "10"], [6.0, 10.0, 12.0]),
)
def test_selected_decision_rejects_malformed_reduction_with_contract_error(
    malformed_reduction: list[object],
) -> None:
    payload = decision_to_dict(evaluate_metastudy(_evidence(), readiness=_ready()))
    payload["selected_reduction"] = malformed_reduction

    with pytest.raises(MetastudyContractError, match="declared candidate window"):
        validate_decision_payload(payload)


def test_evaluation_rejects_noncanonical_materialization_attempt_order() -> None:
    evidence = _evidence()

    with pytest.raises(MetastudyContractError, match="canonical selected-experiment order"):
        evaluate_metastudy_with_attempts(
            evidence,
            readiness=_ready(),
            attempts=tuple(reversed(_attempts(evidence))),
        )


def test_decision_payload_rejects_noncanonical_attempt_and_evaluation_order() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    payload = decision_to_dict(selected)
    payload["materialization_attempts"] = tuple(reversed(payload["materialization_attempts"]))
    with pytest.raises(MetastudyContractError, match="canonical materialization-attempt order"):
        validate_decision_payload(payload)

    payload = decision_to_dict(selected)
    payload["evaluations"] = tuple(reversed(payload["evaluations"]))
    with pytest.raises(MetastudyContractError, match="canonical candidate-window order"):
        validate_decision_payload(payload)


def test_seven_of_eight_decision_serializes_the_unavailable_reader_attempt() -> None:
    blocked_id = KINETIC_IDS[-1]
    evidence = tuple(row for row in _evidence() if row.profile.provenance.reader_experiment_id != blocked_id)
    attempts = list(_attempts(evidence))
    attempts[-1] = replace(
        attempts[-1],
        reader_record_identity=None,
        blockers=(MaterializationBlocker("reader_records_not_ready"),),
    )
    readiness = EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=8,
        ready_experiment_count=7,
        ready_experiment_ids=KINETIC_IDS[:-1],
        blocked_experiment_ids=(blocked_id,),
        receipt_digest=_digest("7"),
    )

    decision = evaluate_metastudy_with_attempts(
        evidence,
        readiness=readiness,
        attempts=attempts,
    )
    payload = decision_to_dict(decision)

    validate_decision_payload(payload)
    unavailable = next(row for row in payload["materialization_attempts"] if row["experiment_id"] == blocked_id)
    assert decision.status == "selected"
    assert unavailable["reader_record_identity"] is None
    assert unavailable["blockers"] == ({"code": "reader_records_not_ready"},)


def test_decision_rejects_attempt_reader_identity_drift_from_primary_profiles() -> None:
    evidence = _evidence()
    attempts = _attempts(evidence)
    changed_identity = replace(
        attempts[0].reader_record_identity,
        reader_record_content_digest=_digest("9"),
    )
    changed_attempt = replace(attempts[0], reader_record_identity=changed_identity)
    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        evaluate_metastudy_with_attempts(
            evidence,
            readiness=_ready(),
            attempts=(changed_attempt, *attempts[1:]),
        )
