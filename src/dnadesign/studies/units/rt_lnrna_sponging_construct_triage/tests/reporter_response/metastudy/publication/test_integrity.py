"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/publication/test_integrity.py

Tests publication provenance, verification, and forgery resistance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    EvidenceReadiness,
    MetastudyContractError,
    MetastudyDecision,
    acquisition_projection_payload,
    build_acquisition_projection,
    decision_evidence_payload,
    decision_from_readiness,
    decision_to_dict,
    evaluate_sensitivity,
    publish_metastudy,
    verify_publication,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    sensitivity_coverage as sensitivity_coverage_contracts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
    canonical_digest,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.evidence_projection import (
    parse_profile_evidence_projection,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.sensitivity import (
    sensitivity_evaluations_to_payload,
)

from .._builders import (
    KINETIC_IDS,
    _digest,
    _evidence,
    _ready,
    evaluate_metastudy,
)
from ..evidence._builders import (
    _complete_sensitivity_evidence,
    _sensitivity_coverages,
)
from ._builders import _publish_selected


def test_selected_publication_is_create_only_and_evidence_bearing(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = tmp_path / "selected"

    _publish_selected(selected, destination)
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        "report.md",
        "evidence.json",
        "acquisition.json",
        "sensitivity.json",
    }
    verify_publication(destination)
    with pytest.raises(FileExistsError):
        _publish_selected(selected, destination)


def test_selected_publication_rejects_missing_or_tampered_evidence(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    with pytest.raises(
        MetastudyContractError,
        match="evidence-bearing publication requires canonical profile evidence",
    ):
        publish_metastudy(selected, tmp_path / "missing")

    destination = _publish_selected(selected, tmp_path / "tampered")
    payload = json.loads((destination / "evidence.json").read_text(encoding="utf-8"))
    payload["profiles"][0]["profile"]["profile_id"] = "tampered"
    (destination / "evidence.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MetastudyContractError, match="evidence file digest mismatch"):
        verify_publication(destination)


def test_publication_rejects_rehashed_acquisition_projection_not_derived_from_profiles(tmp_path: Path) -> None:
    import hashlib

    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = _publish_selected(selected, tmp_path / "acquisition-tamper")
    projection_path = destination / "acquisition.json"
    manifest_path = destination / "manifest.json"
    payload = json.loads(projection_path.read_text(encoding="utf-8"))
    payload["coordinates"][0]["contributions"][0]["normalized_reporter_response"] += 1.0
    payload_without_digest = {key: value for key, value in payload.items() if key != "projection_digest"}
    payload["projection_digest"] = canonical_digest(payload_without_digest)
    projection_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    projection_path.write_bytes(projection_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["acquisition_file_digest"] = "sha256:" + hashlib.sha256(projection_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="differs from bundled profiles"):
        verify_publication(destination)


def test_publication_rejects_tampered_or_reordered_sensitivity_projection(tmp_path: Path) -> None:
    import hashlib

    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = _publish_selected(selected, tmp_path / "sensitivity-tamper")
    sensitivity_path = destination / "sensitivity.json"
    manifest_path = destination / "manifest.json"
    payload = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    payload["evaluations"][0]["evidence_digest"] = _digest("f")
    sensitivity_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    sensitivity_path.write_bytes(sensitivity_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sensitivity_file_digest"] = "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="summaries differ"):
        verify_publication(destination)

    destination = _publish_selected(selected, tmp_path / "sensitivity-reorder")
    sensitivity_path = destination / "sensitivity.json"
    manifest_path = destination / "manifest.json"
    payload = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    payload["profiles"].reverse()
    sensitivity_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    sensitivity_path.write_bytes(sensitivity_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sensitivity_file_digest"] = "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="not canonical"):
        verify_publication(destination)


def test_publication_rejects_self_consistent_sensitivity_chain_with_wrong_reader_revision(
    tmp_path: Path,
) -> None:
    import hashlib

    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = _publish_selected(selected, tmp_path / "revision-drift")
    sensitivity_path = destination / "sensitivity.json"
    evidence_path = destination / "evidence.json"
    report_path = destination / "report.md"
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = manifest["decision"]
    attempt = decision["materialization_attempts"][0]
    experiment_id = attempt["experiment_id"]
    new_revision = attempt["reader_record_identity"]["reader_record_revision"] + 1_000
    attempt["reader_record_identity"]["reader_record_revision"] = new_revision
    attempt["attempt_digest"] = canonical_digest(
        {key: value for key, value in attempt.items() if key != "attempt_digest"}
    )

    payload = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    changed_profile_digests: dict[str, str] = {}
    for profile_row in payload["profiles"]:
        profile = profile_row["profile"]
        if profile["provenance"]["reader_experiment_id"] != experiment_id:
            continue
        old_profile_digest = profile_row["audit"]["profile_digest"]
        profile["provenance"]["reader_record_revision"] = new_revision
        source_identity = {
            **profile["provenance"],
            "observation_policy_identity": profile["observation_policy"]["digest"],
        }
        audit = profile_row["audit"]
        audit["profile_source_digest"] = canonical_digest(source_identity)
        audit["profile_digest"] = canonical_digest(profile)
        audit_without_digest = {key: value for key, value in audit.items() if key != "artifact_digest"}
        audit["artifact_digest"] = canonical_digest(audit_without_digest)
        changed_profile_digests[old_profile_digest] = audit["profile_digest"]
    assert len(changed_profile_digests) == 30

    coverage = next(row for row in payload["coverages"] if row["experiment_id"] == experiment_id)
    coverage["reader_record_identity"]["reader_record_revision"] = new_revision
    coverage["materialization_attempt_digest"] = attempt["attempt_digest"]
    for entry in coverage["entries"]:
        if entry["profile_digest"] in changed_profile_digests:
            entry["profile_digest"] = changed_profile_digests[entry["profile_digest"]]
    coverage["coverage_digest"] = canonical_digest(
        {key: value for key, value in coverage.items() if key != "coverage_digest"}
    )
    projections = tuple(
        parse_profile_evidence_projection(row, index=index) for index, row in enumerate(payload["profiles"])
    )
    payload["evaluations"] = sensitivity_evaluations_to_payload(evaluate_sensitivity(projections))

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    primary_profiles = copy.deepcopy(evidence["profiles"])
    evidence["materialization_attempts"] = copy.deepcopy(decision["materialization_attempts"])
    old_evidence_digest = decision["evidence_digest"]
    decision["evidence_digest"] = canonical_digest(evidence)
    assert evidence["profiles"] == primary_profiles

    sensitivity_bytes = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    evidence_bytes = (json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    sensitivity_path.write_bytes(sensitivity_bytes)
    evidence_path.write_bytes(evidence_bytes)
    report = report_path.read_text(encoding="utf-8")
    assert old_evidence_digest in report
    report = report.replace(old_evidence_digest, decision["evidence_digest"])
    report_path.write_text(report, encoding="utf-8")
    manifest["sensitivity_file_digest"] = "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest()
    manifest["evidence_file_digest"] = "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
    manifest["report_digest"] = "sha256:" + hashlib.sha256(report.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        verify_publication(destination)


def test_selected_decision_cannot_be_reconstructed_from_copied_fields() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())

    with pytest.raises(MetastudyContractError, match="canonical evaluation"):
        MetastudyDecision(
            contract_id=selected.contract_id,
            protocol_id=selected.protocol_id,
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


def test_verify_publication_rejects_report_rewrite_even_with_matching_digest(tmp_path: Path) -> None:
    import hashlib

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = publish_metastudy(decision, tmp_path / "tampered-report")
    rewritten = (destination / "report.md").read_text(encoding="utf-8") + "\nforged\n"
    (destination / "report.md").write_text(rewritten, encoding="utf-8")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    manifest["report_digest"] = "sha256:" + hashlib.sha256(rewritten.encode("utf-8")).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="canonical rendered decision"):
        verify_publication(destination)


def test_verify_publication_rejects_forged_selection_with_recomputed_digests(tmp_path: Path) -> None:
    import hashlib

    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    assert selected.selected_reduction == (6.0, 10.0)
    destination = _publish_selected(selected, tmp_path / "forged-selection", evidence=evidence)

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = manifest["decision"]
    decision["selected_reduction"] = [12.0, 16.0]
    for evaluation in decision["evaluations"]:
        if evaluation["reduction"] == [12.0, 16.0]:
            evaluation["worst_experiment_control_separation"] = 1_000.0
            evaluation["repeated_anchor_drift"] = 0.0
            evaluation["within_acquisition_observation_range"] = 0.0
            evaluation["growth_phase_start"] = 1.0
            evaluation["growth_phase_end"] = 0.5
            evaluation["eligible_experiment_count"] = 8
            evaluation["anchor_ordered_acquisition_count"] = 5
            evaluation["co_measured_anchor_acquisition_count"] = 5
            evaluation["loo_same_or_adjacent_fraction"] = 1.0
            evaluation["eligible"] = True
            evaluation["blockers"] = []
    report = (destination / "report.md").read_text(encoding="utf-8").replace("`6-10 h`", "`12-16 h`")
    (destination / "report.md").write_text(report, encoding="utf-8")
    manifest["report_digest"] = "sha256:" + hashlib.sha256(report.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="canonical evidence evaluation"):
        verify_publication(destination)


def test_selected_source_state_is_compact_and_rejects_phase_ineligible_forgery() -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
        state as operator_state,
    )

    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    decision = json.loads(json.dumps(decision_to_dict(selected)))
    readiness = {
        "schema_id": "rt_lnrna_reporter_response_readiness_snapshot.v1",
        "source_identity": {
            "route_id": "rt_lnrna_reporter_response_metastudy",
            "route_registry_path": ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json",
            "route_registry_digest": _digest("a"),
            "normalized_full_receipt_digest": decision["readiness"]["receipt_digest"],
            "normalization": "omit environment-specific reader_command before canonical JSON hashing",
        },
        "last_verified": "2026-07-30",
        "selected_experiment_count": 8,
        "related_experiment_count": 1,
        "related_experiment_ids": ["20251105_retron_Eco1_RT_variants"],
        "ready_experiment_count": 8,
        "ready_experiment_ids": list(KINETIC_IDS),
        "blocked_experiment_ids": [],
    }
    body = {
        "readiness": readiness,
        "decision": decision,
        "objective_readiness": asdict(DEFAULT_OBJECTIVE_READINESS),
        "sensitivity_evaluations": [],
        "sensitivity_coverage_receipts": [
            sensitivity_coverage_contracts.sensitivity_coverage_receipt_payload(row)
            for row in _sensitivity_coverages(
                _complete_sensitivity_evidence(evidence),
                selected.materialization_attempts,
            )
        ],
        "acquisition_projection": acquisition_projection_payload(
            build_acquisition_projection(
                evidence,
                selected_reduction=selected.selected_reduction,
            )
        ),
    }
    payload = {
        "schema_id": "rt_lnrna_reporter_response_metastudy_state.v6",
        "generation_digest": operator_state.canonical_digest(body),
        **body,
    }
    operator_state.validate_state_payload(payload)

    with_embedded_evidence = {
        **payload,
        "evidence": json.loads(json.dumps(decision_evidence_payload(evidence, decision=selected))),
    }
    with_embedded_evidence["generation_digest"] = operator_state.canonical_digest(
        {
            **body,
            "evidence": with_embedded_evidence["evidence"],
        }
    )
    with pytest.raises(MetastudyContractError, match="fields do not match"):
        operator_state.validate_state_payload(with_embedded_evidence)

    decision["selected_reduction"] = [12.0, 16.0]
    for evaluation in decision["evaluations"]:
        if evaluation["reduction"] == [12.0, 16.0]:
            evaluation.update(
                worst_experiment_control_separation=1_000.0,
                repeated_anchor_drift=0.0,
                within_acquisition_observation_range=0.0,
                eligible_experiment_count=8,
                anchor_ordered_acquisition_count=5,
                co_measured_anchor_acquisition_count=5,
                loo_same_or_adjacent_fraction=1.0,
                eligible=True,
                blockers=[],
            )
    payload["generation_digest"] = operator_state.canonical_digest(body)

    with pytest.raises(MetastudyContractError, match="descriptive support and phase gates"):
        operator_state.validate_state_payload(payload)
