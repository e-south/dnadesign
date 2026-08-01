"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/evidence/test_projection.py

Tests offline evidence projection identity and confinement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    ReporterResponseProfile,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    MetastudyContractError,
    decision_evidence_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    profile_source_identity_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
    canonical_digest,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.evidence_projection import (
    parse_profile_evidence_projection,
    profile_source_identity_projection,
)

from .._builders import (
    _evidence,
    _ready,
    evaluate_metastudy,
)


@pytest.mark.parametrize(
    ("readiness_path", "replacement"),
    (
        (("schema_id",), "wrong-readiness-schema"),
        (("source_identity", "route_id"), "wrong-route"),
        (("source_identity", "route_registry_path"), "wrong-registry.json"),
        (("source_identity", "route_registry_digest"), "sha256:" + "g" * 64),
        (("source_identity", "normalized_full_receipt_digest"), "sha256:" + "g" * 64),
        (("source_identity", "normalization"), "hash the whole receipt"),
        (("last_verified",), "2026-02-30"),
        (("selected_experiment_count",), 7),
        (("related_experiment_count",), 2),
        (("related_experiment_ids",), ["wrong-related-experiment"]),
    ),
)
def test_source_state_rejects_noncanonical_readiness_snapshot_with_recomputed_generation_digest(
    readiness_path: tuple[str, ...],
    replacement: object,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
        state as operator_state,
    )

    state_path = next(
        parent
        / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy/metastudy-state.yaml"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/studies/rt_lnrna_sponging_construct_triage").is_dir()
    )
    payload = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    readiness = copy.deepcopy(payload["readiness"])
    target = readiness
    for key in readiness_path[:-1]:
        target = target[key]
    target[readiness_path[-1]] = replacement
    payload["readiness"] = readiness
    if readiness_path == ("source_identity", "normalized_full_receipt_digest"):
        decision = copy.deepcopy(payload["decision"])
        decision["readiness"]["receipt_digest"] = replacement
        payload["decision"] = decision
    payload["generation_digest"] = operator_state.canonical_digest(
        {
            key: payload[key]
            for key in (
                "readiness",
                "decision",
                "objective_readiness",
                "sensitivity_evaluations",
                "sensitivity_coverage_receipts",
                "acquisition_projection",
            )
            if key in payload
        }
    )

    with pytest.raises(MetastudyContractError, match="readiness"):
        operator_state.validate_state_payload(payload)


def test_publication_projection_parser_mints_no_live_source_or_audit_closure() -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    payload = decision_evidence_payload(evidence, decision=selected)
    row = json.loads(json.dumps(payload["profiles"][0]))
    projection = parse_profile_evidence_projection(row, index=0)

    assert not isinstance(projection.profile, ReporterResponseProfile)
    assert not hasattr(projection.profile.provenance, "is_source_closed")
    assert projection.audit.is_derivation_closed is False


def test_live_and_offline_source_identity_bind_raw_reader_aliases_symmetrically() -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    payload = decision_evidence_payload(evidence, decision=selected)
    profile = evidence[0].profile
    row = next(item for item in payload["profiles"] if item["profile"]["profile_id"] == profile.profile_id)
    projection = parse_profile_evidence_projection(json.loads(json.dumps(row)), index=0).profile

    live_identity = profile_source_identity_payload(profile)
    offline_identity = profile_source_identity_projection(projection)
    assert offline_identity == live_identity
    assert live_identity["raw_design_id"] == profile.provenance.raw_design_id
    assert live_identity["raw_assay_subject_id"] == profile.provenance.raw_assay_subject_id
    assert live_identity["reader_protocol_id"] == profile.provenance.reader_protocol_id
    assert live_identity["reader_record_kind"] == profile.provenance.reader_record_kind
    assert live_identity["reader_record_path"] == profile.provenance.reader_record_path

    changed_offline = replace(
        projection,
        provenance=replace(projection.provenance, raw_design_id="changed-reader-alias"),
    )
    assert canonical_digest(profile_source_identity_projection(changed_offline)) != canonical_digest(offline_identity)


@pytest.mark.parametrize("reader_record_path", ["../outside.parquet", "/outside.parquet"])
def test_publication_projection_rejects_unconfined_reader_record_path(reader_record_path: str) -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    row = decision_evidence_payload(evidence, decision=selected)["profiles"][0]
    row["profile"]["provenance"]["reader_record_path"] = reader_record_path

    with pytest.raises(ValueError, match="reader_record_path must be outputs-relative"):
        parse_profile_evidence_projection(row, index=0)


def test_publication_projection_rejects_forged_null_raw_identity_after_digest_recomputation() -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    row = decision_evidence_payload(evidence, decision=selected)["profiles"][0]
    profile = row["profile"]
    provenance = profile["provenance"]
    provenance["raw_design_id"] = None
    provenance["raw_assay_subject_id"] = None
    audit = row["audit"]
    audit["profile_digest"] = canonical_digest(profile)
    audit["profile_source_digest"] = canonical_digest(
        {
            "raw_design_id": None,
            "raw_assay_subject_id": None,
            "reader_experiment_id": provenance["reader_experiment_id"],
            "reader_record_id": provenance["reader_record_id"],
            "reader_record_revision": provenance["reader_record_revision"],
            "reader_record_revision_digest": provenance["reader_record_revision_digest"],
            "reader_record_content_digest": provenance["reader_record_content_digest"],
            "reader_record_schema_version": provenance["reader_record_schema_version"],
            "reader_record_contract_id": provenance["reader_record_contract_id"],
            "evidence_binding_artifact_id": provenance["evidence_binding_artifact_id"],
            "evidence_binding_artifact_digest": provenance["evidence_binding_artifact_digest"],
            "observation_policy_identity": profile["observation_policy"]["digest"],
        }
    )
    audit_without_digest = {key: value for key, value in audit.items() if key != "artifact_digest"}
    audit["artifact_digest"] = canonical_digest(audit_without_digest)

    with pytest.raises(ValueError, match="at least one raw Reader identity"):
        parse_profile_evidence_projection(row, index=0)


@pytest.mark.parametrize("field_name", ["raw_design_id", "raw_assay_subject_id"])
def test_publication_projection_rejects_empty_raw_identity_coordinate(field_name: str) -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    row = decision_evidence_payload(evidence, decision=selected)["profiles"][0]
    row["profile"]["provenance"][field_name] = ""

    with pytest.raises(ValueError, match=rf"{field_name} must be non-empty text"):
        parse_profile_evidence_projection(row, index=0)


def _rehashed_raw_profile_row() -> dict[str, object]:
    evidence = _evidence(reference_normalized=False)
    selected = evaluate_metastudy(evidence, readiness=_ready())
    row = copy.deepcopy(decision_evidence_payload(evidence, decision=selected)["profiles"][0])
    profile = row["profile"]
    audit = row["audit"]
    audit["profile_digest"] = canonical_digest(profile)
    audit_without_digest = {key: value for key, value in audit.items() if key != "artifact_digest"}
    audit["artifact_digest"] = canonical_digest(audit_without_digest)
    return row


def test_publication_projection_rejects_rehashed_raw_profile_without_baseline() -> None:
    row = _rehashed_raw_profile_row()
    row["profile"]["measurements"] = [
        measurement for measurement in row["profile"]["measurements"] if measurement["role"] != "baseline"
    ]
    audit = row["audit"]
    audit["profile_digest"] = canonical_digest(row["profile"])
    audit["artifact_digest"] = canonical_digest(
        {key: value for key, value in audit.items() if key != "artifact_digest"}
    )

    with pytest.raises(ValueError, match="baseline and dose observations"):
        parse_profile_evidence_projection(row, index=0)


def test_publication_projection_rejects_rehashed_raw_profile_comparability_forgery() -> None:
    row = _rehashed_raw_profile_row()
    row["profile"]["comparability_key"] = "sha256:" + "f" * 64
    audit = row["audit"]
    audit["profile_digest"] = canonical_digest(row["profile"])
    audit["artifact_digest"] = canonical_digest(
        {key: value for key, value in audit.items() if key != "artifact_digest"}
    )

    with pytest.raises(ValueError, match="comparability_key"):
        parse_profile_evidence_projection(row, index=0)
