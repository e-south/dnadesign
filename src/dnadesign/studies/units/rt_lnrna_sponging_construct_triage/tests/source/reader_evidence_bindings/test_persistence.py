"""Binding artifact projection, loading, and atomic publication."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderEvidenceBindingError,
    build_reader_evidence_bindings,
    load_reader_evidence_bindings_json,
    materialize_reader_evidence_bindings_json,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import load_registered_subject_bindings

from ._fixtures import _repo_root, _resolve_record, _write_reader_record


def test_materialized_binding_rows_exclude_measurements_and_interpretations(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
                "OD600": 0.97,
                "RFP/OD600": 7654.0,
                "assay_score": 0.81,
            }
        ],
    )
    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )
    destination = tmp_path / "evidence-bindings.json"

    materialize_reader_evidence_bindings_json(binding_set, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    row = payload["bindings"][0]
    assert payload["artifact_id"] == binding_set.artifact_id
    assert payload["artifact_digest"] == binding_set.artifact_digest
    assert set(row) == {
        "reader_experiment_id",
        "reader_protocol_id",
        "reader_replicate_kind",
        "reader_replicate_identity_field",
        "reader_record_id",
        "reader_record_kind",
        "reader_record_schema_version",
        "reader_record_revision",
        "reader_record_revision_digest",
        "reader_record_contract_id",
        "reader_record_content_digest",
        "reader_record_path",
        "raw_design_id",
        "raw_assay_subject_id",
        "subject_id",
        "observation_identity_field",
        "observation_identity_values",
        "biological_replicate_identity_scopes",
        "binding_state",
        "binding_reason",
    }
    assert not ({"OD600", "RFP", "RFP/OD600", "assay_score", "measurement"} & set(row))
    assert payload["unbound_count"] == 0


def test_materialized_binding_loader_restores_source_closure_and_rejects_digest_drift(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    record = _resolve_record(experiment)
    registry = load_registered_subject_bindings(repo_root=_repo_root())
    binding_set = build_reader_evidence_bindings(record=record, subject_registry=registry)
    destination = tmp_path / "evidence-bindings.json"
    materialize_reader_evidence_bindings_json(binding_set, destination)

    loaded = load_reader_evidence_bindings_json(
        destination,
        record=record,
        subject_registry=registry,
    )

    assert loaded.is_source_closed
    assert loaded.artifact_id == binding_set.artifact_id
    assert loaded.artifact_digest == binding_set.artifact_digest

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["bindings"][0]["subject_id"] = "forged-subject"
    unsigned = dict(payload)
    unsigned.pop("artifact_digest")
    payload["artifact_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    )
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReaderEvidenceBindingError, match="no longer matches"):
        load_reader_evidence_bindings_json(
            destination,
            record=record,
            subject_registry=registry,
        )


def test_binding_publication_rejects_forged_sets_before_mutation_and_never_overwrites(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
            }
        ],
    )
    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )
    forged_destination = tmp_path / "forged" / "bindings.json"

    with pytest.raises(ReaderEvidenceBindingError, match="source-closed set"):
        materialize_reader_evidence_bindings_json(replace(binding_set), forged_destination)
    assert not forged_destination.parent.exists()

    destination = tmp_path / "bindings.json"
    materialize_reader_evidence_bindings_json(binding_set, destination)
    original = destination.read_bytes()
    with pytest.raises(ReaderEvidenceBindingError, match="already exists"):
        materialize_reader_evidence_bindings_json(binding_set, destination)
    assert destination.read_bytes() == original
    assert list(tmp_path.glob(".bindings.json.*.tmp")) == []
