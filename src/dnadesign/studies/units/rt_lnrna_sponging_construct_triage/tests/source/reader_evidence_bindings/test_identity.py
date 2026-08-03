"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/reader_evidence_bindings/test_identity.py

Exact Reader identity binding behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderEvidenceBindingError,
    build_reader_evidence_bindings,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import load_registered_subject_bindings

from ._fixtures import _REVISION_DIGEST, _repo_root, _resolve_record, _write_reader_record


def test_d01_exact_aliases_bind_and_retain_reader_provenance(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-1",
                "time": 12.0,
                "RFP/OD600": 7654.0,
            },
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "colony-2",
                "time": 12.0,
                "RFP/OD600": 7012.0,
            },
        ],
    )

    record = _resolve_record(experiment)
    registry = load_registered_subject_bindings(repo_root=_repo_root())
    binding_set = build_reader_evidence_bindings(record=record, subject_registry=registry)

    assert binding_set.unbound_count == 0
    assert len(binding_set.rows) == 1
    row = binding_set.rows[0]
    assert row.subject_id == "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO"
    assert row.raw_design_id == "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp"
    assert row.raw_assay_subject_id == "retron-205-Eco1RT-G3-D01"
    assert row.reader_replicate_kind == "biological"
    assert row.reader_replicate_identity_field is None
    assert row.observation_identity_field == "position"
    assert row.observation_identity_values == ("colony-1", "colony-2")
    assert row.biological_replicate_identity_scopes == ()
    assert row.binding_state == "bound"
    assert row.binding_reason == "exact_subject_alias_match"
    assert row.reader_record_schema_version == 6
    assert row.reader_record_revision == 1
    assert row.reader_record_revision_digest == _REVISION_DIGEST
    assert row.reader_record_contract_id == "plate_reader.annotated.v1"
    assert row.reader_record_content_digest == _resolve_record(experiment).content_digest


def test_unknown_reader_evidence_remains_unknown_without_guessing(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A1",
            }
        ],
    )

    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment, replicate_kind="unknown"),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    row = binding_set.rows[0]
    assert row.reader_replicate_kind == "unknown"
    assert row.reader_replicate_identity_field is None
    assert row.observation_identity_field == "position"
    assert row.biological_replicate_identity_scopes == ()


def test_unknown_reader_evidence_rejects_a_declared_identity_field(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A1",
                "treatment": "0.0 µM aTc + 0.0 µM IPTG",
                "biological_replicate_id": "culture-1",
            }
        ],
    )
    record = _resolve_record(
        experiment,
        replicate_kind="unknown",
        replicate_identity_field="biological_replicate_id",
    )

    with pytest.raises(ReaderEvidenceBindingError, match="unknown replicate identity"):
        build_reader_evidence_bindings(
            record=record,
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


def test_biological_reader_evidence_uses_the_declared_replicate_identity(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A1",
                "treatment": "0.0 µM aTc + 0.0 µM IPTG",
                "biological_replicate_id": "culture-1",
            },
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-205-Eco1RT-G3-D01",
                "position": "A2",
                "treatment": "0.0 µM aTc + 0.0 µM IPTG",
                "biological_replicate_id": "culture-2",
            },
        ],
    )
    record = _resolve_record(
        experiment,
        replicate_kind="biological",
        replicate_identity_field="biological_replicate_id",
    )

    binding_set = build_reader_evidence_bindings(
        record=record,
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    row = binding_set.rows[0]
    assert row.reader_replicate_kind == "biological"
    assert row.reader_replicate_identity_field == "biological_replicate_id"
    assert row.observation_identity_field == "position"
    assert row.observation_identity_values == ("A1", "A2")
    assert tuple(
        (scope.condition_value, scope.biological_replicate_id) for scope in row.biological_replicate_identity_scopes
    ) == (
        ("0.0 µM aTc + 0.0 µM IPTG", "culture-1"),
        ("0.0 µM aTc + 0.0 µM IPTG", "culture-2"),
    )


def test_unknown_alias_is_reported_unbound_without_guessing(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-999-unknown; pBbS2c-rfp",
                "assay_subject_id": "retron-999-unknown",
                "position": "colony-1",
            }
        ],
    )

    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    assert binding_set.unbound_count == 1
    row = binding_set.rows[0]
    assert row.subject_id is None
    assert row.binding_state == "unbound"
    assert row.binding_reason == "no_exact_subject_alias_match"
    assert row.raw_design_id == "pES-retron-999-unknown; pBbS2c-rfp"
