"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/reader_evidence_bindings/test_source_closure.py

Source-closure and digest-stability binding behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderEvidenceBindingError,
    build_reader_evidence_bindings,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import (
    SubjectBindingRegistry,
    load_registered_subject_bindings,
)

from ._fixtures import _repo_root, _resolve_record, _write_reader_record


def test_binding_builder_rejects_directly_constructed_registry(tmp_path: Path) -> None:
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
    loaded = load_registered_subject_bindings(repo_root=_repo_root())
    forged = SubjectBindingRegistry(
        schema_id=loaded.schema_id,
        study_id=loaded.study_id,
        binding_set_id=loaded.binding_set_id,
        subjects=loaded.subjects,
    )

    with pytest.raises(ReaderEvidenceBindingError, match="source-closed registry"):
        build_reader_evidence_bindings(record=_resolve_record(experiment), subject_registry=forged)


def test_binding_builder_rejects_reader_record_without_source_closure(tmp_path: Path) -> None:
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
    forged = replace(_resolve_record(experiment))

    with pytest.raises(ReaderEvidenceBindingError, match="source-closed Reader record"):
        build_reader_evidence_bindings(
            record=forged,
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


def test_partial_alias_match_remains_unbound(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-999-typo",
                "position": "colony-1",
            }
        ],
    )

    binding_set = build_reader_evidence_bindings(
        record=_resolve_record(experiment),
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    row = binding_set.rows[0]
    assert row.subject_id is None
    assert row.binding_state == "unbound"
    assert row.binding_reason == "partial_exact_subject_alias_match"


def test_conflicting_exact_aliases_are_rejected_as_ambiguous(tmp_path: Path) -> None:
    experiment = _write_reader_record(
        tmp_path,
        [
            {
                "design_id": "pES-retron-205-Eco1RT-G3-D01; pBbS2c-rfp",
                "assay_subject_id": "retron-206-Eco1RT-G3-D02",
                "position": "colony-1",
            }
        ],
    )

    with pytest.raises(ReaderEvidenceBindingError, match="conflicting exact aliases"):
        build_reader_evidence_bindings(
            record=_resolve_record(experiment),
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


def test_binding_builder_rechecks_artifact_digest_before_reading(tmp_path: Path) -> None:
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
    record.path.write_bytes(b"drift after record resolution")

    with pytest.raises(ReaderEvidenceBindingError, match="content digest changed"):
        build_reader_evidence_bindings(
            record=record,
            subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"record_schema_version": 5}, "record schema v6"),
        ({"revision": 0}, "revision must be a positive integer"),
        ({"revision_digest": "sha256:" + ("A" * 64)}, "revision_digest must be a lowercase sha256 digest"),
        ({"content_digest": "sha256:" + ("A" * 64)}, "content_digest must be a lowercase sha256 digest"),
    ],
)
def test_binding_builder_rejects_invalid_exact_record_identity(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
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
    record = replace(_resolve_record(experiment), **changes)

    with pytest.raises(ReaderEvidenceBindingError, match=message):
        build_reader_evidence_bindings(
            record=record,
            subject_registry=SubjectBindingRegistry(
                schema_id="fixture",
                study_id="rt_lnrna_sponging_construct_triage",
                binding_set_id="fixture",
                subjects=(),
            ),
        )


def test_binding_builder_parses_the_same_bytes_it_digest_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    replacement = tmp_path / "replacement.parquet"
    pd.DataFrame(
        [
            {
                "design_id": "pES-retron-206-Eco1RT-G3-D02; pBbS2c-rfp",
                "assay_subject_id": "retron-206-Eco1RT-G3-D02",
                "position": "colony-2",
            }
        ]
    ).to_parquet(replacement, index=False)
    replacement_bytes = replacement.read_bytes()
    read_bytes = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        data = read_bytes(path)
        if path == record.path:
            record.path.write_bytes(replacement_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)

    binding_set = build_reader_evidence_bindings(
        record=record,
        subject_registry=load_registered_subject_bindings(repo_root=_repo_root()),
    )

    assert binding_set.rows[0].subject_id == "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO"
