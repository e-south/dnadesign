"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/materialize/test_identity.py

Owner-aligned materialize contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    BiologicalReplicateIdentityScope,
    ReaderEvidenceBindingSet,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    profile_from_dict,
    profile_to_dict,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import DEFAULT_PROTOCOL
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize.service import (
    materialize_record_evidence,
)

from ._support import (
    _ontology,
    _policy,
    _rehash,
    _source_closed_inputs,
)


def test_materializer_keeps_unknown_replicate_identity_separate_from_acquisition(
    tmp_path: Path,
) -> None:
    record, bindings = _source_closed_inputs(tmp_path)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete", result.blockers
    first = result.candidate_evidence[0]
    assert {row.biological_replicate_id for row in first.profile.measurements} == {None}
    assert {row.acquisition_id for row in first.profile.measurements} == {record.experiment_id}
    assert {row.within_acquisition_observation_count for row in first.profile.measurements} == {6}
    assert first.profile.dose_uncertainties[0].biological_replicate_count == 0
    assert first.profile.dose_uncertainties[0].normalized_reporter_response.reason == (
        "biological_replicate_identity_unknown"
    )


def test_materializer_admits_unknown_replicate_declaration_without_guessing(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path, replicate_kind="unknown")

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete", result.blockers
    assert all(
        measurement.biological_replicate_id is None
        for evidence in result.candidate_evidence
        for measurement in evidence.profile.measurements
    )


def test_materializer_scopes_declared_replicate_labels_by_subject_and_condition(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(
        tmp_path,
        replicate_identity_field="biological_replicate_id",
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete", result.blockers
    profile = result.candidate_evidence[0].profile
    assert {row.biological_replicate_id for row in profile.measurements} == {
        "replicate-1",
        "replicate-2",
    }
    assert len({(row.condition_id, row.biological_replicate_id) for row in profile.measurements}) == 6
    assert profile.dose_uncertainties[0].biological_replicate_count == 2
    assert profile.dose_uncertainties[0].normalized_reporter_response.reason == "insufficient_valid_resamples"


def test_materializer_accepts_distinct_replicate_labels_in_each_condition(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(
        tmp_path,
        replicate_identity_field="biological_replicate_id",
    )
    frame = pd.read_parquet(record.path)
    condition_prefix = {
        condition: prefix
        for condition, prefix in zip(
            sorted(set(frame["treatment"].astype(str))),
            ("baseline", "dose", "positive"),
            strict=True,
        )
    }
    frame["biological_replicate_id"] = frame.apply(
        lambda row: (
            f"{condition_prefix[str(row['treatment'])]}-{1 if str(row['position']).startswith('group-1') else 2}"
        ),
        axis=1,
    )
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)
    scopes = tuple(
        BiologicalReplicateIdentityScope(condition_value=condition, biological_replicate_id=replicate_id)
        for condition, replicate_id in sorted(
            set(zip(frame["treatment"].astype(str), frame["biological_replicate_id"].astype(str)))
        )
    )
    object.__setattr__(bindings.rows[0], "biological_replicate_identity_scopes", scopes)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete", result.blockers
    profile = result.candidate_evidence[0].profile
    scoped_ids = {(row.source_condition_value, row.biological_replicate_id) for row in profile.measurements}
    assert scoped_ids == {(scope.condition_value, scope.biological_replicate_id) for scope in scopes}
    assert profile.dose_uncertainties[0].biological_replicate_count == 2


def test_materializer_joins_reused_design_id_on_full_reader_identity(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    frame["assay_subject_id"] = frame["position"].map(
        lambda value: "assay-subject-a" if str(value).startswith("group-1") else "assay-subject-b"
    )
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)
    base = bindings.rows[0]
    bindings = ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id=bindings.schema_id,
        subject_binding_set_id=bindings.subject_binding_set_id,
        rows=(
            replace(
                base,
                raw_assay_subject_id="assay-subject-a",
                subject_id=DEFAULT_PROTOCOL.anchor_subject_order[0],
                observation_identity_values=(
                    "group-1-position-1",
                    "group-1-position-2",
                    "group-1-position-3",
                ),
            ),
            replace(
                base,
                raw_assay_subject_id="assay-subject-b",
                subject_id=DEFAULT_PROTOCOL.anchor_subject_order[1],
                observation_identity_values=(
                    "group-2-position-1",
                    "group-2-position-2",
                    "group-2-position-3",
                ),
            ),
        ),
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete", result.blockers
    assert len(result.candidate_evidence) == 10
    assert {row.profile.subject_id for row in result.candidate_evidence} == set(DEFAULT_PROTOCOL.anchor_subject_order)
    assert {
        measurement.within_acquisition_observation_count
        for row in result.candidate_evidence
        for measurement in row.profile.measurements
    } == {3}


def test_materializer_blocks_multiple_reader_identities_for_one_subject(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    frame["assay_subject_id"] = frame["position"].map(
        lambda value: "assay-subject-a" if str(value).startswith("group-1") else "assay-subject-b"
    )
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)
    base = bindings.rows[0]
    bindings = ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id=bindings.schema_id,
        subject_binding_set_id=bindings.subject_binding_set_id,
        rows=(
            replace(
                base,
                raw_assay_subject_id="assay-subject-a",
                observation_identity_values=(
                    "group-1-position-1",
                    "group-1-position-2",
                    "group-1-position-3",
                ),
            ),
            replace(
                base,
                raw_assay_subject_id="assay-subject-b",
                observation_identity_values=(
                    "group-2-position-1",
                    "group-2-position-2",
                    "group-2-position-3",
                ),
            ),
        ),
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "blocked"
    assert result.blockers == ("multiple_reader_identities_for_subject",)
    assert result.candidate_evidence == ()


def test_materializer_ignores_unobserved_binding_for_observed_subject(tmp_path: Path) -> None:
    record, observed_bindings = _source_closed_inputs(tmp_path)
    base = observed_bindings.rows[0]
    bindings = ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id=observed_bindings.schema_id,
        subject_binding_set_id=observed_bindings.subject_binding_set_id,
        rows=(
            base,
            replace(
                base,
                raw_design_id="unobserved-reader-alias",
                observation_identity_values=("unobserved-position",),
            ),
        ),
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete", result.blockers
    assert {row.profile.provenance.evidence_binding_artifact_digest for row in result.candidate_evidence} == {
        bindings.artifact_digest
    }
    first_profile = result.candidate_evidence[0].profile
    assert first_profile.provenance.raw_design_id == base.raw_design_id
    assert first_profile.provenance.raw_assay_subject_id == base.raw_assay_subject_id
    assert profile_from_dict(profile_to_dict(first_profile), evidence_bindings=bindings) == first_profile


def test_materializer_blocks_unbound_observed_reader_identity(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    base = bindings.rows[0]
    bindings = ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id=bindings.schema_id,
        subject_binding_set_id=bindings.subject_binding_set_id,
        rows=(
            replace(
                base,
                subject_id=None,
                binding_state="unbound",
                binding_reason="no_exact_subject_alias_match",
            ),
        ),
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "blocked"
    assert result.blockers == ("sample_subject_not_bound",)


def test_materializer_rejects_ambiguous_partial_reader_identity(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    base = bindings.rows[0]
    bindings = ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id=bindings.schema_id,
        subject_binding_set_id=bindings.subject_binding_set_id,
        rows=(
            replace(
                base,
                raw_assay_subject_id="assay-subject-a",
                subject_id=DEFAULT_PROTOCOL.anchor_subject_order[0],
            ),
            replace(
                base,
                raw_assay_subject_id="assay-subject-b",
                subject_id=DEFAULT_PROTOCOL.anchor_subject_order[1],
            ),
        ),
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "blocked"
    assert result.blockers == ("sample_subject_identity_ambiguous",)
