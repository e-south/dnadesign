"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/test_materialize.py

Source-closed materialization tests for the reporter-response meta-study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    BiologicalReplicateIdentityScope,
    ReaderDataframeRecordRef,
    ReaderEvidenceBinding,
    ReaderEvidenceBindingSet,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    ReporterResponseObservationPolicy,
    TemporalSelectedRow,
    TimeWindowReduction,
    UncertaintyPolicy,
    profile_from_dict,
    profile_to_dict,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    MaterializationOmission,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    profile_digest,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.condition_ontology import (
    DEFAULT_CONDITION_ONTOLOGY,
    ReporterResponseConditionOntology,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize import (
    _condition_summary,
    _growth_phase_strata,
    materialize_record_evidence,
)

_REVISION_DIGEST = "sha256:" + "a" * 64
_SUBJECT_ID = DEFAULT_PROTOCOL.anchor_subject_order[1]
_READER_REDUCTION_SOURCE_DIGEST = "sha256:af3e7603928d3fd6f2b4a2fbb3e33d0309986b99473da834e1a8b5a9e6c36ada"
_READER_CONTRACT_SOURCE_DIGEST = "sha256:8c5cc9bf8dfa68eb2102c002c39eb7f4e7119a95712bff1cdc8c873d60d797b0"


def _reader_reduce_trace_rows(
    rows: tuple[TemporalSelectedRow, ...],
    *,
    temporal_policy,
) -> float:
    phd_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "reader").is_dir())
    reader_root = phd_root / "reader"
    sources = {
        reader_root / "src/reader_workbench/domains/time_series/reduction.py": _READER_REDUCTION_SOURCE_DIGEST,
        reader_root / "src/reader_workbench/domains/time_series/contracts.py": _READER_CONTRACT_SOURCE_DIGEST,
    }
    for path, expected in sources.items():
        observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"Reader conformance source changed: {path}"
    traces = []
    for observation_identity in sorted({row.observation_identity for row in rows}):
        trace = sorted(
            (row for row in rows if row.observation_identity == observation_identity),
            key=lambda row: row.time_h,
        )
        traces.append(
            {
                "trace_id": observation_identity,
                "times": [row.time_h for row in trace],
                "values": [row.value for row in trace],
                "policy_clipped": [row.value_policy_clipped for row in trace],
                "instrument_overflow": [row.value_instrument_overflow for row in trace],
                "bound_kinds": [row.value_bound_kind for row in trace],
            }
        )
    payload = {"spec": temporal_policy.to_reader_mapping(), "traces": traces}
    script = """
import json, sys
import numpy as np
from reader_workbench.domains.time_series import TemporalReductionSpec, reduce_temporal_trace
payload = json.load(sys.stdin)
spec = TemporalReductionSpec.from_mapping(payload["spec"])
assert spec.to_mapping() == payload["spec"]
outputs = []
for trace in payload["traces"]:
    result = reduce_temporal_trace(
        np.asarray(trace["times"], dtype=float),
        np.asarray(trace["values"], dtype=float),
        spec=spec,
        trace_id=trace["trace_id"],
        policy_clipped=np.asarray(trace["policy_clipped"], dtype=bool),
        instrument_overflow=np.asarray(trace["instrument_overflow"], dtype=bool),
        bound_kinds=np.asarray(trace["bound_kinds"], dtype=str),
    )
    outputs.append(result.value)
json.dump(outputs, sys.stdout)
"""
    completed = subprocess.run(
        [str(reader_root / ".venv/bin/python"), "-c", script],
        input=json.dumps(payload, sort_keys=True),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip())
    return float(statistics.median(json.loads(completed.stdout)))


def _ontology(*, optional_doses: bool = False) -> ReporterResponseConditionOntology:
    del optional_doses
    return DEFAULT_CONDITION_ONTOLOGY


def _policy() -> ReporterResponseObservationPolicy:
    return ReporterResponseObservationPolicy(
        policy_id="rt_lnrna_reporter_response_observation_policy.v3",
        pairing_kind="pooled_controls_by_design",
        within_acquisition_reduction_statistic="median",
        biological_replicate_uncertainty_policy=UncertaintyPolicy(
            minimum_biological_replicates=2,
            biological_replicate_reduction_statistic="median",
        ),
    )


def _rows(*, quality_columns: bool = True, optional_doses: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timepoints = tuple(index / 6.0 for index in range(109))
    for observation_group in ("group-1", "group-2"):
        for position_index, position in enumerate(
            (
                f"{observation_group}-position-1",
                f"{observation_group}-position-2",
                f"{observation_group}-position-3",
            )
        ):
            treatments = [
                ("0 nm aTc; 0 uM IPTG", 0.0),
                ("200 nm aTc; 0 uM IPTG", 40.0),
                ("0 nm aTc; 500 uM IPTG", 20.0),
            ]
            if optional_doses:
                treatments.extend((("0 nm aTc; 5 uM IPTG", 5.0), ("0 nm aTc; 50 uM IPTG", 12.0)))
            for treatment, offset in treatments:
                for time_h in timepoints:
                    od = 1.0 + position_index / 10.0
                    for channel, value in (
                        ("RFP", (100.0 + offset) * od),
                        ("OD600", od),
                        ("RFP/OD600", 100.0 + offset),
                    ):
                        row: dict[str, object] = {
                            "type": "SAMPLE",
                            "position": position,
                            "time": time_h,
                            "channel": channel,
                            "value": value,
                            "treatment": treatment,
                            "design_id": "reader-anchor-alias",
                        }
                        if quality_columns:
                            row.update(
                                value_policy_clipped=False,
                                value_instrument_overflow=False,
                                value_bound_kind="exact",
                            )
                        rows.append(row)
    return rows


def _source_closed_inputs(
    tmp_path: Path,
    *,
    replicate_kind: str = "biological",
    replicate_identity_field: str | None = None,
    quality_columns: bool = True,
    optional_doses: bool = False,
) -> tuple[ReaderDataframeRecordRef, ReaderEvidenceBindingSet]:
    experiment_id = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[0]
    artifact = tmp_path / "outputs" / "artifacts" / "sample_measurements" / "df.parquet"
    artifact.parent.mkdir(parents=True)
    frame = pd.DataFrame(_rows(quality_columns=quality_columns, optional_doses=optional_doses))
    if replicate_identity_field is not None:
        frame[replicate_identity_field] = frame["position"].map(
            lambda value: "replicate-1" if str(value).startswith("group-1") else "replicate-2"
        )
    frame.to_parquet(artifact, index=False)
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "outputs" / "manifests" / "records.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    record = ReaderDataframeRecordRef._from_source_closed_reader(
        reader_root=tmp_path,
        experiment_id=experiment_id,
        protocol_id="plate_reader/single_reporter_screen",
        replicate_kind=replicate_kind,
        replicate_identity_field=replicate_identity_field,
        record_id="sample_measurements/df",
        record_kind="dataframe_artifact",
        record_schema_version=6,
        revision=1,
        revision_digest=_REVISION_DIGEST,
        contract_id="plate_reader.annotated.v1",
        reader_path="artifacts/sample_measurements/df.parquet",
        path=artifact,
        manifest_path=manifest,
        content_digest=digest,
    )
    binding = ReaderEvidenceBinding(
        reader_experiment_id=experiment_id,
        reader_protocol_id=record.protocol_id,
        reader_replicate_kind=replicate_kind,
        reader_replicate_identity_field=replicate_identity_field,
        reader_record_id=record.record_id,
        reader_record_kind=record.record_kind,
        reader_record_schema_version=record.record_schema_version,
        reader_record_revision=record.revision,
        reader_record_revision_digest=record.revision_digest,
        reader_record_contract_id=record.contract_id,
        reader_record_content_digest=record.content_digest,
        reader_record_path=record.reader_path,
        raw_design_id="reader-anchor-alias",
        raw_assay_subject_id=None,
        subject_id=_SUBJECT_ID,
        observation_identity_field="position",
        observation_identity_values=tuple(
            f"group-{group}-position-{position}" for group in (1, 2) for position in (1, 2, 3)
        ),
        biological_replicate_identity_scopes=(
            tuple(
                BiologicalReplicateIdentityScope(
                    condition_value=condition,
                    biological_replicate_id=replicate_id,
                )
                for condition in sorted(set(frame["treatment"].astype(str)))
                for replicate_id in ("replicate-1", "replicate-2")
            )
            if replicate_identity_field is not None
            else ()
        ),
        binding_state="bound",
        binding_reason="exact_subject_alias_match",
    )
    bindings = ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id="rt_lnrna_reader_evidence_bindings_v4",
        subject_binding_set_id="rt_lnrna_subject_bindings_v1",
        rows=(binding,),
    )
    return record, bindings


def _rehash(record: ReaderDataframeRecordRef, bindings: ReaderEvidenceBindingSet) -> None:
    digest = "sha256:" + hashlib.sha256(record.path.read_bytes()).hexdigest()
    object.__setattr__(record, "content_digest", digest)
    object.__setattr__(bindings.rows[0], "reader_record_content_digest", digest)


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


def test_materializer_blocks_records_without_exact_quality_provenance(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path, quality_columns=False)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "blocked"
    assert result.blockers == ("required_quality_columns_missing",)


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("value_policy_clipped", True),
        ("value_instrument_overflow", True),
        ("value_bound_kind", "upper_bound"),
    ),
)
def test_materializer_omits_only_censored_subject_window_when_policy_rejects_it(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    selected = frame.loc[
        frame["time"].between(4.0, 8.0) & frame["treatment"].eq("0 nm aTc; 0 uM IPTG") & frame["channel"].eq("RFP")
    ]
    frame.loc[selected.index[0], column] = value
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "partial"
    assert result.attempt.blockers == ()
    assert len(result.candidate_evidence) == 4
    assert any(row.code == "censored_observations_rejected" for row in result.omissions)
    assert all(row.subject_id == _SUBJECT_ID for row in result.omissions)
    assert "window-4-8h" in {row.reduction_id for row in result.omissions}


def test_censored_optional_sensitivity_window_does_not_block_primary_candidates(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    sensitivity_only = frame.loc[frame["time"].eq(17.0) & frame["channel"].eq("RFP")]
    frame.loc[sensitivity_only.index[0], "value_policy_clipped"] = True
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete"
    assert len(result.candidate_evidence) == 5
    assert len(result.centered_window_evidence) == 9
    assert result.attempt.blockers == ()
    assert result.sensitivity_coverage is not None
    assert result.sensitivity_coverage.omissions
    assert {row.code for row in result.sensitivity_coverage.omissions} == {"censored_observations_rejected"}
    assert "window-11-17h" in {row.reduction_id for row in result.sensitivity_coverage.omissions}
    assert result.sensitivity_coverage is not None
    assert result.attempt.attempt_digest == result.sensitivity_coverage.materialization_attempt_digest


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("value_policy_clipped", True),
        ("value_instrument_overflow", True),
        ("value_bound_kind", "upper_bound"),
    ),
)
def test_censored_normalizer_outside_reduction_does_not_change_growth_phase_scale(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    normalizer = frame["channel"].eq("OD600")
    frame.loc[normalizer, "value"] = frame.loc[normalizer, "time"].map(
        lambda time_h: math.exp(0.15 * float(time_h) + 0.01 * float(time_h) ** 2)
    )
    censored = frame["channel"].eq("OD600") & frame["treatment"].eq("0 nm aTc; 0 uM IPTG") & frame["time"].eq(2.0)
    frame.loc[~censored].to_parquet(record.path, index=False)
    _rehash(record, bindings)
    baseline = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    frame.loc[censored, "value"] = 1e30
    frame.loc[censored, column] = value
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete"
    assert tuple(row.audit.growth_phase_strata for row in result.candidate_evidence) == tuple(
        row.audit.growth_phase_strata for row in baseline.candidate_evidence
    )


def test_nonnumeric_normalizer_outside_reduction_is_excluded_from_growth_phase_scale() -> None:
    reduction = TimeWindowReduction(
        recorded_start_time_h=4.0,
        recorded_end_time_h=8.0,
        summary_statistic="median",
        ratio_reduction_order="ratio_then_reduce",
    )
    rows = [
        {
            "channel": "OD600",
            "time": time_h,
            "treatment": treatment,
            "value": str(math.exp(0.15 * time_h + 0.01 * time_h**2)),
            "value_policy_clipped": False,
            "value_instrument_overflow": False,
            "value_bound_kind": "exact",
        }
        for treatment in DEFAULT_CONDITION_ONTOLOGY.by_treatment_label
        for time_h in (index / 6.0 for index in range(109))
    ]
    frame = pd.DataFrame(rows)
    invalid = frame["treatment"].eq("0 nm aTc; 0 uM IPTG") & frame["time"].eq(2.0)
    expected = _growth_phase_strata(
        frame.loc[~invalid],
        reduction=reduction,
        ontology=_ontology(),
        protocol=DEFAULT_PROTOCOL,
    )
    frame.loc[invalid, "value"] = "not-numeric"

    observed = _growth_phase_strata(
        frame,
        reduction=reduction,
        ontology=_ontology(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert observed == expected


def test_condition_summary_uses_reader_absolute_boundary_tolerance_without_relative_slack() -> None:
    start = 1_000_000.0
    reduction = TimeWindowReduction(
        recorded_start_time_h=start,
        recorded_end_time_h=start + 4.0,
        summary_statistic="median",
        ratio_reduction_order="ratio_then_reduce",
    )
    rows: list[dict[str, object]] = []
    times = [start + index / 6.0 + 1e-8 for index in range(25)]
    for position in ("A1", "A2", "A3"):
        for time_h in times:
            for channel, value in (("RFP", 100.0), ("OD600", 1.0), ("RFP/OD600", 100.0)):
                rows.append({"position": position, "time": time_h, "channel": channel, "value": value})

    assert (
        _condition_summary(
            pd.DataFrame(rows),
            _ontology(),
            reduction=reduction,
            protocol=DEFAULT_PROTOCOL,
        )
        is None
    )


def test_materializer_derives_profiles_windows_sensitivities_and_audits(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete"
    assert result.blockers == ()
    assert len(result.candidate_evidence) == 5
    assert len(result.endpoint_evidence) == 5
    assert len(result.centered_window_evidence) == 10
    first = result.candidate_evidence[0]
    assert first.profile.provenance.is_source_closed
    assert {row.acquisition_id for row in first.profile.measurements} == {record.experiment_id}
    assert first.audit.required_observation_count > 0
    assert first.audit.clipped_observation_count == 0
    assert first.audit.overflow_observation_count == 0
    assert first.audit.is_derivation_closed
    assert first.audit.condition_ontology_digest == DEFAULT_PROTOCOL.condition_ontology_digest
    assert first.audit.growth_phase_strata
    assert all(row.acquisition_id == record.experiment_id for row in first.profile.measurements)
    assert result.attempt.experiment_id == record.experiment_id
    assert result.attempt.status == "complete"
    assert result.attempt.reader_record_identity.reader_record_revision_digest == record.revision_digest
    assert result.attempt.reader_record_identity.reader_record_content_digest == record.content_digest
    assert result.attempt.candidate_profile_count == len(result.candidate_evidence)
    assert result.attempt.candidate_profile_digests == tuple(
        sorted(profile_digest(row.profile) for row in result.candidate_evidence)
    )


@pytest.mark.parametrize(
    ("time_h", "invalid_value"),
    (
        (0.0, 0.0),
        (18.0, math.inf),
    ),
)
def test_materializer_ignores_invalid_od_outside_candidate_slope_support(
    tmp_path: Path,
    time_h: float,
    invalid_value: float,
) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    irrelevant_od = frame["channel"].eq("OD600") & frame["time"].eq(time_h)
    frame.loc[irrelevant_od, "value"] = invalid_value
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete", result.omissions
    assert len(result.candidate_evidence) == len(DEFAULT_PROTOCOL.candidate_windows_h)


def test_materializer_omits_candidate_with_invalid_od_inside_required_slope_support(
    tmp_path: Path,
) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    required_od = frame["channel"].eq("OD600") & frame["time"].eq(4.0)
    frame.loc[required_od, "value"] = 0.0
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "partial"
    assert (
        MaterializationOmission(
            code="phase_not_estimable_temporal_support",
            subject_id=_SUBJECT_ID,
            reduction_id="window-4-8h",
        )
        in result.omissions
    )


def test_live_reader_matches_source_bound_temporal_conformance_probe(tmp_path: Path) -> None:
    phd_roots = [
        parent for parent in Path(__file__).resolve().parents if (parent / "reader/.venv/bin/python").is_file()
    ]
    if not phd_roots:
        pytest.skip("optional sibling Reader checkout is unavailable")
    record, bindings = _source_closed_inputs(tmp_path)
    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )
    reduction = result.candidate_evidence[0].profile.reduction
    rows = tuple(
        TemporalSelectedRow(
            observation_identity="A1",
            time_h=(4.0 + 5e-10 if index == 0 else 8.0 - 5e-10 if index == 24 else 4.0 + index / 6.0),
            value=100.0 + index,
        )
        for index in range(25)
    )

    assert _reader_reduce_trace_rows(rows, temporal_policy=reduction.temporal_policy) == 112.0


def test_materializer_rejects_rehashed_noncanonical_ontology(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    forged = ReporterResponseConditionOntology(
        ontology_id="rt_lnrna_reporter_response_conditions.v1",
        conditions=tuple(
            replace(row, treatment_label="forged baseline") if row.role == "baseline" else row
            for row in DEFAULT_CONDITION_ONTOLOGY.conditions
        ),
        sample_type_value=DEFAULT_CONDITION_ONTOLOGY.sample_type_value,
        reporter_channel=DEFAULT_CONDITION_ONTOLOGY.reporter_channel,
        normalizer_channel=DEFAULT_CONDITION_ONTOLOGY.normalizer_channel,
        ratio_channel=DEFAULT_CONDITION_ONTOLOGY.ratio_channel,
    )

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=forged,
        observation_policy=_policy(),
    )

    assert result.blockers == ("condition_ontology_digest_mismatch",)


def test_materializer_rejects_noncanonical_observation_policy(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    changed_policy = replace(_policy(), policy_id="caller-selected-policy")

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=DEFAULT_CONDITION_ONTOLOGY,
        observation_policy=changed_policy,
    )

    assert result.blockers == ("observation_policy_digest_mismatch",)


def test_materializer_rejects_unknown_sample_condition_without_inference(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    frame.loc[frame.index[0], "treatment"] = "undeclared-label"
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "blocked"
    assert result.blockers == ("sample_condition_not_declared",)


def test_materializer_rejects_irregular_time_grid_even_with_enough_points(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    mask = frame["time"].eq(5.0)
    frame.loc[mask, "time"] = 5.01
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "partial"
    assert result.blockers == ()
    assert result.attempt.status == "partial"
    assert result.attempt.reader_record_identity.reader_record_content_digest == record.content_digest
    omission = next(row for row in result.omissions if row.reduction_id == "window-4-8h")
    assert omission.code == "condition_or_channel_observations_incomplete"
    assert omission.subject_id == _SUBJECT_ID


def test_optional_doses_are_sensitivity_only_and_do_not_change_primary_profiles(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path, optional_doses=True)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(optional_doses=True),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete"
    assert all(row.profile.dose_grid_uM == (500.0,) for row in result.candidate_evidence)
    assert all(row.profile.dose_grid_uM == (5.0, 50.0, 500.0) for row in result.endpoint_evidence)
    assert all(row.profile.dose_grid_uM == (5.0, 50.0, 500.0) for row in result.centered_window_evidence)
    assert all(row.profile.eligibility.optimization_status == "ineligible" for row in result.endpoint_evidence)


def test_censored_optional_dose_rows_do_not_block_primary_estimand(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path, optional_doses=True)
    frame = pd.read_parquet(record.path)
    optional = frame.loc[
        frame["treatment"].eq("0 nm aTc; 5 uM IPTG") & frame["time"].eq(4.0) & frame["channel"].eq("RFP")
    ]
    frame.loc[optional.index[0], "value_policy_clipped"] = True
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)

    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=_ontology(optional_doses=True),
        observation_policy=_policy(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert result.status == "complete"
    assert len(result.candidate_evidence) == 5
    assert result.attempt.blockers == ()
    assert result.sensitivity_coverage is not None
    assert result.sensitivity_coverage.omissions
    assert {row.code for row in result.sensitivity_coverage.omissions} == {"censored_observations_rejected"}


def test_checked_in_condition_ontology_matches_typed_default() -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    source = (
        repo_root
        / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy"
        / "condition-ontology.yaml"
    )

    expected = json.loads(json.dumps(asdict(DEFAULT_CONDITION_ONTOLOGY)))
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == expected
