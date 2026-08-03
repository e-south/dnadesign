"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/materialize/test_reference_profiles.py

Tests optional reference normalization without dropping raw profiles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jsonschema
import pandas as pd
import pytest
import yaml
from referencing import Registry, Resource

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    EndpointReduction,
    ReferenceNormalizationUnavailable,
    ReporterMeasurementProfile,
    ReporterResponseContractError,
    profile_from_dict,
    profile_to_dict,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import DEFAULT_PROTOCOL
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    profile_audit_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.condition_ontology import (
    DEFAULT_CONDITION_ONTOLOGY,
    ReporterResponseConditionOntology,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.evidence_projection import (
    parse_profile_evidence_projection,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize.service import (
    materialize_record_evidence,
)

from ._support import _ontology, _policy, _rehash, _source_closed_inputs


def _measurement_profile_validator() -> jsonschema.Draft202012Validator:
    root = next(parent for parent in Path(__file__).parents if (parent / "pyproject.toml").exists())
    schema_root = root / "docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/schemas"
    shared = yaml.safe_load((schema_root / "rt-lnrna-reporter-response-profile.schema.yaml").read_text())
    schema = yaml.safe_load((schema_root / "rt-lnrna-reporter-measurement-profile.schema.yaml").read_text())
    registry = Registry().with_resource(shared["$id"], Resource.from_contents(shared))
    return jsonschema.Draft202012Validator(schema, registry=registry)


def _materialize_without_observed_positive(tmp_path: Path, *, ontology: ReporterResponseConditionOntology):
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    frame = frame.loc[~frame["treatment"].eq("200 nm aTc; 0 uM IPTG")]
    frame.to_parquet(record.path, index=False)
    _rehash(record, bindings)
    result = materialize_record_evidence(
        record=record,
        bindings=bindings,
        ontology=ontology,
        observation_policy=_policy(),
        protocol=replace(DEFAULT_PROTOCOL, condition_ontology_digest=ontology.digest),
    )
    return bindings, result


def test_materializer_preserves_raw_profiles_when_no_positive_control_is_declared(tmp_path: Path) -> None:
    ontology = ReporterResponseConditionOntology(
        ontology_id="rt_lnrna_reporter_response_conditions_without_reference.v1",
        conditions=tuple(row for row in DEFAULT_CONDITION_ONTOLOGY.conditions if row.role != "positive_control"),
        sample_type_value=DEFAULT_CONDITION_ONTOLOGY.sample_type_value,
        reporter_channel=DEFAULT_CONDITION_ONTOLOGY.reporter_channel,
        normalizer_channel=DEFAULT_CONDITION_ONTOLOGY.normalizer_channel,
        ratio_channel=DEFAULT_CONDITION_ONTOLOGY.ratio_channel,
    )
    bindings, result = _materialize_without_observed_positive(tmp_path, ontology=ontology)

    assert result.status == "complete"
    assert result.omissions == ()
    assert all(isinstance(row.profile, ReporterMeasurementProfile) for row in result.candidate_evidence)
    first = result.candidate_evidence[0].profile
    assert {row.role for row in first.measurements} == {"baseline", "dose"}
    assert all(row.rfp > 0 and row.od600 > 0 and row.rfp_over_od600 > 0 for row in first.measurements)
    assert first.reference_normalization.reason == "positive_control_not_declared"
    payload = profile_to_dict(first)
    _measurement_profile_validator().validate(payload)
    assert profile_from_dict(payload, evidence_bindings=bindings) == first
    projection = parse_profile_evidence_projection(
        {"profile": profile_to_dict(first), "audit": profile_audit_payload(result.candidate_evidence[0].audit)},
        index=0,
    )
    assert projection.profile.reference_normalization == first.reference_normalization

    malformed = []
    for path in (("provenance", "reader_record_kind"), ("observation_policy", "contract_id")):
        candidate = deepcopy(payload)
        candidate[path[0]][path[1]] = "invalid"
        malformed.append(candidate)
    candidate = deepcopy(payload)
    candidate["eligibility"]["optimization_status"] = "eligible"
    malformed.append(candidate)
    for candidate in malformed:
        with pytest.raises(jsonschema.ValidationError):
            _measurement_profile_validator().validate(candidate)


def test_raw_profile_requires_immutable_measurements_and_a_declared_reduction(tmp_path: Path) -> None:
    _, result = _materialize_without_observed_positive(tmp_path, ontology=_ontology())
    profile = result.candidate_evidence[0].profile
    assert isinstance(profile, ReporterMeasurementProfile)

    with pytest.raises(ReporterResponseContractError, match="typed condition measurements"):
        replace(profile, measurements=list(profile.measurements))
    with pytest.raises(ReporterResponseContractError, match="reduction must be endpoint or time_window"):
        replace(
            profile,
            reduction=profile.observation_policy.biological_replicate_uncertainty_policy,
        )


def test_raw_endpoint_profile_requires_channel_derived_ratios(tmp_path: Path) -> None:
    _, result = _materialize_without_observed_positive(tmp_path, ontology=_ontology())
    profile = result.candidate_evidence[0].profile
    assert isinstance(profile, ReporterMeasurementProfile)
    rows = (
        replace(profile.measurements[0], rfp_over_od600=profile.measurements[0].rfp_over_od600 + 1.0),
        *profile.measurements[1:],
    )

    with pytest.raises(ReporterResponseContractError, match="rfp_over_od600 must equal rfp / od600"):
        replace(
            profile,
            reduction=EndpointReduction(recorded_time_h=10.0),
            measurements=rows,
        )


@pytest.mark.parametrize(
    ("reason", "condition_id"),
    [
        ("positive_control_not_declared", "positive"),
        ("positive_control_observations_missing", None),
    ],
)
def test_reference_absence_discriminator_has_schema_runtime_parity(
    tmp_path: Path,
    reason: str,
    condition_id: str | None,
) -> None:
    bindings, result = _materialize_without_observed_positive(tmp_path, ontology=_ontology())
    payload = profile_to_dict(result.candidate_evidence[0].profile)
    payload["reference_normalization"].update(reason=reason, positive_control_condition_id=condition_id)

    with pytest.raises(ReporterResponseContractError):
        ReferenceNormalizationUnavailable(reason=reason, positive_control_condition_id=condition_id)
    with pytest.raises(jsonschema.ValidationError):
        _measurement_profile_validator().validate(payload)


def test_nonpositive_reference_separation_does_not_drop_raw_profile(tmp_path: Path) -> None:
    record, bindings = _source_closed_inputs(tmp_path)
    frame = pd.read_parquet(record.path)
    positive = frame["treatment"].eq("200 nm aTc; 0 uM IPTG")
    frame.loc[positive & frame["channel"].eq("RFP"), "value"] = (
        frame.loc[positive & frame["channel"].eq("OD600"), "value"].to_numpy() * 100.0
    )
    frame.loc[positive & frame["channel"].eq("RFP/OD600"), "value"] = 100.0
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
    assert result.omissions == ()
    assert all(isinstance(row.profile, ReporterMeasurementProfile) for row in result.candidate_evidence)
    assert {row.profile.reference_normalization.reason for row in result.candidate_evidence} == {
        "positive_control_separation_not_positive"
    }
    payload = profile_to_dict(result.candidate_evidence[0].profile)
    _measurement_profile_validator().validate(payload)
    payload["reference_normalization"]["reason"] = "positive_control_observations_missing"
    with pytest.raises(ReporterResponseContractError):
        profile_from_dict(payload, evidence_bindings=bindings)
    with pytest.raises(jsonschema.ValidationError):
        _measurement_profile_validator().validate(payload)


def test_declared_but_unobserved_positive_control_preserves_raw_profile(tmp_path: Path) -> None:
    _, result = _materialize_without_observed_positive(tmp_path, ontology=_ontology())

    assert result.status == "complete"
    assert result.omissions == ()
    assert {row.profile.reference_normalization.reason for row in result.candidate_evidence} == {
        "positive_control_observations_missing"
    }
    _measurement_profile_validator().validate(profile_to_dict(result.candidate_evidence[0].profile))
