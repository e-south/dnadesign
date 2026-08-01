"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/test_profile.py

Contract tests for scoped biological-replicate reporter-response profiles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    BiologicalReplicateIdentityScope,
    ReaderEvidenceBinding,
    ReaderEvidenceBindingSet,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    ConditionMeasurement,
    ControlAssignment,
    DoseUncertainty,
    EndpointReduction,
    NotEstimableMetricUncertainty,
    PairingPolicy,
    ReporterResponseContractError,
    ReporterResponseObservationPolicy,
    TimeWindowReduction,
    UncertaintyPolicy,
    build_reporter_response_profile,
    profile_from_dict,
    profile_to_dict,
    require_comparable_profiles,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.canonical import (
    comparability_key,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def test_profile_contract_uses_the_bounded_package_layout() -> None:
    study_root = Path(__file__).resolve().parents[2]
    reporter_response_root = study_root / "reporter_response"

    assert (reporter_response_root / "profile").is_dir()
    assert not (reporter_response_root / "profile.py").exists()


def _bindings(
    *,
    subject_id: str = "subject-a",
    experiment_id: str = "experiment-a",
    biological_replicate_ids: tuple[str, ...] = ("replicate-1",),
):
    row = ReaderEvidenceBinding(
        reader_experiment_id=experiment_id,
        reader_protocol_id="plate_reader/single_reporter_screen",
        reader_replicate_kind="biological",
        reader_replicate_identity_field=("biological_replicate_id" if biological_replicate_ids else None),
        reader_record_id="sample_measurements/df",
        reader_record_kind="dataframe_artifact",
        reader_record_schema_version=6,
        reader_record_revision=1,
        reader_record_revision_digest=_digest("a"),
        reader_record_contract_id="plate_reader.annotated.v1",
        reader_record_content_digest=_digest("b"),
        reader_record_path="artifacts/sample_measurements/df.parquet",
        raw_design_id=f"design-{subject_id}",
        raw_assay_subject_id=None,
        subject_id=subject_id,
        observation_identity_field="position",
        observation_identity_values=("A1", "A2", "A3"),
        biological_replicate_identity_scopes=tuple(
            BiologicalReplicateIdentityScope(
                condition_value=f"condition-{condition}",
                biological_replicate_id=replicate_id,
            )
            for condition in ("baseline", "positive", "dose")
            for replicate_id in biological_replicate_ids
        ),
        binding_state="bound",
        binding_reason="exact_subject_alias_match",
    )
    return ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id="rt_lnrna_reader_evidence_bindings_v4",
        subject_binding_set_id="subject-bindings-v1",
        rows=(row,),
    )


def _policy() -> ReporterResponseObservationPolicy:
    return ReporterResponseObservationPolicy(
        policy_id="rt_lnrna_reporter_response_observation_policy.v3",
        pairing_kind="paired_by_design",
        within_acquisition_reduction_statistic="median",
        biological_replicate_uncertainty_policy=UncertaintyPolicy(
            minimum_biological_replicates=2,
            biological_replicate_reduction_statistic="median",
        ),
    )


def _measurement(
    observation_id: str,
    *,
    role: str,
    ratio: float,
    dose_uM: float | None,
    biological_replicate_id: str | None = "replicate-1",
):
    return ConditionMeasurement(
        observation_id=observation_id,
        condition_id=f"condition-{observation_id}",
        source_condition_value=f"condition-{observation_id}",
        role=role,
        dose_uM=dose_uM,
        biological_replicate_id=biological_replicate_id,
        acquisition_id="experiment-a",
        within_acquisition_observation_count=3,
        within_acquisition_reduction_statistic="median",
        rfp=ratio,
        od600=1.0,
        rfp_over_od600=ratio,
    )


def _profile(*, measurements=None, bindings=None, subject_id: str = "subject-a", uncertainties=None):
    rows = tuple(
        measurements
        or (
            _measurement("baseline", role="baseline", ratio=100.0, dose_uM=None),
            _measurement("positive", role="positive_control", ratio=120.0, dose_uM=None),
            _measurement("dose", role="dose", ratio=165.0, dose_uM=500.0),
        )
    )
    evidence_bindings = bindings or _bindings(subject_id=subject_id)
    evidence_binding = evidence_bindings.rows[0]
    return build_reporter_response_profile(
        profile_id=f"profile-{subject_id}",
        subject_id=subject_id,
        raw_design_id=evidence_binding.raw_design_id,
        raw_assay_subject_id=evidence_binding.raw_assay_subject_id,
        evidence_bindings=evidence_bindings,
        observation_policy=_policy(),
        reduction=EndpointReduction(recorded_time_h=10.0),
        dose_grid_uM=(500.0,),
        measurements=rows,
        pairing_policy=PairingPolicy(
            kind="paired_by_design",
            assignments=(
                ControlAssignment(
                    dose_observation_id="dose",
                    baseline_observation_ids=("baseline",),
                    positive_control_observation_ids=("positive",),
                ),
            ),
        ),
        dose_uncertainties=uncertainties
        or (
            DoseUncertainty(
                dose_uM=500.0,
                biological_replicate_count=1,
                normalized_reporter_response=NotEstimableMetricUncertainty(
                    estimate=3.25,
                    reason="below_minimum_biological_replicates",
                ),
                relative_od=NotEstimableMetricUncertainty(
                    estimate=1.0,
                    reason="below_minimum_biological_replicates",
                ),
            ),
        ),
        ineligibility_reasons=("preference_objective_not_defined",),
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_profile_keeps_acquisition_provenance_separate_from_replicate_identity() -> None:
    profile = _profile()

    assert {row.biological_replicate_id for row in profile.measurements} == {"replicate-1"}
    assert {row.acquisition_id for row in profile.measurements} == {"experiment-a"}
    assert profile.dose_uncertainties[0].biological_replicate_count == 1
    assert profile.dose_uncertainties[0].normalized_reporter_response.reason == ("below_minimum_biological_replicates")


def test_profile_round_trip_has_no_second_replicate_tier_or_objective_ontology() -> None:
    profile = _profile()
    payload = profile_to_dict(profile)

    assert profile_from_dict(payload, evidence_bindings=_bindings()) == profile
    keys = _all_keys(payload)
    assert not ({"independent_unit_id", "independent_block_id", "plate_id"} & keys)
    assert not ({"score", "scalar", "objective"} & keys)


def test_profile_v3_preserves_complete_reader_record_identity() -> None:
    payload = profile_to_dict(_profile())

    assert payload["contract_id"] == "rt_lnrna_reporter_response_profile.v3"
    assert payload["provenance"] == {
        "raw_design_id": "design-subject-a",
        "raw_assay_subject_id": None,
        "reader_experiment_id": "experiment-a",
        "reader_protocol_id": "plate_reader/single_reporter_screen",
        "reader_record_id": "sample_measurements/df",
        "reader_record_kind": "dataframe_artifact",
        "reader_record_revision": 1,
        "reader_record_revision_digest": _digest("a"),
        "reader_record_content_digest": _digest("b"),
        "reader_record_schema_version": 6,
        "reader_record_contract_id": "plate_reader.annotated.v1",
        "reader_record_path": "artifacts/sample_measurements/df.parquet",
        "evidence_binding_artifact_id": _bindings().artifact_id,
        "evidence_binding_artifact_digest": _bindings().artifact_digest,
    }


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("reader_protocol_id", "plate_reader/another_protocol"),
        ("reader_record_kind", "another_kind"),
        ("reader_record_path", "artifacts/another.parquet"),
    ],
)
def test_profile_parse_rejects_incomplete_reader_identity_rebinding(
    field_name: str,
    changed_value: str,
) -> None:
    payload = profile_to_dict(_profile())
    payload["provenance"][field_name] = changed_value

    with pytest.raises(ReporterResponseContractError, match="serialized provenance"):
        profile_from_dict(payload, evidence_bindings=_bindings())


def test_profile_parse_rejects_reader_identity_not_in_source_binding_artifact() -> None:
    payload = profile_to_dict(_profile())
    payload["provenance"]["raw_design_id"] = "unbound-design"

    with pytest.raises(ReporterResponseContractError, match="Reader identity"):
        profile_from_dict(payload, evidence_bindings=_bindings())


def test_serialized_profile_matches_checked_in_schema() -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    schema_path = repo_root / (
        "docs/studies/rt_lnrna_sponging_construct_triage/operations/contract/schemas/"
        "rt-lnrna-reporter-response-profile.schema.yaml"
    )
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(profile_to_dict(_profile()))

    legacy = profile_to_dict(_profile())
    legacy["contract_id"] = "rt_lnrna_reporter_response_profile.v1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(legacy)

    unconfined = profile_to_dict(_profile())
    unconfined["provenance"]["reader_record_path"] = "../outside.parquet"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(unconfined)


def test_position_is_only_observation_identity_not_replicate_identity() -> None:
    profile = _profile()

    assert profile.provenance.reader_record_id == "sample_measurements/df"
    assert all(row.biological_replicate_id != "position" for row in profile.measurements)
    assert all(row.within_acquisition_observation_count == 3 for row in profile.measurements)


def test_profile_rejects_a_replicate_id_moved_to_another_condition_scope() -> None:
    rows = list(_profile().measurements)
    rows[0] = replace(rows[0], biological_replicate_id="replicate-2")

    with pytest.raises(ReporterResponseContractError, match="condition-scoped biological-replicate identities"):
        _profile(measurements=rows)


def test_same_replicate_label_is_valid_across_conditions() -> None:
    profile = _profile()

    assert {row.biological_replicate_id for row in profile.measurements} == {"replicate-1"}


def test_unknown_replicate_identity_is_admissible_but_uncertainty_is_not_estimable() -> None:
    rows = tuple(replace(row, biological_replicate_id=None) for row in _profile().measurements)
    uncertainty = DoseUncertainty(
        dose_uM=500.0,
        biological_replicate_count=0,
        normalized_reporter_response=NotEstimableMetricUncertainty(
            estimate=3.25,
            reason="biological_replicate_identity_unknown",
        ),
        relative_od=NotEstimableMetricUncertainty(
            estimate=1.0,
            reason="biological_replicate_identity_unknown",
        ),
    )

    bindings = _bindings(biological_replicate_ids=())
    profile = _profile(measurements=rows, uncertainties=(uncertainty,), bindings=bindings)
    parsed = profile_from_dict(profile_to_dict(profile), evidence_bindings=bindings)

    assert parsed.measurements[0].biological_replicate_id is None
    assert parsed.dose_uncertainties[0].biological_replicate_count == 0


def test_profile_cannot_invent_replicate_identity_absent_from_reader_binding() -> None:
    with pytest.raises(ReporterResponseContractError, match="cannot invent"):
        _profile(bindings=_bindings(biological_replicate_ids=()))


def test_profile_rejects_replicate_identity_not_declared_by_reader_binding() -> None:
    with pytest.raises(ReporterResponseContractError, match="condition-scoped biological-replicate identities"):
        _profile(bindings=_bindings(biological_replicate_ids=("replicate-2",)))


def test_duplicate_dose_for_one_scoped_replicate_is_rejected() -> None:
    rows = list(_profile().measurements)
    rows.append(replace(rows[-1], observation_id="dose-copy"))

    with pytest.raises(ReporterResponseContractError, match="duplicate dose rows"):
        _profile(measurements=rows)


@pytest.mark.parametrize("minimum", [0, 1])
def test_uncertainty_requires_at_least_two_biological_replicates(minimum: int) -> None:
    with pytest.raises(ReporterResponseContractError, match="minimum_biological_replicates"):
        UncertaintyPolicy(
            minimum_biological_replicates=minimum,
            biological_replicate_reduction_statistic="median",
        )


def test_comparability_requires_exact_shared_contract_identity() -> None:
    left = _profile()
    right_bindings = _bindings(subject_id="subject-b", experiment_id="experiment-b")
    right_rows = tuple(
        replace(
            row,
            acquisition_id="experiment-b",
        )
        for row in left.measurements
    )
    right = _profile(measurements=right_rows, bindings=right_bindings, subject_id="subject-b")

    assert require_comparable_profiles((left, right)) == left.comparability_key


def test_comparability_distinguishes_biological_replicate_support_modes() -> None:
    def key_for(*, count: int, reason: str) -> str:
        uncertainty = DoseUncertainty(
            dose_uM=500.0,
            biological_replicate_count=count,
            normalized_reporter_response=NotEstimableMetricUncertainty(
                estimate=3.25,
                reason=reason,
            ),
            relative_od=NotEstimableMetricUncertainty(
                estimate=1.0,
                reason=reason,
            ),
        )
        return comparability_key(
            observation_policy_digest=_policy().digest,
            reduction=EndpointReduction(recorded_time_h=10.0),
            dose_grid_uM=(500.0,),
            dose_uncertainties=(uncertainty,),
        )

    keys = {
        key_for(count=0, reason="biological_replicate_identity_unknown"),
        key_for(count=1, reason="below_minimum_biological_replicates"),
        key_for(count=2, reason="insufficient_valid_resamples"),
    }

    assert len(keys) == 3


def test_comparability_rejects_conflicting_metric_identity_support_modes() -> None:
    uncertainty = DoseUncertainty(
        dose_uM=500.0,
        biological_replicate_count=1,
        normalized_reporter_response=NotEstimableMetricUncertainty(
            estimate=3.25,
            reason="biological_replicate_identity_unknown",
        ),
        relative_od=NotEstimableMetricUncertainty(
            estimate=1.0,
            reason="below_minimum_biological_replicates",
        ),
    )

    with pytest.raises(ReporterResponseContractError, match="identity-support modes"):
        comparability_key(
            observation_policy_digest=_policy().digest,
            reduction=EndpointReduction(recorded_time_h=10.0),
            dose_grid_uM=(500.0,),
            dose_uncertainties=(uncertainty,),
        )


def test_profile_pins_complete_neutral_temporal_policy_without_reader_runtime() -> None:
    payload = profile_to_dict(_profile())
    projection = payload["reduction"]["temporal_policy"]

    assert set(projection) == {"selection", "method", "output_space", "support", "digest"}
    assert projection["selection"] == {
        "kind": "endpoint",
        "time_basis": "absolute",
        "time_h": 10.0,
        "mode": "exact",
        "tolerance_h": 0.0,
    }
    assert projection["method"] == "identity"
    assert projection["support"] == {
        "boundary_support": "none",
        "minimum_observations": 1,
        "maximum_interior_gap_h": None,
        "positive_floor": None,
        "positive_value_scope": "selected_support",
        "censored_values": "reject",
    }
    assert payload["reduction"]["expected_cadence_h"] == pytest.approx(1.0 / 6.0)
    assert payload["reduction"]["ratio_reduction_order"] == "reduce_channels_then_ratio"
    assert projection["digest"] == _profile().reduction.temporal_policy.digest


@pytest.mark.parametrize(
    ("start_h", "end_h", "cadence_h", "minimum_observations"),
    (
        (4.0, 8.0, 0.5, 9),
        (4.0, 8.0, 1.0 / 6.0, 25),
        (0.0, 0.9, 0.3, 4),
    ),
)
def test_time_window_temporal_support_is_derived_from_declared_cadence(
    start_h: float,
    end_h: float,
    cadence_h: float,
    minimum_observations: int,
) -> None:
    reduction = TimeWindowReduction(
        recorded_start_time_h=start_h,
        recorded_end_time_h=end_h,
        summary_statistic="median",
        ratio_reduction_order="ratio_then_reduce",
        expected_cadence_h=cadence_h,
    )

    assert reduction.temporal_policy is not None
    assert reduction.temporal_policy.support.minimum_observations == minimum_observations
    assert reduction.temporal_policy.support.maximum_interior_gap_h == cadence_h


def test_temporal_policy_changes_profile_comparability() -> None:
    left = _profile()
    changed_reduction = EndpointReduction(recorded_time_h=11.0)
    right = replace(left, profile_id="changed-temporal-policy", reduction=changed_reduction)

    with pytest.raises(ReporterResponseContractError, match="comparability keys"):
        require_comparable_profiles((left, right))
