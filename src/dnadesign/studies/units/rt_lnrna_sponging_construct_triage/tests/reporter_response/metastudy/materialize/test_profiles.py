"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/materialize/test_profiles.py

Owner-aligned materialize contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    TimeWindowReduction,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    MaterializationOmission,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import profile_digest
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.condition_ontology import (
    DEFAULT_CONDITION_ONTOLOGY,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize import temporal
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize.service import (
    materialize_record_evidence,
)

from ._support import (
    _SUBJECT_ID,
    _ontology,
    _policy,
    _rehash,
    _source_closed_inputs,
)


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
    expected = temporal._growth_phase_strata(
        frame.loc[~invalid],
        reduction=reduction,
        ontology=_ontology(),
        protocol=DEFAULT_PROTOCOL,
    )
    frame.loc[invalid, "value"] = "not-numeric"

    observed = temporal._growth_phase_strata(
        frame,
        reduction=reduction,
        ontology=_ontology(),
        protocol=DEFAULT_PROTOCOL,
    )

    assert observed == expected


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
