"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/test_temporal_conformance.py

Byte-identical Reader conformance vectors exercised through the live study kernel.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    EndpointReduction,
    ReporterResponseContractError,
    TemporalPolicyProjection,
    TemporalSelectedRow,
    TimeWindowReduction,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.condition_ontology import (
    DEFAULT_CONDITION_ONTOLOGY,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize import temporal
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.temporal import (
    endpoint_temporal_policy_projection,
    reduce_temporal_input_trace,
)

_FIXTURE_SHA256 = "2f2c8f7a9d328faffdaaa44f4525e0b985eb71d6fcca2279a64c98bc6af2bc87"  # pragma: allowlist secret
_SCHEMA = "reader.temporal_reduction_conformance.v1"
_CONTRACT = "reader.domains.time_series.temporal_reduction.v1"


def _fixture_path() -> Path:
    return Path(__file__).parent / "fixtures/temporal_reduction_conformance_v1.json"


def _vector() -> dict[str, object]:
    raw = _fixture_path().read_bytes()
    if hashlib.sha256(raw).hexdigest() != _FIXTURE_SHA256:
        raise ReporterResponseContractError("Reader temporal conformance fixture bytes changed")
    assert b"technical" not in raw.lower()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"schema", "contract", "time_unit", "cases"}:
        raise ReporterResponseContractError("Reader temporal conformance fixture fields changed")
    if payload["schema"] != _SCHEMA or payload["contract"] != _CONTRACT or payload["time_unit"] != "hour":
        raise ReporterResponseContractError("Reader temporal conformance fixture contract identity changed")
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise ReporterResponseContractError("Reader temporal conformance fixture requires cases")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "kind", "case_payload_digest", "payload", "expected"}:
            raise ReporterResponseContractError("Reader temporal conformance case fields changed")
        observed = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(case["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        )
        if observed != case["case_payload_digest"]:
            raise ReporterResponseContractError("Reader temporal conformance case payload digest changed")
    return payload


def _case(vector: dict[str, object], case_id: str) -> dict[str, object]:
    return next(case for case in vector["cases"] if case["id"] == case_id)


def _trace_rows(payload: dict[str, object]) -> tuple[TemporalSelectedRow, ...]:
    times = payload["times_h"]
    values = payload["values"]
    clipped = payload.get("policy_clipped", [False] * len(times))
    overflow = payload.get("instrument_overflow", [False] * len(times))
    return tuple(
        TemporalSelectedRow(
            observation_identity="observation-1",
            time_h=float(time_h),
            value=float(value),
            value_policy_clipped=bool(clipped[index]),
            value_instrument_overflow=bool(overflow[index]),
        )
        for index, (time_h, value) in enumerate(zip(times, values, strict=True))
    )


def test_ratio_then_reduce_vector_exercises_actual_condition_summary() -> None:
    case = _case(_vector(), "ratio_then_reduce_10_minute_4_8h_observed_median")
    payload = case["payload"]
    expanded: list[dict[str, object]] = []
    for well in payload["wells"]:
        for time_h, signal, reference in well["rows"]:
            for channel, value in (
                ("RFP", signal),
                ("OD600", reference),
                ("RFP/OD600", signal / reference),
            ):
                expanded.append(
                    {
                        "position": well["well_id"],
                        "time": time_h,
                        "channel": channel,
                        "value": value,
                        "value_policy_clipped": False,
                        "value_instrument_overflow": False,
                        "value_bound_kind": "exact",
                    }
                )
    reduction = TimeWindowReduction(
        recorded_start_time_h=4.0,
        recorded_end_time_h=8.0,
        summary_statistic="median",
        ratio_reduction_order="ratio_then_reduce",
    )
    summary = temporal._condition_summary(
        pd.DataFrame(expanded),
        DEFAULT_CONDITION_ONTOLOGY,
        reduction=reduction,
        protocol=DEFAULT_PROTOCOL,
    )
    assert summary is not None
    assert summary[2] == case["expected"]["observation_median"] == 26.0
    forbidden = case["expected"]["alternative_reduce_then_ratio"]["observation_median"]
    assert summary[0] / summary[1] == forbidden == 46.0
    assert summary[2] != forbidden


def test_historical_endpoint_scalar_remains_distinct_from_shared_ratio_then_reduce() -> None:
    expanded: list[dict[str, object]] = []
    for position, reporter, normalizer in (("A1", 10.0, 1.0), ("A2", 40.0, 2.0), ("A3", 100.0, 1.0)):
        for channel, value in (
            ("RFP", reporter),
            ("OD600", normalizer),
            ("RFP/OD600", reporter / normalizer),
        ):
            expanded.append(
                {
                    "position": position,
                    "time": 10.0,
                    "channel": channel,
                    "value": value,
                    "value_policy_clipped": False,
                    "value_instrument_overflow": False,
                    "value_bound_kind": "exact",
                }
            )

    summary = temporal._condition_summary(
        pd.DataFrame(expanded),
        DEFAULT_CONDITION_ONTOLOGY,
        reduction=EndpointReduction(recorded_time_h=10.0),
        protocol=DEFAULT_PROTOCOL,
    )

    assert summary is not None
    shared_ratio_then_reduce = 20.0
    assert summary[2] == summary[0] / summary[1] == 40.0
    assert summary[2] != shared_ratio_then_reduce


@pytest.mark.parametrize(
    "case_id",
    (
        "exact_endpoint",
        "observed_boundaries_within_absolute_tolerance",
    ),
)
def test_supported_reader_vectors_round_trip_and_match_expected_value(case_id: str) -> None:
    case = _case(_vector(), case_id)
    payload = case["payload"]
    policy = TemporalPolicyProjection.from_reader_mapping(payload["temporal_reduction"])
    assert policy.to_reader_mapping() == payload["temporal_reduction"]
    assert (
        reduce_temporal_input_trace(_trace_rows(payload), policy=policy, within_acquisition_statistic="median")
        == case["expected"]["value"]
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "observed_boundaries_beyond_absolute_tolerance",
        "interior_gap_beyond_absolute_tolerance",
        "policy_clipped_observation_rejected",
        "instrument_overflow_observation_rejected",
    ),
)
def test_reader_negative_vectors_are_rejected_by_shared_study_kernel(case_id: str) -> None:
    case = _case(_vector(), case_id)
    payload = case["payload"]
    policy = TemporalPolicyProjection.from_reader_mapping(payload["temporal_reduction"])
    with pytest.raises(ReporterResponseContractError):
        reduce_temporal_input_trace(_trace_rows(payload), policy=policy, within_acquisition_statistic="median")


def test_reader_positive_floor_equality_vector_is_rejected_by_narrow_linear_projection() -> None:
    case = _case(_vector(), "positive_floor_equality_rejected")
    payload = case["payload"]
    policy = endpoint_temporal_policy_projection(time_h=1.0)
    policy = replace(policy, support=replace(policy.support, positive_floor=1e-12))
    with pytest.raises(ReporterResponseContractError, match="violates positive floor"):
        reduce_temporal_input_trace(_trace_rows(payload), policy=policy, within_acquisition_statistic="median")


def test_reader_fixture_rejects_contract_payload_or_digest_mutation() -> None:
    vector = _vector()
    changed_contract = deepcopy(vector)
    changed_contract["contract"] = "forged"
    with pytest.raises(ReporterResponseContractError, match="contract identity changed"):
        _validate_mutated_vector(changed_contract)

    changed_payload = deepcopy(vector)
    changed_payload["cases"][0]["payload"]["operation_order"] = "forged"
    with pytest.raises(ReporterResponseContractError, match="case payload digest changed"):
        _validate_mutated_vector(changed_payload)


def _validate_mutated_vector(payload: dict[str, object]) -> None:
    if payload["schema"] != _SCHEMA or payload["contract"] != _CONTRACT:
        raise ReporterResponseContractError("Reader temporal conformance fixture contract identity changed")
    for case in payload["cases"]:
        observed = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(case["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        )
        if observed != case["case_payload_digest"]:
            raise ReporterResponseContractError("Reader temporal conformance case payload digest changed")
