"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/materialize/test_temporal.py

Owner-aligned materialize contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    TemporalSelectedRow,
    TimeWindowReduction,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import DEFAULT_PROTOCOL
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize import temporal
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.materialize.service import (
    materialize_record_evidence,
)

from ._support import (
    _SUBJECT_ID,
    _ontology,
    _policy,
    _reader_reduce_trace_rows,
    _rehash,
    _source_closed_inputs,
)


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
        temporal._condition_summary(
            pd.DataFrame(rows),
            _ontology(),
            reduction=reduction,
            protocol=DEFAULT_PROTOCOL,
        )
        is None
    )


def test_live_reader_matches_source_bound_temporal_conformance_probe(tmp_path: Path) -> None:
    phd_roots = [
        parent for parent in Path(__file__).resolve().parents if (parent / "reader/src/reader_workbench").is_dir()
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
