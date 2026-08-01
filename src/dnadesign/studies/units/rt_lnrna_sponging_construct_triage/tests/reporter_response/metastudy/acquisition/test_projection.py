"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/acquisition/test_projection.py

Tests descriptive acquisition projections and leave-one-acquisition-out summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    TimeWindowReduction,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    ACQUISITION_PROJECTION_CONTRACT_ID,
    MetastudyContractError,
    build_acquisition_projection,
)

from .._builders import (
    HIGH_ANCHOR,
    KINETIC_IDS,
    _evidence,
)


def test_acquisition_projection_is_descriptive_and_exposes_leave_one_acquisition_out() -> None:
    projection = build_acquisition_projection(_evidence(), selected_reduction=(6.0, 10.0))

    assert projection.contract_id == ACQUISITION_PROJECTION_CONTRACT_ID
    assert projection.selected_reduction == (6.0, 10.0)
    assert {row.reduction_id for row in projection.coordinates} == {"window-6-10h"}
    high = next(
        row
        for row in projection.coordinates
        if row.subject_id == HIGH_ANCHOR and row.reduction_id == "window-6-10h" and row.dose_uM == 500.0
    )
    assert high.acquisition_ids == KINETIC_IDS
    assert all(row.declared_biological_replicate_ids == () for row in high.contributions)
    assert high.normalized_reporter_response.method == "median_across_acquisitions"
    assert high.normalized_reporter_response.acquisition_count == 8
    assert len(high.normalized_reporter_response.leave_one_acquisition_out_estimates) == 8
    assert not hasattr(high.normalized_reporter_response, "interval_lower")


def test_acquisition_projection_keeps_single_acquisition_descriptive() -> None:
    source = next(
        row
        for row in _evidence()
        if row.profile.subject_id == HIGH_ANCHOR
        and row.profile.provenance.reader_experiment_id == KINETIC_IDS[0]
        and isinstance(row.profile.reduction, TimeWindowReduction)
        and row.profile.reduction.recorded_start_time_h == 6.0
    )
    projection = build_acquisition_projection((source,), selected_reduction=(6.0, 10.0))

    assert len(projection.coordinates) == 1
    coordinate = projection.coordinates[0]
    assert coordinate.acquisition_ids == (KINETIC_IDS[0],)
    assert coordinate.normalized_reporter_response.acquisition_count == 1
    assert coordinate.normalized_reporter_response.leave_one_acquisition_out_estimates == ()


def test_acquisition_projection_preserves_raw_measurements_without_inventing_normalized_values() -> None:
    projection = build_acquisition_projection(
        _evidence(reference_normalized=False),
        selected_reduction=(6.0, 10.0),
    )

    high = next(
        row
        for row in projection.coordinates
        if row.subject_id == HIGH_ANCHOR and row.reduction_id == "window-6-10h" and row.dose_uM == 500.0
    )
    assert high.rfp.acquisition_count == 8
    assert high.od600.acquisition_count == 8
    assert high.rfp_over_od600.acquisition_count == 8
    assert high.normalized_reporter_response is None
    assert high.relative_od is None
    assert all(row.normalized_reporter_response is None and row.relative_od is None for row in high.contributions)


def test_acquisition_projection_keeps_raw_and_normalized_metric_spaces_distinct() -> None:
    normalized = _evidence()
    raw = _evidence(reference_normalized=False)
    mixed = tuple(
        raw_row if raw_row.profile.provenance.reader_experiment_id == KINETIC_IDS[0] else normalized_row
        for normalized_row, raw_row in zip(normalized, raw, strict=True)
    )

    projection = build_acquisition_projection(mixed, selected_reduction=(6.0, 10.0))
    high = tuple(
        row
        for row in projection.coordinates
        if row.subject_id == HIGH_ANCHOR and row.reduction_id == "window-6-10h" and row.dose_uM == 500.0
    )
    assert len(high) == 2
    assert {row.rfp is not None for row in high} == {False, True}


def test_acquisition_projection_rejects_duplicate_acquisition_coordinate() -> None:
    source = _evidence()[0]
    with pytest.raises(MetastudyContractError, match="duplicate acquisition"):
        build_acquisition_projection((source, source), selected_reduction=(4.0, 8.0))
