"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_realization_slots.py

Unit contracts for construct named-slot realization metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.realization.slots import (
    assembly_mode,
    input_slot_sequence_length,
    oriented_slot_span_records,
    require_required_slot_bounds,
    slot_span_records,
)


@dataclass(frozen=True)
class _Part:
    name: str
    role: str
    kind: str
    sequence_source: str
    sequence_field: str | None
    orientation: str
    start: int
    end: int
    sequence: str
    realized_start: int
    realized_end: int


@dataclass(frozen=True)
class _Window:
    semantics: str
    reference: str
    direction: str
    size_bp: int | None = None
    upstream_bp: int | None = None
    downstream_bp: int | None = None


def _parts() -> list[_Part]:
    return [
        _Part(
            name="lnrna",
            role="lnrna_cassette",
            kind="replace",
            sequence_source="input_field",
            sequence_field="candidate__lnrna_sequence",
            orientation="forward",
            start=4,
            end=8,
            sequence="GG",
            realized_start=4,
            realized_end=6,
        ),
        _Part(
            name="rt_cds",
            role="rt_cds",
            kind="replace",
            sequence_source="input_field",
            sequence_field="candidate__rt_cds_sequence",
            orientation="forward",
            start=12,
            end=16,
            sequence="AATTAA",
            realized_start=10,
            realized_end=16,
        ),
    ]


def test_slot_records_preserve_forward_and_reverse_complement_coordinates() -> None:
    forward = slot_span_records(
        parts=_parts(),
        output_length=16,
        full_construct_length=16,
        window_start=0,
        mode="full_construct",
    )

    assert [(slot["slot_id"], slot["start"], slot["end"]) for slot in forward] == [
        ("lnrna", 4, 6),
        ("rt_cds", 10, 16),
    ]

    reverse = oriented_slot_span_records(
        forward_slots=forward,
        output_length=16,
        orientation="reverse_complement",
    )

    assert [(slot["slot_id"], slot["start"], slot["end"]) for slot in reverse] == [
        ("lnrna", 10, 12),
        ("rt_cds", 0, 6),
    ]
    assert [(slot["slot_id"], slot["forward_start"], slot["forward_end"]) for slot in reverse] == [
        ("lnrna", 4, 6),
        ("rt_cds", 10, 16),
    ]


def test_required_slot_check_rejects_clipped_slot_with_diagnostic_window() -> None:
    clipped = slot_span_records(
        parts=_parts(),
        output_length=8,
        full_construct_length=16,
        window_start=4,
        mode="window",
    )

    assert clipped[0]["start"] == 0
    assert clipped[1]["start"] is None

    with pytest.raises(ValidationError, match="required slot 'rt_cds'.*fixed_total"):
        require_required_slot_bounds(
            row_id="candidate-1",
            required_slots=["lnrna", "rt_cds"],
            slot_spans=clipped,
            mode="window",
            window=_Window(
                semantics="fixed_total",
                reference="start",
                direction="three_prime",
                size_bp=8,
            ),
        )


def test_assembly_mode_and_input_length_are_driven_by_input_slots() -> None:
    parts = [
        *_parts(),
        _Part(
            name="constant_spacer",
            role="spacer",
            kind="insert",
            sequence_source="literal",
            sequence_field=None,
            orientation="forward",
            start=8,
            end=8,
            sequence="CCCC",
            realized_start=6,
            realized_end=10,
        ),
    ]

    assert assembly_mode(parts) == "multi_slot"
    assert input_slot_sequence_length(parts) == 8
