"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_window_geometry.py

Direct tests for construct window-geometry normalization helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

import pytest

from dnadesign.construct.src.config import WindowConfig
from dnadesign.construct.src.errors import ValidationError
from dnadesign.construct.src.runtime import _normalize_window_geometry, _ResolvedPart


def _resolved_part(
    *,
    sequence: str = "ACGT",
    orientation: Literal["forward", "reverse_complement"] = "forward",
    realized_start: int = 8,
) -> _ResolvedPart:
    return _ResolvedPart(
        name="anchor",
        role="anchor",
        kind="replace",
        sequence_source="input_field",
        sequence_field="sequence",
        orientation=orientation,
        start=0,
        end=len(sequence),
        sequence=sequence,
        realized_start=realized_start,
        realized_end=realized_start + len(sequence),
    )


def test_normalize_window_geometry_supports_fixed_total_symmetric_linear() -> None:
    geometry = _normalize_window_geometry(
        full_construct_length=16,
        template_circular=False,
        focal=_resolved_part(),
        window=WindowConfig(
            semantics="fixed_total",
            reference="center",
            direction="symmetric",
            size_bp=8,
            offset_bp=0,
        ),
    )

    assert (geometry.start_raw, geometry.end_raw) == (6, 14)
    assert (geometry.start, geometry.end, geometry.span_bp) == (6, 14, 8)


@pytest.mark.parametrize(
    ("orientation", "direction", "expected"),
    [
        ("forward", "five_prime", (4, 9)),
        ("forward", "three_prime", (8, 13)),
        ("reverse_complement", "three_prime", (4, 9)),
        ("reverse_complement", "five_prime", (8, 13)),
    ],
)
def test_normalize_window_geometry_supports_directional_fixed_total_windows(
    *,
    orientation: Literal["forward", "reverse_complement"],
    direction: Literal["five_prime", "three_prime"],
    expected: tuple[int, int],
) -> None:
    geometry = _normalize_window_geometry(
        full_construct_length=16,
        template_circular=False,
        focal=_resolved_part(orientation=orientation),
        window=WindowConfig(
            semantics="fixed_total",
            reference="start",
            direction=direction,
            size_bp=5,
            offset_bp=0,
        ),
    )

    assert (geometry.start_raw, geometry.end_raw) == expected
    assert geometry.span_bp == 5


def test_normalize_window_geometry_supports_anchor_plus_context_reverse_orientation() -> None:
    geometry = _normalize_window_geometry(
        full_construct_length=20,
        template_circular=False,
        focal=_resolved_part(sequence="ACGT", orientation="reverse_complement"),
        window=WindowConfig(
            semantics="anchor_plus_context",
            upstream_bp=2,
            downstream_bp=3,
        ),
    )

    assert (geometry.start_raw, geometry.end_raw) == (5, 14)
    assert (geometry.start, geometry.end, geometry.span_bp) == (5, 14, 9)


def test_normalize_window_geometry_wraps_coordinates_for_circular_templates() -> None:
    geometry = _normalize_window_geometry(
        full_construct_length=8,
        template_circular=True,
        focal=_resolved_part(sequence="GG", realized_start=6),
        window=WindowConfig(
            semantics="fixed_total",
            reference="center",
            direction="symmetric",
            size_bp=6,
            offset_bp=0,
        ),
    )

    assert (geometry.start_raw, geometry.end_raw) == (4, 10)
    assert (geometry.start, geometry.end, geometry.span_bp) == (4, 2, 6)


def test_normalize_window_geometry_rejects_linear_windows_outside_bounds() -> None:
    with pytest.raises(ValidationError, match="extends beyond the linear construct boundaries"):
        _normalize_window_geometry(
            full_construct_length=16,
            template_circular=False,
            focal=_resolved_part(realized_start=1),
            window=WindowConfig(
                semantics="fixed_total",
                reference="center",
                direction="symmetric",
                size_bp=8,
                offset_bp=0,
            ),
        )


def test_normalize_window_geometry_rejects_linear_windows_past_right_boundary() -> None:
    with pytest.raises(ValidationError, match="extends beyond the linear construct boundaries"):
        _normalize_window_geometry(
            full_construct_length=16,
            template_circular=False,
            focal=_resolved_part(realized_start=12),
            window=WindowConfig(
                semantics="fixed_total",
                reference="center",
                direction="symmetric",
                size_bp=8,
                offset_bp=0,
            ),
        )


def test_normalize_window_geometry_rejects_fixed_total_window_shorter_than_focal_part() -> None:
    with pytest.raises(ValidationError, match="exceeds fixed_total window size_bp=5"):
        _normalize_window_geometry(
            full_construct_length=16,
            template_circular=False,
            focal=_resolved_part(sequence="ACGTAA"),
            window=WindowConfig(
                semantics="fixed_total",
                reference="center",
                direction="symmetric",
                size_bp=5,
                offset_bp=0,
            ),
        )


def test_normalize_window_geometry_rejects_anchor_plus_context_windows_longer_than_construct() -> None:
    with pytest.raises(ValidationError, match="Requested anchor_plus_context window length 17 exceeds"):
        _normalize_window_geometry(
            full_construct_length=16,
            template_circular=False,
            focal=_resolved_part(sequence="ACGT"),
            window=WindowConfig(
                semantics="anchor_plus_context",
                upstream_bp=6,
                downstream_bp=7,
            ),
        )
