"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/windows.py

Window geometry contracts for emitted construct sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.config import WindowConfig
from ..contracts.errors import ValidationError
from .parts import RealizedPart


@dataclass(frozen=True)
class WindowGeometry:
    start_raw: int
    end_raw: int
    start: int
    end: int
    span_bp: int


def normalize_window_geometry(
    *,
    full_construct_length: int,
    template_circular: bool,
    focal: RealizedPart,
    window: WindowConfig,
) -> WindowGeometry:
    start_raw, end_raw = _window_raw_bounds(
        full_construct_length=full_construct_length,
        focal=focal,
        window=window,
    )
    span_bp = end_raw - start_raw
    if span_bp > full_construct_length:
        raise ValidationError(
            f"Requested window span {span_bp} exceeds realized construct length {full_construct_length}."
        )
    if template_circular:
        start = start_raw % full_construct_length
        end = (start + span_bp) % full_construct_length
        return WindowGeometry(
            start_raw=start_raw,
            end_raw=end_raw,
            start=start,
            end=end,
            span_bp=span_bp,
        )
    if start_raw < 0 or end_raw > full_construct_length:
        raise ValidationError(
            "Requested window extends beyond the linear construct boundaries. "
            "Adjust the window settings or choose a circular template."
        )
    return WindowGeometry(
        start_raw=start_raw,
        end_raw=end_raw,
        start=start_raw,
        end=end_raw,
        span_bp=span_bp,
    )


def _window_raw_bounds(
    *,
    full_construct_length: int,
    focal: RealizedPart,
    window: WindowConfig,
) -> tuple[int, int]:
    if window.semantics == "fixed_total":
        window_bp = int(window.size_bp)
        if window_bp > full_construct_length:
            raise ValidationError(
                f"Requested fixed_total window size_bp={window_bp} exceeds realized construct length "
                f"{full_construct_length}."
            )
        if len(focal.sequence) > window_bp:
            raise ValidationError(
                f"Focal part '{focal.name}' length {len(focal.sequence)} exceeds "
                f"fixed_total window size_bp={window_bp}. "
                "Choose a larger fixed_total window or use anchor_plus_context semantics."
            )
        point = _window_reference_index(focal, reference=window.reference)
        offset_bp = int(window.offset_bp)
        if window.direction == "symmetric":
            start_raw = point - (window_bp // 2) + offset_bp
            return start_raw, start_raw + window_bp

        step = _orientation_step(orientation=focal.orientation, direction=window.direction)
        if step > 0:
            start_raw = point + offset_bp
            return start_raw, start_raw + window_bp

        end_raw = point + 1 + offset_bp
        return end_raw - window_bp, end_raw

    upstream_bp = int(window.upstream_bp)
    downstream_bp = int(window.downstream_bp)
    window_bp = len(focal.sequence) + upstream_bp + downstream_bp
    if window_bp > full_construct_length:
        raise ValidationError(
            f"Requested anchor_plus_context window length {window_bp} exceeds realized construct length "
            f"{full_construct_length}."
        )
    if focal.orientation == "forward":
        return focal.realized_start - upstream_bp, focal.realized_end + downstream_bp
    return focal.realized_start - downstream_bp, focal.realized_end + upstream_bp


def _window_reference_index(part: RealizedPart, *, reference: str) -> int:
    if reference == "start":
        return part.realized_start
    if reference == "end":
        return part.realized_end - 1
    return part.realized_start + (len(part.sequence) // 2)


def _orientation_step(*, orientation: str, direction: str) -> int:
    if direction == "five_prime":
        return -1 if orientation == "forward" else 1
    if direction == "three_prime":
        return 1 if orientation == "forward" else -1
    raise ValidationError(f"Unsupported window direction '{direction}'.")
