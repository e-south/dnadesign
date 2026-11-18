"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/sequence_layout.py

Strict physical layout settings shared by sequence API and job admission.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Mapping

from ..core import LayoutError


def _measure(value: object, name: str, *, positive: bool) -> None:
    valid = not isinstance(value, bool) and isinstance(value, (int, float))
    try:
        valid = valid and isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise LayoutError(f"layout.{name} must be a finite number in points")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise LayoutError(f"layout.{name} must be {qualifier}, in points")


@dataclass(frozen=True)
class SequenceLayout:
    """Preserve type and pitch; canvas coordinates start at the top left.

    sequence_origin_pt is the left edge of base column zero on the forward
    strand's grid-reference line. It is not the baseline of a particular glyph.
    Content extents reserve space about the midpoint between the strand lines.
    """

    base_pitch_pt: float | None = None
    canvas_size_pt: tuple[float, float] | None = None
    sequence_origin_pt: tuple[float, float] | None = None
    top_extent_pt: float | None = None
    bottom_extent_pt: float | None = None
    extra_bottom_padding_pt: float | None = None

    def __post_init__(self) -> None:
        for name in ("base_pitch_pt", "top_extent_pt", "bottom_extent_pt", "extra_bottom_padding_pt"):
            value = getattr(self, name)
            if value is not None:
                _measure(value, name, positive=name != "extra_bottom_padding_pt")
        for name in ("canvas_size_pt", "sequence_origin_pt"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise LayoutError(f"layout.{name} must contain two numbers in points")
            for index, item in enumerate(value):
                _measure(item, f"{name}[{index}]", positive=name == "canvas_size_pt")
            object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SequenceLayout:
        if not isinstance(value, Mapping):
            raise LayoutError("layout must be a mapping with explicit point units")
        allowed = {field.name for field in fields(cls)}
        unknown = [key for key in value if key not in allowed]
        if unknown:
            raise LayoutError(f"layout received unknown fields: {unknown!r}; allowed fields: {sorted(allowed)}")
        return cls(**value)

    def to_mapping(self) -> dict[str, object]:
        return {
            field.name: list(value) if isinstance(value, tuple) else value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


def sequence_layout_from_options(options: Mapping[str, object] | None) -> SequenceLayout | None:
    if options is None or "layout" not in options:
        return None
    value = options["layout"]
    return value if isinstance(value, SequenceLayout) else SequenceLayout.from_mapping(value)


def normalize_sequence_renderer_options(options: Mapping[str, object]) -> dict[str, object]:
    result = dict(options)
    layout = sequence_layout_from_options(options)
    if layout is not None:
        result["layout"] = layout.to_mapping()
    return result
