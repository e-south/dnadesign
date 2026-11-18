"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/video.py

Validate video settings shared by parsed jobs and already-typed API callers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from math import isfinite
from numbers import Real
from typing import Mapping, Protocol

from ..core import SchemaError, ensure


class VideoSettings(Protocol):
    fps: int
    frames_per_record: int
    pauses: Mapping[str, float]
    width_px: int | None
    height_px: int | None
    aspect_ratio: float | None
    total_duration: float | None
    content_fit: str
    title_text: str | None
    title_font_size: int | None
    title_align: str
    encode_timeout_s: float | None


def _positive_integer(value: object, field: str, minimum: int = 1) -> None:
    ensure(type(value) is int and value >= minimum, f"{field} must be an integer >= {minimum}", SchemaError)


def _finite_number(value: object, field: str, *, allow_zero: bool = False) -> None:
    ensure(
        isinstance(value, Real) and not isinstance(value, bool) and isfinite(value),
        f"{field} must be finite and numeric",
        SchemaError,
    )
    ensure(value >= 0 if allow_zero else value > 0, f"{field} must be {'>= 0' if allow_zero else '> 0'}", SchemaError)


def validate_video_settings(output: VideoSettings, *, field: str) -> None:
    """Admit timing, sizing and presentation settings before any renderer or input I/O."""
    _positive_integer(output.fps, f"{field}.fps")
    _positive_integer(output.frames_per_record, f"{field}.frames_per_record")
    ensure(isinstance(output.pauses, Mapping), f"{field}.pauses must be a mapping", SchemaError)
    for key, duration in output.pauses.items():
        ensure(isinstance(key, str) and bool(key), f"{field}.pauses keys must be nonempty record IDs", SchemaError)
        _finite_number(duration, f"{field}.pauses.{key}", allow_zero=True)
    for name in ("width_px", "height_px", "title_font_size"):
        value = getattr(output, name)
        if value is not None:
            _positive_integer(value, f"{field}.{name}", 6 if name == "title_font_size" else 1)
    for name in ("aspect_ratio", "total_duration", "encode_timeout_s"):
        value = getattr(output, name)
        if value is not None:
            _finite_number(value, f"{field}.{name}")
    if output.width_px is not None and output.height_px is not None and output.aspect_ratio is not None:
        ensure(
            abs(output.width_px / output.height_px - output.aspect_ratio) <= 1.0e-6,
            f"{field}.aspect conflicts with width_px/height_px",
            SchemaError,
        )
    ensure(
        output.content_fit in ("native", "fill_width", "fill_width_per_frame"),
        f"{field}.content_fit is unsupported",
        SchemaError,
    )
    ensure(
        output.title_text is None or isinstance(output.title_text, str), f"{field}.title_text must be text", SchemaError
    )
    ensure(output.title_align in ("left", "center", "right"), f"{field}.title_align is unsupported", SchemaError)
