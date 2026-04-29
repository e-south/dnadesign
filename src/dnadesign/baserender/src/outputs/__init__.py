"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/outputs/__init__.py

Output writer public compatibility surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..render import render_record
from .images import _figure_rgba, _render_record_grid_figure_local, write_images
from .names import _safe_stem, _unique_stem
from .video import (
    _apply_fixed_content_radius,
    _apply_sequence_rows_content_envelope,
    _apply_sequence_rows_extra_bottom_padding,
    _content_bounds_rgba,
    _even_ceil,
    _letterbox_rgba,
    _pause_frames,
    _rendered_content_top_norm_for_video_frame,
    _scale_rgba_to_fit,
    _scale_rgba_to_width,
    _scaled_dimensions_to_fit,
    _sequence_rows_actual_content_bounds_px,
    _sequence_rows_content_envelope_norms,
    _sequence_rows_content_extents_px,
    _sequence_rows_layout_context,
    _sequence_rows_required_extra_bottom_padding_px,
    _target_frame_size,
    _trim_white_border_rgba,
    _union_centered_content_bounds,
    effective_video_frames_per_record,
    planned_video_frame_count,
    write_video,
)

__all__ = [
    "_apply_fixed_content_radius",
    "_apply_sequence_rows_content_envelope",
    "_apply_sequence_rows_extra_bottom_padding",
    "_content_bounds_rgba",
    "_even_ceil",
    "_figure_rgba",
    "_letterbox_rgba",
    "_pause_frames",
    "_render_record_grid_figure_local",
    "_rendered_content_top_norm_for_video_frame",
    "_safe_stem",
    "_scale_rgba_to_fit",
    "_scale_rgba_to_width",
    "_scaled_dimensions_to_fit",
    "_sequence_rows_actual_content_bounds_px",
    "_sequence_rows_content_envelope_norms",
    "_sequence_rows_content_extents_px",
    "_sequence_rows_layout_context",
    "_sequence_rows_required_extra_bottom_padding_px",
    "_target_frame_size",
    "_trim_white_border_rgba",
    "_union_centered_content_bounds",
    "effective_video_frames_per_record",
    "planned_video_frame_count",
    "_unique_stem",
    "render_record",
    "write_images",
    "write_video",
]
