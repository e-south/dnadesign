"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/sequence_rows_metadata.py

Allocation-free validation for sequence-row renderer-owned metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.colors as mcolors

from ..config import Style
from ..core import Record, RenderingError

_ROWS = ("primary", "complement")


def _validate_color(value: object, *, path: str) -> None:
    if not isinstance(value, str) or not value.strip() or not mcolors.is_color_like(value.strip()):
        raise RenderingError(f"{path} must be a valid color")


def _validated_indices(value: object, *, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RenderingError(f"{path} must contain integer indices")
    try:
        return tuple(int(index) for index in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RenderingError(f"{path} must contain integer indices") from exc


def _row_indices(meta: Mapping[str, object], key: str, *, rows: Sequence[str]) -> dict[str, tuple[int, ...]]:
    raw = meta.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        return {}  # The renderer ignores non-mapping row metadata.
    # The renderer consumes only these two keys; extension metadata remains opaque.
    return {
        row_id: _validated_indices(raw[row_id], path=f"record.meta.{key}.{row_id}") for row_id in rows if row_id in raw
    }


def _active(indices: Sequence[int], sequence_length: int) -> set[int]:
    return {index for index in indices if 0 <= index < sequence_length}


def _validate_highlight_colors(
    meta: Mapping[str, object],
    *,
    highlights: Mapping[str, tuple[int, ...]],
    sequence_length: int,
    rows: Sequence[str],
) -> None:
    fallback_colors = meta.get("base_highlight_color")
    indexed_raw = meta.get("base_highlight_colors")

    for row_id in rows:
        indexed_colors: dict[int, object] = {}
        if isinstance(indexed_raw, Mapping) and row_id in indexed_raw:
            row_colors = indexed_raw[row_id]
            if not isinstance(row_colors, Mapping):
                raise RenderingError(f"record.meta.base_highlight_colors.{row_id} must be a mapping")
            for index, color in row_colors.items():
                try:
                    parsed_index = int(index)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RenderingError(
                        f"record.meta.base_highlight_colors.{row_id} keys must be integer indices"
                    ) from exc
                indexed_colors[parsed_index] = color
        active = _active(highlights.get(row_id, ()), sequence_length)
        if indexed_colors and active:
            for index in active.intersection(indexed_colors):
                _validate_color(indexed_colors[index], path=f"record.meta.base_highlight_colors.{row_id}.{index}")
        if isinstance(fallback_colors, Mapping) and row_id in fallback_colors and active.difference(indexed_colors):
            _validate_color(fallback_colors[row_id], path=f"record.meta.base_highlight_color.{row_id}")


def _validate_connector_metadata(meta: Mapping[str, object], *, sequence_length: int, style: Style) -> None:
    if not bool(style.connectors and style.show_reverse_complement):
        return
    index_sets: dict[str, tuple[int, ...]] = {}
    for key in ("connector_hidden_indices", "connector_cross_indices", "connector_emphasis_indices"):
        raw = meta.get(key)
        if raw is not None:
            index_sets[key] = _validated_indices(raw, path=f"record.meta.{key}")

    active_emphasis = bool(_active(index_sets.get("connector_emphasis_indices", ()), sequence_length))
    active_cross = bool(_active(index_sets.get("connector_cross_indices", ()), sequence_length))
    for key, consumed in (
        ("connector_emphasis_color", active_emphasis),
        ("connector_cross_color", active_cross),
    ):
        raw = meta.get(key)
        if consumed and raw is not None and str(raw).strip():
            _validate_color(raw, path=f"record.meta.{key}")
    for key, consumed in (
        ("connector_emphasis_linewidth", active_emphasis),
        ("connector_cross_linewidth", active_cross),
    ):
        raw = meta.get(key)
        if not consumed or raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue  # Drawing uses its validated style fallback.
        if not math.isfinite(value):
            raise RenderingError(f"record.meta.{key} must be finite")
    raw_alpha = meta.get("connector_cross_alpha")
    if active_cross and raw_alpha is not None:
        try:
            alpha = float(raw_alpha)
        except (TypeError, ValueError, OverflowError):
            return  # Drawing uses its validated style fallback.
        if not math.isfinite(alpha):
            raise RenderingError("record.meta.connector_cross_alpha must be finite")


def _iter_mappings(meta: Mapping[str, object], key: str):
    raw = meta.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return
    yield from (item for item in raw if isinstance(item, Mapping))


def _drawn_span(raw: Mapping[str, object]) -> bool:
    try:
        return int(raw.get("end")) > int(raw.get("start"))
    except (TypeError, ValueError, OverflowError):
        return False


def _drawn_position(raw: Mapping[str, object], key: str) -> bool:
    try:
        int(raw.get(key))
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _validate_float(
    raw: object,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return  # Drawing uses a fallback or ignores the item.
    invalid = not math.isfinite(value)
    invalid = invalid or (minimum is not None and value < minimum)
    invalid = invalid or (maximum is not None and value > maximum)
    if not invalid:
        return
    expected = "finite"
    if minimum is not None and maximum is not None:
        expected = f"finite and in [{minimum:g}, {maximum:g}]"
    elif minimum is not None:
        expected = f"finite and >= {minimum:g}"
    raise RenderingError(f"{path} must be {expected}")


def _validate_annotation_metadata(record: Record, *, style: Style) -> None:
    meta = record.meta
    assert isinstance(meta, Mapping)
    feature_ids = {feature.id for feature in record.features}

    for raw in _iter_mappings(meta, "span_backdrops"):
        try:
            alpha = float(raw.get("alpha"))
            corner_radius = float(raw.get("corner_radius"))
        except (TypeError, ValueError, OverflowError):
            continue
        fill = str(raw.get("fill", "")).strip()
        cover_rows = str(raw.get("cover_rows", "both")).strip().lower()
        if not _drawn_span(raw) or not fill or cover_rows not in {"primary", "complement", "both"}:
            continue
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise RenderingError("record.meta.span_backdrops alpha must be finite and in [0, 1]")
        if not math.isfinite(corner_radius) or corner_radius < 0.0:
            raise RenderingError("record.meta.span_backdrops corner_radius must be finite and >= 0")
        _validate_color(fill, path="record.meta.span_backdrops fill")
        edge_color = str(raw.get("edge_color", "")).strip()
        try:
            edge_linewidth = max(0.0, float(raw.get("edge_linewidth", 0.0)))
        except (TypeError, ValueError, OverflowError):
            edge_linewidth = 0.0
        if edge_linewidth > 0.0 and not math.isfinite(edge_linewidth):
            raise RenderingError("record.meta.span_backdrops edge_linewidth must be finite")
        if edge_color and edge_linewidth > 0.0:
            _validate_color(edge_color, path="record.meta.span_backdrops edge_color")

    for raw in _iter_mappings(meta, "span_edge_markers"):
        cover_rows = str(raw.get("cover_rows", "both")).strip().lower()
        if not _drawn_span(raw) or cover_rows not in {"primary", "complement", "both", "all"}:
            continue
        _validate_color(
            str(raw.get("color", "#111827")).strip() or "#111827",
            path="record.meta.span_edge_markers color",
        )
        _validate_float(
            raw.get("alpha", 1.0),
            path="record.meta.span_edge_markers alpha",
            minimum=0.0,
            maximum=1.0,
        )
        _validate_float(
            raw.get("linewidth", 1.0),
            path="record.meta.span_edge_markers linewidth",
            minimum=0.0,
        )

    for raw in _iter_mappings(meta, "boundary_ticks"):
        if not style.show_coordinate_ticks or not _drawn_position(raw, "position"):
            continue
        _validate_color(str(raw.get("color", "#111827")).strip() or "#111827", path="record.meta.boundary_ticks color")
        _validate_float(raw.get("linewidth", 1.0), path="record.meta.boundary_ticks linewidth", minimum=0.0)
        _validate_float(raw.get("font_size", 14.0), path="record.meta.boundary_ticks font_size", minimum=0.0)

    for raw in _iter_mappings(meta, "span_brackets"):
        if str(raw.get("target_feature_id", "")).strip() not in feature_ids:
            continue
        _validate_color(str(raw.get("color", "#111827")).strip() or "#111827", path="record.meta.span_brackets color")
        _validate_float(raw.get("offset_px", 4.0), path="record.meta.span_brackets offset_px")
        _validate_float(raw.get("height_px", 6.0), path="record.meta.span_brackets height_px")
        _validate_float(raw.get("linewidth", 1.15), path="record.meta.span_brackets linewidth", minimum=0.0)
        _validate_float(raw.get("font_size", 13.0), path="record.meta.span_brackets font_size", minimum=0.0)

    for raw in _iter_mappings(meta, "segment_labels"):
        if not str(raw.get("text", "")).strip() or not _drawn_span(raw):
            continue
        _validate_color(str(raw.get("color", "#111827")).strip() or "#111827", path="record.meta.segment_labels color")
        _validate_float(raw.get("label_offset_px", 0.0), path="record.meta.segment_labels label_offset_px")


def validate_sequence_rows_metadata(record: Record, style: Style) -> None:
    """Reject consumed metadata that would otherwise fail after figure allocation.

    Keep this in lockstep with raw ``record.meta`` conversions in
    ``SequenceRowsRenderer.render`` and its drawing helpers.
    """

    meta = record.meta
    if not isinstance(meta, Mapping):
        return
    rows = _ROWS if bool(style.show_reverse_complement and record.alphabet in {"DNA", "IUPAC_DNA"}) else ("primary",)
    highlights = _row_indices(meta, "base_highlights", rows=rows)
    dim_indices = _row_indices(meta, "dim_base_indices", rows=rows)
    _row_indices(meta, "base_hidden_indices", rows=rows)
    _validate_highlight_colors(
        meta,
        highlights=highlights,
        sequence_length=len(record.sequence),
        rows=rows,
    )
    active_dim = any(
        _active(indices, len(record.sequence)).difference(_active(highlights.get(row_id, ()), len(record.sequence)))
        for row_id, indices in dim_indices.items()
    )
    raw_dim_color = meta.get("base_dim_color")
    if active_dim and isinstance(raw_dim_color, str) and raw_dim_color.strip():
        _validate_color(raw_dim_color, path="record.meta.base_dim_color")
    _validate_connector_metadata(meta, sequence_length=len(record.sequence), style=style)
    _validate_annotation_metadata(record, style=style)


__all__ = ["validate_sequence_rows_metadata"]
