"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/sequence_evidence_meta.py

Shared helpers for producer metadata attached to sequence-evidence contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

SEQUENCE_EVIDENCE_DEFAULT_SPAN_BACKDROP_FILL = "#BFDBFE"
SEQUENCE_EVIDENCE_DEFAULT_SPAN_BACKDROP_ALPHA = 0.3
SEQUENCE_EVIDENCE_DEFAULT_SPAN_BACKDROP_CORNER_RADIUS = 8.0
SEQUENCE_EVIDENCE_ALLOWED_SPAN_BACKDROP_ROW_COVERAGE = frozenset({"primary", "complement", "both"})


def normalize_sequence_evidence_row_labels(meta: Mapping[str, object]) -> dict[str, str]:
    row_labels_raw = meta.get("row_labels")
    if row_labels_raw is None:
        row_labels: Mapping[str, object] = {}
    elif isinstance(row_labels_raw, Mapping):
        row_labels = row_labels_raw
    else:
        raise ValueError("sequence-evidence meta.row_labels must be a mapping when provided")
    return {
        "primary": str(row_labels.get("primary") or "").strip(),
        "complement": str(row_labels.get("complement") or "").strip(),
    }


def build_sequence_evidence_connector_meta(
    *,
    start: int,
    end: int,
    cross_indices: Iterable[int],
    coordinate_space: str | None = None,
) -> dict[str, object]:
    if end <= start:
        raise ValueError("sequence-evidence connector spans require end > start")
    normalized_cross_indices = sorted({int(index) for index in cross_indices})
    if any(index < start or index >= end for index in normalized_cross_indices):
        raise ValueError("sequence-evidence connector cross indices must lie within the connector span")
    span: dict[str, object] = {"start": int(start), "end": int(end)}
    if coordinate_space is not None:
        span["coordinate_space"] = str(coordinate_space)
    crossed = set(normalized_cross_indices)
    return {
        "connector_hidden_indices": [index for index in range(int(start), int(end)) if index not in crossed],
        "connector_cross_indices": normalized_cross_indices,
        "connector_overhang_spans": [span],
    }


def build_sequence_evidence_connector_span_meta(
    *,
    start: int,
    end: int,
    coordinate_space: str | None = None,
) -> dict[str, object]:
    if end <= start:
        raise ValueError("sequence-evidence connector spans require end > start")
    span: dict[str, object] = {"start": int(start), "end": int(end)}
    if coordinate_space is not None:
        span["coordinate_space"] = str(coordinate_space)
    return {
        "connector_hidden_indices": [],
        "connector_cross_indices": [],
        "connector_overhang_spans": [span],
    }


def build_sequence_evidence_span_backdrop_meta(
    *,
    start: int,
    end: int,
    coordinate_space: str | None = None,
    fill: str = SEQUENCE_EVIDENCE_DEFAULT_SPAN_BACKDROP_FILL,
    alpha: float = SEQUENCE_EVIDENCE_DEFAULT_SPAN_BACKDROP_ALPHA,
    corner_radius: float = SEQUENCE_EVIDENCE_DEFAULT_SPAN_BACKDROP_CORNER_RADIUS,
    cover_rows: str = "both",
) -> dict[str, object]:
    if end <= start:
        raise ValueError("sequence-evidence span backdrops require end > start")
    fill_text = str(fill).strip()
    if not fill_text:
        raise ValueError("sequence-evidence span backdrops require a non-empty fill color")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or alpha_value < 0.0 or alpha_value > 1.0:
        raise ValueError("sequence-evidence span backdrop alpha must be finite and within [0, 1]")
    corner_radius_value = float(corner_radius)
    if not math.isfinite(corner_radius_value) or corner_radius_value < 0.0:
        raise ValueError("sequence-evidence span backdrop corner radius must be finite and >= 0")
    cover_rows_value = str(cover_rows).strip().lower()
    if cover_rows_value not in SEQUENCE_EVIDENCE_ALLOWED_SPAN_BACKDROP_ROW_COVERAGE:
        raise ValueError("sequence-evidence span backdrop cover_rows must be primary, complement, or both")
    span: dict[str, object] = {
        "start": int(start),
        "end": int(end),
        "fill": fill_text,
        "alpha": alpha_value,
        "corner_radius": corner_radius_value,
        "cover_rows": cover_rows_value,
    }
    if coordinate_space is not None:
        span["coordinate_space"] = str(coordinate_space)
    return {"span_backdrops": [span]}


__all__ = [
    "build_sequence_evidence_connector_meta",
    "build_sequence_evidence_connector_span_meta",
    "build_sequence_evidence_span_backdrop_meta",
    "normalize_sequence_evidence_row_labels",
]
