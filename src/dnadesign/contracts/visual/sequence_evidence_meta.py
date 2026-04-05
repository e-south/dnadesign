"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/sequence_evidence_meta.py

Shared helpers for producer metadata attached to sequence-evidence contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


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


__all__ = [
    "build_sequence_evidence_connector_meta",
    "normalize_sequence_evidence_row_labels",
]
