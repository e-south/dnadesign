"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/models.py

Value objects for generic linear ssDNA composition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.contracts.sequence import LinearSsdnaCompositionV1


@dataclass(frozen=True)
class LinearSsdnaCompositionResult:
    composition_id: str
    sequence_length: int
    sequence_sha256: str
    artifact_bundle: Path
    manifest_path: Path


@dataclass(frozen=True)
class LinearSsdnaCompositionSummary:
    composition_id: str
    unit_count: int
    expanded_copy_count: int
    sequence_length: int


@dataclass(frozen=True)
class ResolvedSegment:
    unit_id: str
    segment_id: str
    role: str
    sequence: str
    source: dict[str, Any] | None
    transform: dict[str, Any] | None


@dataclass(frozen=True)
class SegmentSpan:
    copy_index: int
    unit_id: str
    segment_id: str
    role: str
    start: int
    end: int
    sequence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "copy_index": self.copy_index,
            "unit_id": self.unit_id,
            "segment_id": self.segment_id,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class AnnotationSpan:
    copy_index: int
    unit_id: str
    annotation_id: str
    role: str
    semantic_label: str | None
    start: int
    end: int
    sequence: str

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "copy_index": self.copy_index,
            "unit_id": self.unit_id,
            "annotation_id": self.annotation_id,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "sequence": self.sequence,
        }
        if self.semantic_label is not None:
            payload["semantic_label"] = self.semantic_label
        return payload


@dataclass(frozen=True)
class UnitCopySpan:
    unit_id: str
    copy_index: int
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "copy_index": self.copy_index,
            "span": {"start": self.start, "end": self.end},
        }


@dataclass(frozen=True)
class ComposedLinearSsdna:
    config: LinearSsdnaCompositionV1
    sequence: str
    sequence_sha256: str
    config_sha256: str
    unit_copies: list[UnitCopySpan]
    segment_spans: list[SegmentSpan]
    annotation_spans: list[AnnotationSpan]
    assertions: list[dict[str, object]]
    provenance_inputs: list[dict[str, object]]
