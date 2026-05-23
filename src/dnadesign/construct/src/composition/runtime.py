"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/runtime.py

Generic linear ssDNA composition runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from dnadesign.contracts.sequence import LinearSsdnaCompositionV1
from dnadesign.contracts.sequence.linear_ssdna_composition_v1 import (
    LinearSsdnaAnnotationV1,
    LinearSsdnaSegmentV1,
    LinearSsdnaUnitV1,
)

from ..contracts.errors import ConfigError, ValidationError
from ..sequences.orientation import reverse_complement
from .bundle import write_composition_bundle
from .models import (
    AnnotationSpan,
    ComposedLinearSsdna,
    LinearSsdnaCompositionResult,
    LinearSsdnaCompositionSummary,
    ResolvedSegment,
    SegmentSpan,
    UnitCopySpan,
)

_DNA4 = frozenset("ACGTacgt")
_IUPAC_DNA = frozenset("ACGTRYSWKMBDHVNacgtryswkmbdhvn")


def load_linear_ssdna_composition_config(path: str | Path) -> tuple[LinearSsdnaCompositionV1, Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Composition config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in composition config: {config_path}") from exc
    try:
        return LinearSsdnaCompositionV1.model_validate(data), config_path
    except PydanticValidationError as exc:
        raise ConfigError(f"Invalid composition config {config_path}: {exc}") from exc


def summarize_linear_ssdna_composition(path: str | Path) -> LinearSsdnaCompositionSummary:
    config, config_path = load_linear_ssdna_composition_config(path)
    composed = _compose(config, config_path=config_path)
    return LinearSsdnaCompositionSummary(
        composition_id=config.composition_id,
        unit_count=len(config.units),
        expanded_copy_count=sum(unit.repeat_count for unit in config.units),
        sequence_length=len(composed.sequence),
    )


def run_linear_ssdna_composition(path: str | Path) -> LinearSsdnaCompositionResult:
    config, config_path = load_linear_ssdna_composition_config(path)
    composed = _compose(config, config_path=config_path)
    artifact_bundle = _resolve_artifact_bundle(config, config_path=config_path)
    write_composition_bundle(composed, artifact_bundle=artifact_bundle, config_path=config_path)
    return LinearSsdnaCompositionResult(
        composition_id=config.composition_id,
        sequence_length=len(composed.sequence),
        sequence_sha256=composed.sequence_sha256,
        artifact_bundle=artifact_bundle,
        manifest_path=artifact_bundle / "manifest.json",
    )


def _compose(config: LinearSsdnaCompositionV1, *, config_path: Path) -> ComposedLinearSsdna:
    sequence_parts: list[str] = []
    unit_copies: list[UnitCopySpan] = []
    segment_spans: list[SegmentSpan] = []
    annotation_spans: list[AnnotationSpan] = []
    assertion_results: list[dict[str, object]] = []
    provenance_inputs: list[dict[str, object]] = []
    cursor = 0
    config_sha256 = _sha256_text(config_path.read_text(encoding="utf-8"))

    for unit in config.units:
        resolved_segments = _resolve_unit_segments(config, unit)
        resolved_by_id = {segment.segment_id: segment for segment in resolved_segments}
        unit_sequence = "".join(segment.sequence for segment in resolved_segments)
        if not unit_sequence:
            raise ValidationError(f"unit '{unit.unit_id}' assembled to an empty sequence.")
        for assertion in unit.assertions:
            status = _check_reverse_complement(
                left=resolved_by_id[assertion.left_segment_id].sequence,
                right=resolved_by_id[assertion.right_segment_id].sequence,
            )
            if not status and assertion.severity == "error":
                raise ValidationError(
                    f"assertion '{assertion.assertion_id}' failed: "
                    f"{assertion.right_segment_id} is not the reverse complement of {assertion.left_segment_id}."
                )
            assertion_results.append(
                {
                    "assertion_id": assertion.assertion_id,
                    "kind": assertion.kind,
                    "severity": assertion.severity,
                    "status": "pass" if status else "fail",
                }
            )
        for copy_index in range(unit.repeat_count):
            unit_start = cursor
            sequence_parts.append(unit_sequence)
            unit_end = unit_start + len(unit_sequence)
            unit_copies.append(
                UnitCopySpan(unit_id=unit.unit_id, copy_index=copy_index, start=unit_start, end=unit_end)
            )
            copy_segment_spans = _segment_spans_for_copy(
                copy_index=copy_index,
                unit_id=unit.unit_id,
                unit_start=unit_start,
                resolved_segments=resolved_segments,
            )
            segment_spans.extend(copy_segment_spans)
            annotation_spans.extend(
                _annotation_spans_for_copy(
                    unit=unit,
                    copy_index=copy_index,
                    unit_start=unit_start,
                    unit_end=unit_end,
                    sequence="".join(sequence_parts),
                    segment_spans=copy_segment_spans,
                )
            )
            cursor = unit_end
        for segment in resolved_segments:
            if segment.source is None:
                continue
            provenance_inputs.append(
                {
                    "unit_id": unit.unit_id,
                    "segment_id": segment.segment_id,
                    "source": segment.source,
                    "transform": segment.transform,
                }
            )

    sequence = "".join(sequence_parts)
    return ComposedLinearSsdna(
        config=config,
        sequence=sequence,
        sequence_sha256=_sha256_text(sequence),
        config_sha256=config_sha256,
        unit_copies=unit_copies,
        segment_spans=segment_spans,
        annotation_spans=annotation_spans,
        assertions=assertion_results,
        provenance_inputs=provenance_inputs,
    )


def _resolve_unit_segments(config: LinearSsdnaCompositionV1, unit: LinearSsdnaUnitV1) -> list[ResolvedSegment]:
    resolved: list[ResolvedSegment] = []
    resolved_by_id: dict[str, ResolvedSegment] = {}
    for segment in unit.segments:
        source_sequence = _resolve_segment_sequence(config, unit=unit, segment=segment, resolved_by_id=resolved_by_id)
        resolved_segment = ResolvedSegment(
            unit_id=unit.unit_id,
            segment_id=segment.segment_id,
            role=segment.role,
            sequence=source_sequence,
            source=_dump_model(segment.source),
            transform=_dump_model(segment.transform),
        )
        resolved.append(resolved_segment)
        resolved_by_id[segment.segment_id] = resolved_segment
    return resolved


def _resolve_segment_sequence(
    config: LinearSsdnaCompositionV1,
    *,
    unit: LinearSsdnaUnitV1,
    segment: LinearSsdnaSegmentV1,
    resolved_by_id: dict[str, ResolvedSegment],
) -> str:
    if segment.transform is None:
        if segment.sequence is None:
            raise ValidationError(f"segment '{segment.segment_id}' requires sequence.")
        _validate_sequence(config, segment.sequence, label=f"segment '{segment.segment_id}'")
        return segment.sequence

    if segment.transform.kind != "reverse_complement":
        raise ValidationError(f"segment '{segment.segment_id}' has unsupported transform '{segment.transform.kind}'.")
    source = resolved_by_id.get(segment.transform.source_segment_id)
    if source is None:
        raise ValidationError(
            f"segment '{segment.segment_id}' transform source '{segment.transform.source_segment_id}' "
            f"must appear earlier in unit '{unit.unit_id}'."
        )
    derived = reverse_complement(source.sequence)
    if segment.sequence is None:
        _validate_sequence(config, derived, label=f"segment '{segment.segment_id}'")
        return derived
    _validate_sequence(config, segment.sequence, label=f"segment '{segment.segment_id}'")
    if segment.transform.assert_expected_sequence and segment.sequence.upper() != derived.upper():
        raise ValidationError(
            f"{segment.segment_id} reverse_complement does not match {segment.transform.source_segment_id}: "
            f"expected {derived}, observed {segment.sequence}."
        )
    return segment.sequence


def _validate_sequence(config: LinearSsdnaCompositionV1, sequence: str, *, label: str) -> None:
    allowed = _IUPAC_DNA if config.qa.allow_degenerate_bases else _DNA4
    invalid = sorted({base for base in sequence if base not in allowed})
    if invalid:
        joined = "".join(invalid)
        if config.qa.allow_degenerate_bases:
            raise ValidationError(f"{label} contains invalid IUPAC DNA characters: {joined}.")
        raise ValidationError(f"{label} contains non-ACGT bases: {joined}.")
    if config.qa.require_no_unknown_bases and any(base.upper() == "N" for base in sequence):
        raise ValidationError(f"{label} contains unknown base N while require_no_unknown_bases is true.")


def _segment_spans_for_copy(
    *,
    copy_index: int,
    unit_id: str,
    unit_start: int,
    resolved_segments: list[ResolvedSegment],
) -> list[SegmentSpan]:
    spans: list[SegmentSpan] = []
    cursor = unit_start
    for segment in resolved_segments:
        end = cursor + len(segment.sequence)
        spans.append(
            SegmentSpan(
                copy_index=copy_index,
                unit_id=unit_id,
                segment_id=segment.segment_id,
                role=segment.role,
                start=cursor,
                end=end,
                sequence=segment.sequence,
            )
        )
        cursor = end
    return spans


def _annotation_spans_for_copy(
    *,
    unit: LinearSsdnaUnitV1,
    copy_index: int,
    unit_start: int,
    unit_end: int,
    sequence: str,
    segment_spans: list[SegmentSpan],
) -> list[AnnotationSpan]:
    spans_by_segment = {span.segment_id: span for span in segment_spans}
    annotations: list[AnnotationSpan] = []
    for annotation in unit.annotations:
        start, end = _resolve_annotation_bounds(
            annotation,
            unit_start=unit_start,
            unit_end=unit_end,
            spans_by_segment=spans_by_segment,
        )
        annotations.append(
            AnnotationSpan(
                copy_index=copy_index,
                unit_id=unit.unit_id,
                annotation_id=annotation.annotation_id,
                role=annotation.role,
                semantic_label=annotation.semantic_label,
                start=start,
                end=end,
                sequence=sequence[start:end],
            )
        )
    return annotations


def _resolve_annotation_bounds(
    annotation: LinearSsdnaAnnotationV1,
    *,
    unit_start: int,
    unit_end: int,
    spans_by_segment: dict[str, SegmentSpan],
) -> tuple[int, int]:
    location = annotation.location
    if location.basis == "segment":
        if location.segment_id is None:
            raise ValidationError(f"annotation '{annotation.annotation_id}' uses segment basis without segment_id.")
        if location.segment_id not in spans_by_segment:
            raise ValidationError(
                f"annotation '{annotation.annotation_id}' references unknown segment '{location.segment_id}'."
            )
        segment = spans_by_segment[location.segment_id]
        if location.end > segment.end - segment.start:
            raise ValidationError(
                f"annotation '{annotation.annotation_id}' exceeds segment '{location.segment_id}' bounds."
            )
        return segment.start + location.start, segment.start + location.end
    if location.basis == "unit":
        if unit_start + location.end > unit_end:
            raise ValidationError(f"annotation '{annotation.annotation_id}' exceeds unit bounds.")
        return unit_start + location.start, unit_start + location.end
    return location.start, location.end


def _check_reverse_complement(*, left: str, right: str) -> bool:
    return reverse_complement(left).upper() == right.upper()


def _resolve_artifact_bundle(config: LinearSsdnaCompositionV1, *, config_path: Path) -> Path:
    configured = config.output.artifact_bundle or f"artifacts/{config.composition_id}"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump_model(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return]
    return None


__all__ = [
    "LinearSsdnaCompositionResult",
    "LinearSsdnaCompositionSummary",
    "load_linear_ssdna_composition_config",
    "run_linear_ssdna_composition",
    "summarize_linear_ssdna_composition",
]
