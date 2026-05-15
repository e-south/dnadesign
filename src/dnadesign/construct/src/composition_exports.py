"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition_exports.py

Export writers for linear ssDNA composition bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_sequence_exports(artifact_bundle: Path, composed: Any) -> None:
    _write_fasta(artifact_bundle / "sequence.fa", composed)
    _write_features_csv(artifact_bundle / "features.csv", composed)
    _write_genbank(artifact_bundle / "sequence.gb", composed)


def _write_fasta(path: Path, composed: Any) -> None:
    lines = [
        f">{composed.config.composition_id} length={len(composed.sequence)} topology={composed.config.topology}",
        composed.sequence,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_features_csv(path: Path, composed: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "copy_index",
                "feature_kind",
                "feature_id",
                "role",
                "semantic_label",
                "start_0",
                "end_0",
                "strand",
                "source_segment_id",
                "transform_kind",
                "genbank_location",
                "sequence",
            ],
        )
        writer.writeheader()
        for span in composed.segment_spans:
            writer.writerow(
                _feature_row(
                    span.copy_index,
                    "segment",
                    span.segment_id,
                    span.role,
                    None,
                    span.start,
                    span.end,
                    span.sequence,
                    **_segment_export_metadata(composed, unit_id=span.unit_id, segment_id=span.segment_id),
                )
            )
        for span in composed.annotation_spans:
            writer.writerow(
                _feature_row(
                    span.copy_index,
                    "annotation",
                    span.annotation_id,
                    span.role,
                    span.semantic_label,
                    span.start,
                    span.end,
                    span.sequence,
                    **_annotation_export_metadata(
                        composed,
                        unit_id=span.unit_id,
                        annotation_id=span.annotation_id,
                    ),
                )
            )


def _write_genbank(path: Path, composed: Any) -> None:
    lines = [
        f"LOCUS       {composed.config.composition_id} {len(composed.sequence)} bp ss-DNA linear SYN",
        "FEATURES             Location/Qualifiers",
    ]
    for span in composed.segment_spans:
        lines.extend(
            _genbank_feature_lines(
                "misc_feature",
                span.start,
                span.end,
                f"{span.segment_id} copy {span.copy_index}",
                **_segment_export_metadata(composed, unit_id=span.unit_id, segment_id=span.segment_id),
            )
        )
    for span in composed.annotation_spans:
        lines.extend(
            _genbank_feature_lines(
                "misc_feature",
                span.start,
                span.end,
                f"{span.annotation_id} copy {span.copy_index}",
                **_annotation_export_metadata(composed, unit_id=span.unit_id, annotation_id=span.annotation_id),
            )
        )
    lines.append("ORIGIN")
    for offset in range(0, len(composed.sequence), 60):
        chunk = composed.sequence[offset : offset + 60].lower()
        grouped = " ".join(chunk[index : index + 10] for index in range(0, len(chunk), 10))
        lines.append(f"{offset + 1:>9} {grouped}")
    lines.append("//")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _feature_row(
    copy_index: int,
    feature_kind: str,
    feature_id: str,
    role: str,
    semantic_label: str | None,
    start: int,
    end: int,
    sequence: str,
    strand: int = 1,
    source_segment_id: str | None = None,
    transform_kind: str | None = None,
) -> dict[str, object]:
    return {
        "copy_index": copy_index,
        "feature_kind": feature_kind,
        "feature_id": feature_id,
        "role": role,
        "semantic_label": semantic_label or "",
        "start_0": start,
        "end_0": end,
        "strand": strand,
        "source_segment_id": source_segment_id or "",
        "transform_kind": transform_kind or "",
        "genbank_location": _genbank_location(start, end, strand=strand),
        "sequence": sequence,
    }


def _genbank_feature_lines(
    feature_kind: str,
    start: int,
    end: int,
    label: str,
    strand: int = 1,
    source_segment_id: str | None = None,
    transform_kind: str | None = None,
) -> list[str]:
    qualifiers = [f'                     /label="{label}"']
    if strand == -1:
        qualifiers.append('                     /strand="-1"')
    if source_segment_id:
        qualifiers.append(f'                     /dnadesign_source_segment="{source_segment_id}"')
    if transform_kind:
        qualifiers.append(f'                     /dnadesign_transform="{transform_kind}"')
    return [f"     {feature_kind:<16}{_genbank_location(start, end, strand=strand)}", *qualifiers]


def _genbank_location(start: int, end: int, *, strand: int = 1) -> str:
    location = f"{start + 1}..{end}"
    if strand == -1:
        return f"complement({location})"
    return location


def _segment_export_metadata(composed: Any, *, unit_id: str, segment_id: str) -> dict[str, object]:
    source_segment_id = _reverse_complement_source_segment_id(composed, unit_id=unit_id, segment_id=segment_id)
    if source_segment_id is None:
        return {"strand": 1, "source_segment_id": None, "transform_kind": None}
    return {
        "strand": -1,
        "source_segment_id": source_segment_id,
        "transform_kind": "reverse_complement",
    }


def _annotation_export_metadata(composed: Any, *, unit_id: str, annotation_id: str) -> dict[str, object]:
    segment_id = _annotation_segment_id(composed, unit_id=unit_id, annotation_id=annotation_id)
    if segment_id is None:
        return {"strand": 1, "source_segment_id": None, "transform_kind": None}
    return _segment_export_metadata(composed, unit_id=unit_id, segment_id=segment_id)


def _reverse_complement_source_segment_id(composed: Any, *, unit_id: str, segment_id: str) -> str | None:
    for unit in composed.config.units:
        if unit.unit_id != unit_id:
            continue
        for segment in unit.segments:
            if segment.segment_id != segment_id or segment.transform is None:
                continue
            if segment.transform.kind == "reverse_complement":
                return segment.transform.source_segment_id
    return None


def _annotation_segment_id(composed: Any, *, unit_id: str, annotation_id: str) -> str | None:
    for unit in composed.config.units:
        if unit.unit_id != unit_id:
            continue
        for annotation in unit.annotations:
            if annotation.annotation_id != annotation_id:
                continue
            if annotation.location.basis == "segment":
                return annotation.location.segment_id
    return None


__all__ = ["write_sequence_exports"]
