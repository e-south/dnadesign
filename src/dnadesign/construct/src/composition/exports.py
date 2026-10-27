"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/exports.py

Export writers for linear ssDNA composition bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from ..sequences.orientation import reverse_complement


def write_sequence_exports(artifact_bundle: Path, composed: Any) -> None:
    _write_fasta(
        artifact_bundle / "sequence.fa",
        composed,
        sequence=composed.sequence,
        sequence_id=composed.config.composition_id,
        orientation="forward",
    )
    _write_features_csv(artifact_bundle / "features.csv", composed)
    _write_genbank(
        artifact_bundle / "sequence.gb",
        composed,
        sequence=composed.sequence,
        locus_id=composed.config.composition_id,
        orientation="forward",
    )
    _write_fasta(
        artifact_bundle / "sequence.reverse_complement.fa",
        composed,
        sequence=reverse_complement(composed.sequence),
        sequence_id=f"{composed.config.composition_id}_reverse_complement",
        orientation="reverse_complement",
    )
    _write_genbank(
        artifact_bundle / "sequence.reverse_complement.gb",
        composed,
        sequence=reverse_complement(composed.sequence),
        locus_id=f"{composed.config.composition_id}_reverse_complement",
        orientation="reverse_complement",
    )


def _write_fasta(path: Path, composed: Any, *, sequence: str, sequence_id: str, orientation: str) -> None:
    header = f">{sequence_id} length={len(sequence)} topology={composed.config.topology}"
    if orientation != "forward":
        header = f"{header} orientation={orientation}"
    lines = [
        header,
        sequence,
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
                "display_label",
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
                    _feature_display_label(
                        composed,
                        feature_kind="segment",
                        feature_id=span.segment_id,
                        role=span.role,
                    ),
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
                    _feature_display_label(
                        composed,
                        feature_kind="annotation",
                        feature_id=span.annotation_id,
                        role=span.role,
                        semantic_label=span.semantic_label,
                    ),
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


def _write_genbank(path: Path, composed: Any, *, sequence: str, locus_id: str, orientation: str) -> None:
    lines = [
        f"LOCUS       {locus_id} {len(sequence)} bp ss-DNA linear SYN",
        "FEATURES             Location/Qualifiers",
    ]
    for span in composed.segment_spans:
        display_label = _feature_display_label(
            composed,
            feature_kind="segment",
            feature_id=span.segment_id,
            role=span.role,
        )
        lines.extend(
            _genbank_feature_lines(
                "misc_feature",
                *_feature_location_for_orientation(span.start, span.end, len(composed.sequence), orientation),
                display_label,
                feature_kind_id="segment",
                feature_id=span.segment_id,
                role=span.role,
                copy_index=span.copy_index,
                orientation=orientation,
                forward_start=span.start,
                forward_end=span.end,
                **_segment_export_metadata(composed, unit_id=span.unit_id, segment_id=span.segment_id),
            )
        )
    for span in composed.annotation_spans:
        display_label = _feature_display_label(
            composed,
            feature_kind="annotation",
            feature_id=span.annotation_id,
            role=span.role,
            semantic_label=span.semantic_label,
        )
        lines.extend(
            _genbank_feature_lines(
                "misc_feature",
                *_feature_location_for_orientation(span.start, span.end, len(composed.sequence), orientation),
                display_label,
                feature_kind_id="annotation",
                feature_id=span.annotation_id,
                role=span.role,
                copy_index=span.copy_index,
                orientation=orientation,
                forward_start=span.start,
                forward_end=span.end,
                **_annotation_export_metadata(composed, unit_id=span.unit_id, annotation_id=span.annotation_id),
            )
        )
    lines.append("ORIGIN")
    for offset in range(0, len(sequence), 60):
        chunk = sequence[offset : offset + 60].lower()
        grouped = " ".join(chunk[index : index + 10] for index in range(0, len(chunk), 10))
        lines.append(f"{offset + 1:>9} {grouped}")
    lines.append("//")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _feature_row(
    copy_index: int,
    feature_kind: str,
    feature_id: str,
    role: str,
    display_label: str,
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
        "display_label": display_label,
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
    feature_kind_id: str,
    feature_id: str,
    role: str,
    copy_index: int,
    orientation: str = "forward",
    forward_start: int | None = None,
    forward_end: int | None = None,
    strand: int = 1,
    source_segment_id: str | None = None,
    transform_kind: str | None = None,
) -> list[str]:
    display_strand = strand if orientation == "forward" else -strand
    qualifiers = [
        f'                     /label="{label}"',
        f'                     /dnadesign_feature_kind="{feature_kind_id}"',
        f'                     /dnadesign_feature_id="{feature_id}"',
        f'                     /dnadesign_role="{role}"',
        f'                     /dnadesign_copy_index="{copy_index}"',
        f'                     /dnadesign_orientation="{orientation}"',
    ]
    if forward_start is not None:
        qualifiers.append(f'                     /dnadesign_forward_start_0="{forward_start}"')
    if forward_end is not None:
        qualifiers.append(f'                     /dnadesign_forward_end_0="{forward_end}"')
    if display_strand == -1:
        qualifiers.append('                     /strand="-1"')
    if source_segment_id:
        qualifiers.append(f'                     /dnadesign_source_segment="{source_segment_id}"')
    if transform_kind:
        qualifiers.append(f'                     /dnadesign_transform="{transform_kind}"')
    return [f"     {feature_kind:<16}{_genbank_location(start, end, strand=display_strand)}", *qualifiers]


def _feature_location_for_orientation(
    forward_start: int,
    forward_end: int,
    sequence_length: int,
    orientation: str,
) -> tuple[int, int]:
    if orientation == "forward":
        return forward_start, forward_end
    if orientation == "reverse_complement":
        return sequence_length - forward_end, sequence_length - forward_start
    raise ValueError(f"Unsupported sequence export orientation: {orientation}")


def _feature_display_label(
    composed: Any,
    *,
    feature_kind: str,
    feature_id: str,
    role: str,
    semantic_label: str | None = None,
) -> str:
    display_profile = composed.config.visual.display_profile
    if feature_kind == "segment":
        return _display_label(feature_id, display_profile.component_labels, fallback=role)
    return _display_label(
        feature_id,
        display_profile.annotation_labels,
        fallback=semantic_label or role,
    )


def _display_label(raw: str, mapping: dict[str, str], *, fallback: str | None = None) -> str:
    label = mapping.get(raw)
    if label is not None:
        return label
    return _pretty_display_label(fallback or raw)


def _pretty_display_label(raw: str) -> str:
    text = re.sub(r"\s+", " ", str(raw).replace("_", " ").replace("-", " ")).strip()
    if not text:
        return str(raw)
    if any(character.isupper() for character in text[1:]):
        return text
    return text[:1].upper() + text[1:]


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
