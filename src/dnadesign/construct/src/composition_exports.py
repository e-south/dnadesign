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
                "genbank_location",
                "sequence",
            ],
        )
        writer.writeheader()
        for span in composed.segment_spans:
            writer.writerow(
                _feature_row(
                    span.copy_index, "segment", span.segment_id, span.role, None, span.start, span.end, span.sequence
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
                )
            )


def _write_genbank(path: Path, composed: Any) -> None:
    lines = [
        f"LOCUS       {composed.config.composition_id} {len(composed.sequence)} bp ss-DNA linear SYN",
        "FEATURES             Location/Qualifiers",
    ]
    for span in composed.segment_spans:
        lines.extend(
            _genbank_feature_lines("misc_feature", span.start, span.end, f"{span.segment_id} copy {span.copy_index}")
        )
    for span in composed.annotation_spans:
        lines.extend(
            _genbank_feature_lines("misc_feature", span.start, span.end, f"{span.annotation_id} copy {span.copy_index}")
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
) -> dict[str, object]:
    return {
        "copy_index": copy_index,
        "feature_kind": feature_kind,
        "feature_id": feature_id,
        "role": role,
        "semantic_label": semantic_label or "",
        "start_0": start,
        "end_0": end,
        "genbank_location": _genbank_location(start, end),
        "sequence": sequence,
    }


def _genbank_feature_lines(feature_kind: str, start: int, end: int, label: str) -> list[str]:
    return [
        f"     {feature_kind:<15}{_genbank_location(start, end)}",
        f'                     /label="{label}"',
    ]


def _genbank_location(start: int, end: int) -> str:
    return f"{start + 1}..{end}"


__all__ = ["write_sequence_exports"]
