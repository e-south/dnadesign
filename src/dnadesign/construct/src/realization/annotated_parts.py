"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/annotated_parts.py

Atomic placement of digest-pinned annotated sequence parts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from dnadesign.contracts.sequence import (
    AnnotatedSequenceFeatureV1,
    AnnotatedSequencePartV1,
    AnnotatedSequenceSourceRefV1,
)

from ..contracts.errors import ValidationError
from ..sequences.orientation import reverse_complement

_IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")


def _digest(sequence: str) -> str:
    return f"sha256:{hashlib.sha256(sequence.encode()).hexdigest()}"


def _dna_sequence(value: str, *, label: str) -> str:
    sequence = str(value or "").strip().upper()
    if not sequence:
        raise ValidationError(f"{label} cannot be empty.")
    invalid = sorted(set(sequence) - _IUPAC_DNA)
    if invalid:
        raise ValidationError(f"{label} contains invalid IUPAC DNA: {', '.join(invalid)}.")
    return sequence


@dataclass(frozen=True)
class RealizedAnnotatedFeature:
    """One source feature projected into realized construct coordinates."""

    feature_id: str
    role: str
    owner: str
    source_digest: str
    source_start: int
    source_end: int
    realized_start: int
    realized_end: int
    orientation: str
    sequence: str


@dataclass(frozen=True)
class AnnotatedPartPlacement:
    """Construct result preserving one atomic part and its nested lineage."""

    template_id: str
    template_sequence_digest: str
    source_part_id: str
    source_part_digest: str
    source_refs: tuple[AnnotatedSequenceSourceRefV1, ...]
    placement_kind: str
    orientation: str
    template_start: int
    template_end: int
    part_start: int
    part_end: int
    realized_part_sequence: str
    realized_part_digest: str
    sequence: str
    sequence_digest: str
    features: tuple[RealizedAnnotatedFeature, ...]


def _realize_feature(
    feature: AnnotatedSequenceFeatureV1,
    *,
    part_length: int,
    part_start: int,
    orientation: Literal["forward", "reverse_complement"],
) -> RealizedAnnotatedFeature:
    if orientation == "forward":
        local_start, local_end = feature.start, feature.end
        realized_orientation = feature.orientation
        sequence = feature.sequence
    else:
        local_start = part_length - feature.end
        local_end = part_length - feature.start
        realized_orientation = {
            "forward": "reverse_complement",
            "reverse_complement": "forward",
            "not_asserted": "not_asserted",
        }[feature.orientation]
        sequence = reverse_complement(feature.sequence).upper()
    return RealizedAnnotatedFeature(
        feature_id=feature.feature_id,
        role=feature.role,
        owner=feature.owner,
        source_digest=feature.source_digest,
        source_start=feature.start,
        source_end=feature.end,
        realized_start=part_start + local_start,
        realized_end=part_start + local_end,
        orientation=realized_orientation,
        sequence=sequence,
    )


def place_annotated_part(
    *,
    template_id: str,
    template_sequence: str,
    part: AnnotatedSequencePartV1,
    placement_kind: Literal["insert", "replace"],
    start: int,
    end: int,
    orientation: Literal["forward", "reverse_complement"] = "forward",
) -> AnnotatedPartPlacement:
    """Place one annotated part without reconstructing its nested features."""
    normalized_template_id = str(template_id or "").strip()
    if not normalized_template_id:
        raise ValidationError("template_id cannot be empty.")
    template = _dna_sequence(template_sequence, label="template_sequence")
    if part.topology == "circular":
        raise ValidationError("A circular annotated part requires an explicit linearization contract before placement.")
    if placement_kind not in {"insert", "replace"}:
        raise ValidationError("placement_kind must be 'insert' or 'replace'.")
    if start < 0 or end < start or end > len(template):
        raise ValidationError("Annotated-part placement bounds must lie within the template.")
    if placement_kind == "insert" and end != start:
        raise ValidationError("Annotated-part insert placement requires end == start.")
    if placement_kind == "replace" and end == start:
        raise ValidationError("Annotated-part replace placement requires end > start.")
    if orientation not in {"forward", "reverse_complement"}:
        raise ValidationError("Annotated-part orientation must be 'forward' or 'reverse_complement'.")

    realized_part_sequence = part.sequence if orientation == "forward" else reverse_complement(part.sequence).upper()
    realized_sequence = f"{template[:start]}{realized_part_sequence}{template[end:]}"
    part_end = start + len(realized_part_sequence)
    features = tuple(
        sorted(
            (
                _realize_feature(
                    feature,
                    part_length=len(part.sequence),
                    part_start=start,
                    orientation=orientation,
                )
                for feature in part.features
            ),
            key=lambda feature: (
                feature.realized_start,
                feature.realized_end,
                feature.feature_id,
            ),
        )
    )
    return AnnotatedPartPlacement(
        template_id=normalized_template_id,
        template_sequence_digest=_digest(template),
        source_part_id=part.part_id,
        source_part_digest=part.sequence_digest,
        source_refs=part.source_refs,
        placement_kind=placement_kind,
        orientation=orientation,
        template_start=start,
        template_end=end,
        part_start=start,
        part_end=part_end,
        realized_part_sequence=realized_part_sequence,
        realized_part_digest=_digest(realized_part_sequence),
        sequence=realized_sequence,
        sequence_digest=_digest(realized_sequence),
        features=features,
    )


__all__ = [
    "AnnotatedPartPlacement",
    "RealizedAnnotatedFeature",
    "place_annotated_part",
]
