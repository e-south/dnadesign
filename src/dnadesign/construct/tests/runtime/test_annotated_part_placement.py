"""Atomic placement of a digest-pinned annotated sequence part."""

from __future__ import annotations

import hashlib

import pytest

from dnadesign.construct import place_annotated_part
from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.contracts.sequence import AnnotatedSequencePartV1


def _digest(sequence: str) -> str:
    return f"sha256:{hashlib.sha256(sequence.encode()).hexdigest()}"


def _part() -> AnnotatedSequencePartV1:
    sequence = "AACCGG"
    return AnnotatedSequencePartV1.model_validate(
        {
            "part_id": "msd-retron-43-sequence",
            "strandedness": "not_asserted",
            "topology": "not_asserted",
            "sequence": sequence,
            "sequence_digest": _digest(sequence),
            "source_refs": [
                {
                    "kind": "artifact",
                    "authority": "hop-design",
                    "identifier": "hop:bundle/demo/1234",
                    "digest": _digest("bundle"),
                }
            ],
            "features": [
                {
                    "feature_id": "context-5p",
                    "role": "msd_context_5p",
                    "owner": "research-study",
                    "start": 0,
                    "end": 2,
                    "orientation": "forward",
                    "sequence": "AA",
                    "source_digest": _digest(sequence),
                },
                {
                    "feature_id": "hop-core",
                    "role": "hairpin_encoding",
                    "owner": "hop-design",
                    "start": 2,
                    "end": 6,
                    "orientation": "forward",
                    "sequence": "CCGG",
                    "source_digest": _digest("encoding"),
                },
            ],
        }
    )


def test_construct_places_one_annotated_part_and_offsets_nested_features() -> None:
    part = _part()

    placed = place_annotated_part(
        template_id="expression-cassette",
        template_sequence="TTTTTT",
        part=part,
        placement_kind="replace",
        start=2,
        end=4,
        orientation="forward",
    )

    assert placed.sequence == "TTAACCGGTT"
    assert (placed.part_start, placed.part_end) == (2, 8)
    assert placed.source_part_digest == part.sequence_digest
    assert placed.source_refs == part.source_refs
    assert [(feature.feature_id, feature.realized_start, feature.realized_end) for feature in placed.features] == [
        ("context-5p", 2, 4),
        ("hop-core", 4, 8),
    ]
    assert placed.features[1].source_digest == part.features[1].source_digest


def test_reverse_complement_placement_transforms_nested_coordinates() -> None:
    part = _part()

    placed = place_annotated_part(
        template_id="expression-cassette",
        template_sequence="TTTTTT",
        part=part,
        placement_kind="replace",
        start=2,
        end=4,
        orientation="reverse_complement",
    )

    assert placed.sequence == "TTCCGGTTTT"
    assert placed.realized_part_sequence == "CCGGTT"
    assert [
        (
            feature.feature_id,
            feature.source_start,
            feature.source_end,
            feature.realized_start,
            feature.realized_end,
            feature.orientation,
        )
        for feature in placed.features
    ] == [
        ("hop-core", 2, 6, 2, 6, "reverse_complement"),
        ("context-5p", 0, 2, 6, 8, "reverse_complement"),
    ]
    assert placed.features[0].source_digest == part.features[1].source_digest


def test_atomic_placement_rejects_circular_part_without_a_linearization_contract() -> None:
    circular_part = _part().model_copy(update={"topology": "circular"})

    with pytest.raises(ValidationError, match="circular annotated part"):
        place_annotated_part(
            template_id="expression-cassette",
            template_sequence="TTTTTT",
            part=circular_part,
            placement_kind="replace",
            start=2,
            end=4,
        )


def test_atomic_placement_rejects_unknown_placement_kind() -> None:
    with pytest.raises(ValidationError, match="placement_kind"):
        place_annotated_part(
            template_id="expression-cassette",
            template_sequence="TTTTTT",
            part=_part(),
            placement_kind="append",  # type: ignore[arg-type]
            start=2,
            end=4,
        )
