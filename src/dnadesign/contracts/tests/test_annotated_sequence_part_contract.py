"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/test_annotated_sequence_part_contract.py

Neutral annotated sequence-part handoff contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

import dnadesign.contracts as contracts
from dnadesign.contracts.sequence import AnnotatedSequencePartV1


def _digest(sequence: str) -> str:
    return f"sha256:{hashlib.sha256(sequence.encode()).hexdigest()}"


def _payload() -> dict[str, object]:
    sequence = "AACCGG"
    return {
        "contract": "annotated_sequence_part_v1",
        "schema_version": 1,
        "part_id": "msd-retron-43-sequence",
        "representation": "one_dimensional_sequence",
        "molecule_type": "dna",
        "strandedness": "not_asserted",
        "topology": "not_asserted",
        "coordinate_system": "zero_based_half_open",
        "sequence": sequence,
        "sequence_digest": _digest(sequence),
        "source_refs": [
            {
                "kind": "record",
                "authority": "retron-hairpin-design",
                "identifier": "msd-retron-43",
                "digest": _digest("study-record"),
            },
            {
                "kind": "artifact",
                "authority": "hop-design",
                "identifier": "hop:bundle/demo/1234",
                "digest": _digest("hop-bundle"),
            },
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
                "source_digest": _digest("hop-encoding"),
            },
        ],
    }


def test_annotated_sequence_part_accepts_explicit_nonphysical_posture() -> None:
    part = AnnotatedSequencePartV1.model_validate(_payload())

    assert part.strandedness == "not_asserted"
    assert part.topology == "not_asserted"
    assert part.features[1].sequence == "CCGG"
    assert part.source_refs[1].identifier.startswith("hop:bundle/")
    assert contracts.AnnotatedSequencePartV1 is AnnotatedSequencePartV1


def test_annotated_sequence_part_rejects_sequence_digest_drift() -> None:
    payload = _payload()
    payload["sequence_digest"] = _digest("different")

    with pytest.raises(ValidationError, match="sequence_digest does not match"):
        AnnotatedSequencePartV1.model_validate(payload)


def test_annotated_sequence_part_rejects_feature_sequence_drift() -> None:
    payload = _payload()
    payload["features"][1]["sequence"] = "AAAA"

    with pytest.raises(ValidationError, match="feature sequence does not match"):
        AnnotatedSequencePartV1.model_validate(payload)


def test_annotated_sequence_part_rejects_noncanonical_lowercase_sequence() -> None:
    payload = _payload()
    payload["sequence"] = "aaccgg"
    payload["sequence_digest"] = _digest("aaccgg")
    for feature in payload["features"]:
        feature["sequence"] = feature["sequence"].lower()

    with pytest.raises(ValidationError, match="canonical uppercase"):
        AnnotatedSequencePartV1.model_validate(payload)


def test_annotated_sequence_part_rejects_noncanonical_lowercase_feature() -> None:
    payload = _payload()
    payload["features"][0]["sequence"] = "aa"

    with pytest.raises(ValidationError, match="canonical uppercase"):
        AnnotatedSequencePartV1.model_validate(payload)
