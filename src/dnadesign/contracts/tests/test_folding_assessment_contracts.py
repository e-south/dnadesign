"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/test_folding_assessment_contracts.py

Strict contracts for digest-addressed advisory structure assessments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from dnadesign.contracts.folding import (
    AssessmentIntendedPairV1,
    AssessmentTargetSequenceV1,
    AssessmentTargetV1,
)
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import SecondaryStructurePredictionV1


def _target() -> AssessmentTargetV1:
    sequence = "GCATGC"
    return AssessmentTargetV1(
        state_id="hop:encoding/example",
        state_type="hairpin_encoding_insert",
        state_schema="hop.plan/v2",
        state_digest=f"sha256:{'1' * 64}",
        sequence_id="hop:encoding/example",
        sequence_sha256=f"sha256:{hashlib.sha256(sequence.encode()).hexdigest()}",
        sequence=sequence,
        alphabet="dna",
        strandedness="not_asserted",
        topology="not_asserted",
        intended_pairs=(AssessmentIntendedPairV1(left=0, right=5),),
    )


def test_assessment_target_binds_state_identity_sequence_and_intended_pairs() -> None:
    target = _target()

    assert target.sequence == "GCATGC"
    assert target.intended_pairs[0].left == 0
    assert target.intended_pairs[0].right == 5


def test_assessment_target_rejects_sequence_digest_drift() -> None:
    payload = _target().model_dump()
    payload["sequence_sha256"] = f"sha256:{'0' * 64}"

    with pytest.raises(ValidationError, match="sequence_sha256"):
        AssessmentTargetV1.model_validate(payload)


def test_assessment_target_rejects_pair_outside_target() -> None:
    payload = _target().model_dump()
    payload["intended_pairs"] = [{"left": 0, "right": 6}]

    with pytest.raises(ValidationError, match="intended pair coordinate"):
        AssessmentTargetV1.model_validate(payload)


def test_assessment_target_artifact_rejects_its_own_sequence_digest_drift() -> None:
    with pytest.raises(ValidationError, match="artifact digest"):
        AssessmentTargetSequenceV1.model_validate(
            {
                "sequence": {
                    "id": "hop:encoding/example",
                    "sha256": "0" * 64,
                    "sequence": "GCATGC",
                }
            }
        )


def test_failed_prediction_rejects_attached_structure_result() -> None:
    with pytest.raises(ValidationError, match="result must be absent"):
        SecondaryStructurePredictionV1.model_validate(
            {
                "prediction_id": "failed-prediction",
                "status": "error",
                "input": {
                    "sequence_id": "example",
                    "sequence_sha256": "sha256:" + "1" * 64,
                    "alphabet": "dna",
                    "topology": "linear_ssdna",
                    "length": 6,
                },
                "result": {"dot_bracket": "......"},
            }
        )
