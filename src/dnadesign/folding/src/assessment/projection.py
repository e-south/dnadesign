"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/projection.py

Deterministic low-level projections for one structure-assessment request.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.folding import (
    AssessmentTargetSequenceV1,
    AssessmentTargetSequenceValueV1,
    StructureAssessmentRequestV1,
)
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import (
    SecondaryStructurePredictionRequestV1,
)

_TARGET_SEQUENCE = "assessment-target-sequence.json"


def project_prediction_request(request: StructureAssessmentRequestV1) -> SecondaryStructurePredictionRequestV1:
    """Project one high-level request into the exact worker request contract."""
    target = request.target
    return SecondaryStructurePredictionRequestV1.model_validate(
        {
            "request_id": request.assessment_id,
            "input": {
                "sequence_id": target.sequence_id,
                "sequence_sha256": target.sequence_sha256.removeprefix("sha256:"),
                "alphabet": target.alphabet,
                "topology": "linear_ssdna",
                "length": len(target.sequence),
                "sequence_artifact": f"../{_TARGET_SEQUENCE}",
            },
            "backend": request.backend.model_dump(mode="json"),
            "policy": {
                "required": request.policy.required,
                "fail_on_malformed_output": request.policy.fail_on_malformed_output,
                "fail_on_length_mismatch": request.policy.fail_on_length_mismatch,
            },
        }
    )


def project_target_sequence(request: StructureAssessmentRequestV1) -> AssessmentTargetSequenceV1:
    """Project one high-level target into the exact worker-readable sequence artifact."""
    target = request.target
    return AssessmentTargetSequenceV1(
        sequence=AssessmentTargetSequenceValueV1(
            id=target.sequence_id,
            sha256=target.sequence_sha256.removeprefix("sha256:"),
            sequence=target.sequence,
        )
    )


__all__ = ["project_prediction_request", "project_target_sequence"]
