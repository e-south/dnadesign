"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/folding/__init__.py

Neutral folding-contract exports, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .assessment_v1 import (  # noqa: F401
        AssessmentIntendedPairV1,
        AssessmentProducerV1,
        AssessmentStatus,
        AssessmentTargetSequenceV1,
        AssessmentTargetSequenceValueV1,
        AssessmentTargetV1,
        StructureAssessmentPolicyV1,
        StructureAssessmentPublicationV1,
        StructureAssessmentRecordV1,
        StructureAssessmentRequestV1,
    )
    from .secondary_structure_prediction_v2 import (  # noqa: F401
        SecondaryStructureFailureKindV2,
        SecondaryStructureFailureV2,
        SecondaryStructurePredictionRequestV1,
        SecondaryStructurePredictionV2,
    )

__all__ = [
    "AssessmentIntendedPairV1",
    "AssessmentProducerV1",
    "AssessmentStatus",
    "AssessmentTargetSequenceV1",
    "AssessmentTargetSequenceValueV1",
    "AssessmentTargetV1",
    "SecondaryStructureFailureKindV2",
    "SecondaryStructureFailureV2",
    "SecondaryStructurePredictionRequestV1",
    "SecondaryStructurePredictionV2",
    "StructureAssessmentPolicyV1",
    "StructureAssessmentPublicationV1",
    "StructureAssessmentRecordV1",
    "StructureAssessmentRequestV1",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = (
        ".assessment_v1"
        if name.startswith("Assessment") or name.startswith("StructureAssessment")
        else ".secondary_structure_prediction_v2"
    )
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
