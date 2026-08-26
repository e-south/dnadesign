"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/__init__.py

Neutral cross-tool contract exports, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .folding import (  # noqa: F401
        AssessmentIntendedPairV1,
        AssessmentProducerV1,
        AssessmentStatus,
        AssessmentTargetSequenceV1,
        AssessmentTargetSequenceValueV1,
        AssessmentTargetV1,
        SecondaryStructurePredictionRequestV1,
        SecondaryStructurePredictionV2,
        StructureAssessmentPolicyV1,
        StructureAssessmentPublicationV1,
        StructureAssessmentRecordV1,
        StructureAssessmentRequestV1,
    )
    from .sequence import (  # noqa: F401
        AnnotatedSequenceFeatureV1,
        AnnotatedSequencePartV1,
        AnnotatedSequenceSourceRefV1,
        LinearSsdnaCompositionV1,
        RtPartPublicationProvenanceV1,
        RtPartPublicationV1,
        RtPartV1,
    )
    from .visual import (  # noqa: F401
        CassetteViewsManifestV1,
        CompositionReviewSvgV1,
        HairpinTopologyViewV1,
        LinearDuplexViewV1,
        SequenceEvidenceMapV1,
        ViennaRNAStructureSvgV1,
        YiuHairpinTopologyV1,
        YiuLinearStateV1,
        YiuTopologyCartoonV1,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AssessmentIntendedPairV1": (".folding", "AssessmentIntendedPairV1"),
    "AssessmentProducerV1": (".folding", "AssessmentProducerV1"),
    "AssessmentStatus": (".folding", "AssessmentStatus"),
    "AssessmentTargetSequenceV1": (".folding", "AssessmentTargetSequenceV1"),
    "AssessmentTargetSequenceValueV1": (".folding", "AssessmentTargetSequenceValueV1"),
    "AssessmentTargetV1": (".folding", "AssessmentTargetV1"),
    "AnnotatedSequenceFeatureV1": (".sequence", "AnnotatedSequenceFeatureV1"),
    "AnnotatedSequencePartV1": (".sequence", "AnnotatedSequencePartV1"),
    "AnnotatedSequenceSourceRefV1": (".sequence", "AnnotatedSequenceSourceRefV1"),
    "CassetteViewsManifestV1": (".visual", "CassetteViewsManifestV1"),
    "CompositionReviewSvgV1": (".visual", "CompositionReviewSvgV1"),
    "HairpinTopologyViewV1": (".visual", "HairpinTopologyViewV1"),
    "LinearDuplexViewV1": (".visual", "LinearDuplexViewV1"),
    "LinearSsdnaCompositionV1": (".sequence", "LinearSsdnaCompositionV1"),
    "RtPartPublicationProvenanceV1": (".sequence", "RtPartPublicationProvenanceV1"),
    "RtPartPublicationV1": (".sequence", "RtPartPublicationV1"),
    "RtPartV1": (".sequence", "RtPartV1"),
    "SecondaryStructurePredictionRequestV1": (".folding", "SecondaryStructurePredictionRequestV1"),
    "SecondaryStructurePredictionV2": (".folding", "SecondaryStructurePredictionV2"),
    "StructureAssessmentPolicyV1": (".folding", "StructureAssessmentPolicyV1"),
    "StructureAssessmentPublicationV1": (".folding", "StructureAssessmentPublicationV1"),
    "StructureAssessmentRecordV1": (".folding", "StructureAssessmentRecordV1"),
    "StructureAssessmentRequestV1": (".folding", "StructureAssessmentRequestV1"),
    "SequenceEvidenceMapV1": (".visual", "SequenceEvidenceMapV1"),
    "ViennaRNAStructureSvgV1": (".visual", "ViennaRNAStructureSvgV1"),
    "YiuHairpinTopologyV1": (".visual", "YiuHairpinTopologyV1"),
    "YiuLinearStateV1": (".visual", "YiuLinearStateV1"),
    "YiuTopologyCartoonV1": (".visual", "YiuTopologyCartoonV1"),
}

__all__ = [
    "AnnotatedSequenceFeatureV1",
    "AnnotatedSequencePartV1",
    "AnnotatedSequenceSourceRefV1",
    "AssessmentIntendedPairV1",
    "AssessmentProducerV1",
    "AssessmentStatus",
    "AssessmentTargetSequenceV1",
    "AssessmentTargetSequenceValueV1",
    "AssessmentTargetV1",
    "LinearDuplexViewV1",
    "HairpinTopologyViewV1",
    "CassetteViewsManifestV1",
    "CompositionReviewSvgV1",
    "LinearSsdnaCompositionV1",
    "RtPartPublicationProvenanceV1",
    "RtPartPublicationV1",
    "RtPartV1",
    "SecondaryStructurePredictionRequestV1",
    "SecondaryStructurePredictionV2",
    "StructureAssessmentPolicyV1",
    "StructureAssessmentPublicationV1",
    "StructureAssessmentRecordV1",
    "StructureAssessmentRequestV1",
    "SequenceEvidenceMapV1",
    "ViennaRNAStructureSvgV1",
    "YiuLinearStateV1",
    "YiuHairpinTopologyV1",
    "YiuTopologyCartoonV1",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
