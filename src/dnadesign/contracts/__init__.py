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
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CassetteViewsManifestV1": (".visual", "CassetteViewsManifestV1"),
    "CompositionReviewSvgV1": (".visual", "CompositionReviewSvgV1"),
    "HairpinTopologyViewV1": (".visual", "HairpinTopologyViewV1"),
    "LinearDuplexViewV1": (".visual", "LinearDuplexViewV1"),
    "LinearSsdnaCompositionV1": (".sequence", "LinearSsdnaCompositionV1"),
    "MsdDesignCatalogV1": (".sequence", "MsdDesignCatalogV1"),
    "MsdDesignReferenceV1": (".sequence", "MsdDesignReferenceV1"),
    "SecondaryStructurePredictionRequestV1": (".folding", "SecondaryStructurePredictionRequestV1"),
    "SecondaryStructurePredictionV1": (".folding", "SecondaryStructurePredictionV1"),
    "SequenceEvidenceMapV1": (".visual", "SequenceEvidenceMapV1"),
    "ViennaRNAStructureSvgV1": (".visual", "ViennaRNAStructureSvgV1"),
    "YiuHairpinTopologyV1": (".visual", "YiuHairpinTopologyV1"),
    "YiuLinearStateV1": (".visual", "YiuLinearStateV1"),
    "YiuTopologyCartoonV1": (".visual", "YiuTopologyCartoonV1"),
}

__all__ = [
    "LinearDuplexViewV1",
    "HairpinTopologyViewV1",
    "CassetteViewsManifestV1",
    "CompositionReviewSvgV1",
    "LinearSsdnaCompositionV1",
    "MsdDesignCatalogV1",
    "MsdDesignReferenceV1",
    "SecondaryStructurePredictionRequestV1",
    "SecondaryStructurePredictionV1",
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
