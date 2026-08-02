"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/visual/__init__.py

Neutral visual-contract exports, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cassette_views_manifest_v1 import CassetteViewsManifestV1  # noqa: F401
    from .composition_review_svg_v1 import CompositionReviewSvgV1  # noqa: F401
    from .hairpin_topology_v1 import HairpinTopologyViewV1  # noqa: F401
    from .linear_duplex_v1 import LinearDuplexViewV1  # noqa: F401
    from .scar_nick_visual_v1 import ScarNickVisualV1  # noqa: F401
    from .sequence_evidence_map_v1 import SequenceEvidenceMapV1  # noqa: F401
    from .snapback_visual_v1 import SnapbackVisualV1  # noqa: F401
    from .three_way_junction_review_v1 import ThreeWayJunctionReviewV1  # noqa: F401
    from .viennarna_secondary_structure_svg_v1 import ViennaRNAStructureSvgV1  # noqa: F401
    from .yiu_hairpin_topology_v1 import YiuHairpinTopologyV1  # noqa: F401
    from .yiu_linear_state_v1 import YiuLinearStateV1  # noqa: F401
    from .yiu_payload_visual_v1 import YiuPayloadVisualV1  # noqa: F401
    from .yiu_topology_cartoon_v1 import YiuTopologyCartoonV1  # noqa: F401

_LAZY_EXPORTS = {
    "CassetteViewsManifestV1": (".cassette_views_manifest_v1", "CassetteViewsManifestV1"),
    "CompositionReviewSvgV1": (".composition_review_svg_v1", "CompositionReviewSvgV1"),
    "HairpinTopologyViewV1": (".hairpin_topology_v1", "HairpinTopologyViewV1"),
    "LinearDuplexViewV1": (".linear_duplex_v1", "LinearDuplexViewV1"),
    "ScarNickVisualV1": (".scar_nick_visual_v1", "ScarNickVisualV1"),
    "SequenceEvidenceMapV1": (".sequence_evidence_map_v1", "SequenceEvidenceMapV1"),
    "SnapbackVisualV1": (".snapback_visual_v1", "SnapbackVisualV1"),
    "ThreeWayJunctionReviewV1": (".three_way_junction_review_v1", "ThreeWayJunctionReviewV1"),
    "ViennaRNAStructureSvgV1": (".viennarna_secondary_structure_svg_v1", "ViennaRNAStructureSvgV1"),
    "YiuHairpinTopologyV1": (".yiu_hairpin_topology_v1", "YiuHairpinTopologyV1"),
    "YiuLinearStateV1": (".yiu_linear_state_v1", "YiuLinearStateV1"),
    "YiuPayloadVisualV1": (".yiu_payload_visual_v1", "YiuPayloadVisualV1"),
    "YiuTopologyCartoonV1": (".yiu_topology_cartoon_v1", "YiuTopologyCartoonV1"),
}

__all__ = [
    "LinearDuplexViewV1",
    "HairpinTopologyViewV1",
    "CassetteViewsManifestV1",
    "CompositionReviewSvgV1",
    "ScarNickVisualV1",
    "SnapbackVisualV1",
    "ThreeWayJunctionReviewV1",
    "SequenceEvidenceMapV1",
    "ViennaRNAStructureSvgV1",
    "YiuLinearStateV1",
    "YiuHairpinTopologyV1",
    "YiuPayloadVisualV1",
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
