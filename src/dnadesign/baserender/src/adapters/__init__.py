"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/adapters/__init__.py

Adapter protocol and row-to-record factories, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cruncher_best_window import CruncherBestWindowAdapter  # noqa: F401
    from .densegen_tfbs import DensegenTfbsAdapter  # noqa: F401
    from .duplex_sequence_v1 import DuplexSequenceV1Adapter  # noqa: F401
    from .generic_features import GenericFeaturesAdapter  # noqa: F401
    from .hairpin_topology_v1 import HairpinTopologyV1Adapter  # noqa: F401
    from .registry import (  # noqa: F401
        Adapter,
        build_adapter,
        finalize_adapter,
        get_adapter_descriptor,
        list_adapter_descriptors,
        required_source_columns,
    )
    from .sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter  # noqa: F401
    from .sequence_windows_v1 import SequenceWindowsV1Adapter  # noqa: F401
    from .snapback_visual_v1 import SnapbackVisualV1Adapter  # noqa: F401
    from .three_way_junction_review_v1 import ThreeWayJunctionReviewV1Adapter  # noqa: F401
    from .usr_genbank_annotations_v1 import UsrGenbankAnnotationsV1Adapter  # noqa: F401
    from .yiu_hairpin_topology_v1 import YiuHairpinTopologyV1Adapter  # noqa: F401
    from .yiu_linear_state_v1 import YiuLinearStateV1Adapter  # noqa: F401
    from .yiu_payload_visual_v1 import YiuPayloadVisualV1Adapter  # noqa: F401
    from .yiu_topology_cartoon_v1 import YiuTopologyCartoonV1Adapter  # noqa: F401

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Adapter": (".registry", "Adapter"),
    "build_adapter": (".registry", "build_adapter"),
    "finalize_adapter": (".registry", "finalize_adapter"),
    "get_adapter_descriptor": (".registry", "get_adapter_descriptor"),
    "list_adapter_descriptors": (".registry", "list_adapter_descriptors"),
    "required_source_columns": (".registry", "required_source_columns"),
    "CruncherBestWindowAdapter": (".cruncher_best_window", "CruncherBestWindowAdapter"),
    "DensegenTfbsAdapter": (".densegen_tfbs", "DensegenTfbsAdapter"),
    "DuplexSequenceV1Adapter": (".duplex_sequence_v1", "DuplexSequenceV1Adapter"),
    "GenericFeaturesAdapter": (".generic_features", "GenericFeaturesAdapter"),
    "HairpinTopologyV1Adapter": (".hairpin_topology_v1", "HairpinTopologyV1Adapter"),
    "SequenceEvidenceMapV1Adapter": (".sequence_evidence_map_v1", "SequenceEvidenceMapV1Adapter"),
    "SequenceWindowsV1Adapter": (".sequence_windows_v1", "SequenceWindowsV1Adapter"),
    "SnapbackVisualV1Adapter": (".snapback_visual_v1", "SnapbackVisualV1Adapter"),
    "ThreeWayJunctionReviewV1Adapter": (
        ".three_way_junction_review_v1",
        "ThreeWayJunctionReviewV1Adapter",
    ),
    "UsrGenbankAnnotationsV1Adapter": (".usr_genbank_annotations_v1", "UsrGenbankAnnotationsV1Adapter"),
    "YiuHairpinTopologyV1Adapter": (".yiu_hairpin_topology_v1", "YiuHairpinTopologyV1Adapter"),
    "YiuLinearStateV1Adapter": (".yiu_linear_state_v1", "YiuLinearStateV1Adapter"),
    "YiuPayloadVisualV1Adapter": (".yiu_payload_visual_v1", "YiuPayloadVisualV1Adapter"),
    "YiuTopologyCartoonV1Adapter": (".yiu_topology_cartoon_v1", "YiuTopologyCartoonV1Adapter"),
}

__all__ = list(_LAZY_EXPORTS)


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
