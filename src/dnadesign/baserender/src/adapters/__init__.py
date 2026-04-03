"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/__init__.py

Adapter protocol and factory for row-to-record conversion.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .cruncher_best_window import CruncherBestWindowAdapter
from .densegen_tfbs import DensegenTfbsAdapter
from .duplex_sequence_v1 import DuplexSequenceV1Adapter
from .generic_features import GenericFeaturesAdapter
from .hairpin_topology_v1 import HairpinTopologyV1Adapter
from .registry import (
    Adapter,
    build_adapter,
    get_adapter_descriptor,
    list_adapter_descriptors,
    required_source_columns,
)
from .sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter
from .sequence_windows_v1 import SequenceWindowsV1Adapter
from .yiu_hairpin_topology_v1 import YiuHairpinTopologyV1Adapter
from .yiu_linear_state_v1 import YiuLinearStateV1Adapter
from .yiu_payload_visual_v1 import YiuPayloadVisualV1Adapter
from .yiu_topology_cartoon_v1 import YiuTopologyCartoonV1Adapter

__all__ = [
    "Adapter",
    "build_adapter",
    "list_adapter_descriptors",
    "get_adapter_descriptor",
    "required_source_columns",
    "DensegenTfbsAdapter",
    "GenericFeaturesAdapter",
    "CruncherBestWindowAdapter",
    "SequenceEvidenceMapV1Adapter",
    "SequenceWindowsV1Adapter",
    "DuplexSequenceV1Adapter",
    "HairpinTopologyV1Adapter",
    "YiuLinearStateV1Adapter",
    "YiuHairpinTopologyV1Adapter",
    "YiuPayloadVisualV1Adapter",
    "YiuTopologyCartoonV1Adapter",
]
