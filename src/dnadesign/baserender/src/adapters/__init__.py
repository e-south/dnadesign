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
from .generic_features import GenericFeaturesAdapter
from .registry import Adapter, build_adapter, required_source_columns
from .sequence_windows_v1 import SequenceWindowsV1Adapter

__all__ = [
    "Adapter",
    "build_adapter",
    "required_source_columns",
    "DensegenTfbsAdapter",
    "GenericFeaturesAdapter",
    "CruncherBestWindowAdapter",
    "SequenceWindowsV1Adapter",
]
