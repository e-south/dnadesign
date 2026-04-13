"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/dataset_overlay_catalog.py

Public USR overlay-catalog surface for cross-tool consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.dataset_overlay_catalog import build_dataset_info, load_overlay_catalog, merge_dataset_schema

__all__ = [
    "build_dataset_info",
    "load_overlay_catalog",
    "merge_dataset_schema",
]
