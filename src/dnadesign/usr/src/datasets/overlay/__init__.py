"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/overlay/__init__.py

Overlay helper package for dataset attach/write/maintenance operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .attach import _attach_frame_dataset, attach_columns_dataset, attach_dataset
from .maintenance import (
    compact_overlay_namespace,
    list_overlay_infos,
    remove_overlay_namespace,
    write_overlay_digest_ledger_namespace,
)
from .write import write_overlay_dataset, write_overlay_part_dataset

attach_frame_dataset = _attach_frame_dataset

__all__ = [
    "attach_columns_dataset",
    "attach_dataset",
    "attach_frame_dataset",
    "compact_overlay_namespace",
    "list_overlay_infos",
    "remove_overlay_namespace",
    "write_overlay_dataset",
    "write_overlay_digest_ledger_namespace",
    "write_overlay_part_dataset",
]
