"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/merge/__init__.py

Dataset merge helper package for maintenance-gated merge execution and overlay
carry behavior.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .execution import MergeColumnsMode, MergePolicy, MergePreview, merge_usr_to_usr
from .overlay_carry import OverlayCarryPlan, apply_overlay_carry, plan_overlay_carry

__all__ = [
    "MergeColumnsMode",
    "MergePolicy",
    "MergePreview",
    "OverlayCarryPlan",
    "apply_overlay_carry",
    "merge_usr_to_usr",
    "plan_overlay_carry",
]
