"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/errors.py

Explicit error contracts for the snapback workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class SnapbackError(ValueError):
    """Base error for snapback workflow failures."""


class SnapbackSpecError(SnapbackError):
    """Raised when a snapback spec is invalid or cannot be loaded."""


class SnapbackPlanningError(SnapbackError):
    """Raised when snapback planning cannot proceed due to invariant violations."""
