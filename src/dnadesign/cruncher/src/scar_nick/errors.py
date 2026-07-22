"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/errors.py

Exceptions for the scar-nick workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class ScarNickSpecError(ValueError):
    """Raised when a scar-nick spec is invalid or cannot be resolved."""


class ScarNickPlanningError(ValueError):
    """Raised when scar-nick planning cannot continue."""


__all__ = ["ScarNickPlanningError", "ScarNickSpecError"]
