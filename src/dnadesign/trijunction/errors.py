"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/errors.py

Typed failures exposed by the TriJunction public boundary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class TriJunctionError(RuntimeError):
    """Base class for actionable TriJunction failures."""


class TriJunctionConfigError(TriJunctionError, ValueError):
    """Raised when a TriJunction configuration violates its contract."""


class TriJunctionDesignError(TriJunctionError):
    """Raised when explicit design constraints cannot be satisfied."""


class TriJunctionBundleError(TriJunctionError):
    """Raised when a bundle cannot be published or verified."""


__all__ = [
    "TriJunctionBundleError",
    "TriJunctionConfigError",
    "TriJunctionDesignError",
    "TriJunctionError",
]
