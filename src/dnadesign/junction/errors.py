"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/errors.py

Typed failures exposed by the junction public boundary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class JunctionError(RuntimeError):
    """Base class for actionable junction failures."""


class JunctionConfigError(JunctionError, ValueError):
    """Raised when a junction configuration violates its contract."""


class JunctionDesignError(JunctionError):
    """Raised when explicit design constraints cannot be satisfied."""


class JunctionBundleError(JunctionError):
    """Raised when a bundle cannot be published or verified."""


__all__ = [
    "JunctionBundleError",
    "JunctionConfigError",
    "JunctionDesignError",
    "JunctionError",
]
