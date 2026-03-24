"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/errors.py

Explicit error contracts for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class CassetteError(ValueError):
    """Base error for cassette workflow failures."""


class CassetteSpecError(CassetteError):
    """Raised when a cassette spec is invalid or cannot be loaded."""


class NickaseCatalogError(CassetteError):
    """Raised when a nickase catalog is invalid or cannot be loaded."""


class CassettePlanningError(CassetteError):
    """Raised when cassette planning cannot proceed due to invariant violations."""
