"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/nickases/errors.py

Shared explicit error contracts for normalized nickase catalog workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class NickaseCatalogError(ValueError):
    """Raised when a nickase catalog is invalid or cannot be loaded."""
