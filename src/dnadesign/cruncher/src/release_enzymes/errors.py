"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/release_enzymes/errors.py

Explicit error contracts for release-enzyme catalog workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class ReleaseEnzymeCatalogError(ValueError):
    """Raised when a release-enzyme catalog is invalid or cannot be loaded."""
