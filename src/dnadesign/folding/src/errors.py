"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/errors.py

Errors for secondary-structure folding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class FoldingError(RuntimeError):
    """Raised when a folding request cannot be completed."""


class FoldingConfigError(FoldingError):
    """Raised when a folding request or input artifact is malformed."""


class FoldingExecutionError(FoldingError):
    """Raised when a backend run fails."""


__all__ = ["FoldingConfigError", "FoldingError", "FoldingExecutionError"]
