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


class FoldingMalformedOutputError(FoldingExecutionError):
    """Raised when backend output cannot be parsed as a folding result."""


class FoldingLengthMismatchError(FoldingExecutionError):
    """Raised when backend output does not match the declared input length."""


__all__ = [
    "FoldingConfigError",
    "FoldingError",
    "FoldingExecutionError",
    "FoldingLengthMismatchError",
    "FoldingMalformedOutputError",
]
