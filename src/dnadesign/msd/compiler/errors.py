"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/msd/compiler/errors.py

Errors raised by the public MSD compiler surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class RetronMsdCompilerError(ValueError):
    """Raised when MSD design-reference compilation cannot proceed safely."""


__all__ = ["RetronMsdCompilerError"]
