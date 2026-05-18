"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/compiler/exceptions.py

Fail-fast Retron MSD compiler exceptions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class RetronMsdCompilerError(ValueError):
    """Raised when MSD design-reference compilation cannot proceed safely."""


__all__ = ["RetronMsdCompilerError"]
