"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/_support.py

Notebook-set template builders for support OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from textwrap import dedent


def block(source: str) -> str:
    """Normalize generated marimo source fragments."""

    return dedent(source).strip("\n")


__all__ = ["block"]
