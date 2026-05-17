"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/__init__.py

Public OPAL package API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.config.loader import load_config

__all__ = ["load_config"]
