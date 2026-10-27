"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_scope/__init__.py

Notebook generation scope helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .resolution import resolve_notebook_run_scope
from .usr_root import resolve_notebook_usr_root

__all__ = ["resolve_notebook_run_scope", "resolve_notebook_usr_root"]
