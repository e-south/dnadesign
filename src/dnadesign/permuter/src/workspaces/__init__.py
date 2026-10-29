"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/workspaces/__init__.py

Workspace facade for Permuter config discovery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.permuter.src.workspaces.contracts import PermuterWorkspace
from dnadesign.permuter.src.workspaces.loader import (
    find_workspaces,
    load_workspace,
)

__all__ = [
    "PermuterWorkspace",
    "find_workspaces",
    "load_workspace",
]
