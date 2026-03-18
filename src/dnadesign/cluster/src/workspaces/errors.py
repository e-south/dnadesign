"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/workspaces/errors.py

Workspace error contracts for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class WorkspaceConfigError(RuntimeError):
    """Raised when a workspace config cannot be resolved or parsed safely."""


__all__ = ["WorkspaceConfigError"]
