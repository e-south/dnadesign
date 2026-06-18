"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/project.py

Project-scope discovery helpers for OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..dashboard.api import find_repo_root as _find_repo_root
from ..dashboard.api import list_campaign_paths as _list_campaign_paths


def find_notebook_repo_root(start: Path) -> Path | None:
    """Find the repo root for a checked-in notebook path."""

    return _find_repo_root(start)


def list_notebook_campaign_paths(repo_root: Path | None) -> list[Path]:
    """List campaign configs for project-scoped campaign notebooks."""

    return _list_campaign_paths(repo_root)


__all__ = ["find_notebook_repo_root", "list_notebook_campaign_paths"]
