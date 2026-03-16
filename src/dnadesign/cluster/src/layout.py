"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/layout.py

Cluster project-layout helpers.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


class ClusterLayoutError(RuntimeError):
    """Raised when cluster layout resolution would cross ownership boundaries."""


def builtin_cluster_dir() -> Path:
    """Return the installed package's cluster directory for built-in assets."""
    return Path(__file__).resolve().parents[1]


def package_cluster_dir() -> Path:
    """Compatibility alias for the built-in package asset root."""
    return builtin_cluster_dir()


def is_builtin_cluster_path(path: Path) -> bool:
    resolved = path.resolve()
    builtin = builtin_cluster_dir()
    return resolved == builtin or resolved.is_relative_to(builtin)


def configured_workspace_cluster_dir() -> Path | None:
    value = os.environ.get("DNADESIGN_CLUSTER_ROOT")
    if not value:
        return None
    resolved = Path(value).expanduser().resolve()
    if is_builtin_cluster_path(resolved):
        raise ClusterLayoutError(
            "DNADESIGN_CLUSTER_ROOT must point to a writable workspace 'cluster/' directory, "
            f"not the built-in package tree at '{builtin_cluster_dir()}'."
        )
    return resolved


def configured_cluster_dir() -> Path | None:
    """Compatibility alias for the configured workspace cluster directory."""
    return configured_workspace_cluster_dir()


def nearest_workspace_cluster_dir(start: Path | None = None) -> Path | None:
    """
    Walk upward from ``start`` (or CWD) and return the nearest project-level
    ``cluster/`` directory. This is a project workspace lookup, not a package
    install lookup.
    """
    origin = (start or Path.cwd()).resolve()
    for base in [origin, *origin.parents]:
        candidate: Path | None = None
        if (base / "cluster").is_dir():
            candidate = (base / "cluster").resolve()
        elif base.name == "cluster":
            candidate = base.resolve()
        if candidate is None or is_builtin_cluster_path(candidate):
            continue
        return candidate
    return None


def nearest_cluster_dir(start: Path | None = None) -> Path | None:
    """Compatibility alias for workspace cluster discovery."""
    return nearest_workspace_cluster_dir(start)


def preferred_workspace_cluster_dir(start: Path | None = None) -> Path | None:
    """
    Prefer an explicit project/workspace root over ambient discovery.
    Never fall back to the installed package directory for mutable runtime state.
    """
    return configured_workspace_cluster_dir() or nearest_workspace_cluster_dir(start)


def preferred_cluster_dir(start: Path | None = None) -> Path | None:
    """Compatibility alias for preferred workspace cluster discovery."""
    return preferred_workspace_cluster_dir(start)


def _package_tree_runtime_error(origin: Path) -> ClusterLayoutError:
    return ClusterLayoutError(
        "Cluster runtime state cannot default under the built-in package tree. "
        f"Current path '{origin}' resolves inside '{builtin_cluster_dir()}'. "
        "Set DNADESIGN_CLUSTER_RESULTS_DIR, set DNADESIGN_CLUSTER_ROOT to a writable workspace "
        "'cluster/' directory, or run the command from your workspace."
    )


def default_results_root(start: Path | None = None) -> Path:
    """
    Choose a writable default results root.

    Order:
    1. nearest configured/discovered project ``cluster/`` directory
    2. current working directory ``./results`` when outside the built-in package tree
    """
    project_cluster = preferred_workspace_cluster_dir(start)
    if project_cluster is not None:
        return project_cluster / "results"
    origin = (start or Path.cwd()).resolve()
    if is_builtin_cluster_path(origin):
        raise _package_tree_runtime_error(origin)
    return origin / "results"
