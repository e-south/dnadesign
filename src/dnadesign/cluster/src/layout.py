"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/layout.py

Cluster project-layout helpers.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


class ClusterLayoutError(RuntimeError):
    """Raised when cluster layout resolution would cross ownership boundaries."""


def builtin_cluster_dir() -> Path:
    """Return the installed package's cluster directory for built-in assets."""
    return Path(__file__).resolve().parents[1]


def builtin_workspaces_dir() -> Path:
    """Return the built-in cluster workspace directory."""
    return builtin_cluster_dir() / "workspaces"


def is_builtin_cluster_path(path: Path) -> bool:
    resolved = path.resolve()
    builtin = builtin_cluster_dir()
    return resolved == builtin or resolved.is_relative_to(builtin)


def is_builtin_workspace_results_root(path: Path) -> bool:
    """
    Return whether ``path`` is the canonical artifact root for one checked-in workspace.

    The only package-tree runtime state cluster allows is ``workspaces/<id>/outputs/cluster``.
    """
    resolved = path.resolve()
    workspaces_root = builtin_workspaces_dir().resolve()
    if resolved == workspaces_root or not resolved.is_relative_to(workspaces_root):
        return False
    parts = resolved.relative_to(workspaces_root).parts
    return len(parts) == 3 and parts[1] == "outputs" and parts[2] == "cluster"


def _package_tree_runtime_error(origin: Path) -> ClusterLayoutError:
    return ClusterLayoutError(
        "Cluster runtime state cannot default under the built-in package tree "
        "outside an explicit workspace outputs/cluster root. "
        f"Current path '{origin}' resolves inside '{builtin_cluster_dir()}'. "
        "Pass an explicit cluster workspace or results root outside the package tree."
    )


def explicit_results_root(results_root: str | Path | None) -> Path:
    """
    Cluster artifact roots are explicit-only.
    Callers must pass a workspace-local root or a deliberate standalone results root.
    """
    if results_root is None:
        raise ClusterLayoutError("Cluster artifact roots are explicit. Pass --workspace or --results-root.")
    origin = Path(results_root).expanduser().resolve()
    if is_builtin_cluster_path(origin) and not is_builtin_workspace_results_root(origin):
        raise _package_tree_runtime_error(origin)
    return origin
