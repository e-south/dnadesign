"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/paths.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def resolve_workspace_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    parent = resolved.parent
    if parent.name == "configs":
        return parent.parent
    return parent


def resolve_catalog_root(config_path: Path, catalog_root: Path | str) -> Path:
    root = Path(catalog_root)
    if root.is_absolute():
        return root.resolve()
    workspace_root = resolve_workspace_root(config_path).resolve()
    resolved = (workspace_root / root).resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("cruncher.catalog.root must stay within the workspace root.") from exc
    return resolved


def workspace_state_root(config_path: Path) -> Path:
    return resolve_workspace_root(config_path) / ".cruncher"


def resolve_lock_path(config_path: Path, *, name: str | None = None) -> Path:
    stem = name or config_path.stem
    return workspace_state_root(config_path) / "locks" / f"{stem}.lock.json"


def resolve_run_index_path(config_path: Path) -> Path:
    return workspace_state_root(config_path) / "run_index.json"
