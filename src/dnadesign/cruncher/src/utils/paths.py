"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/paths.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

_REPO_MARKERS = ("pyproject.toml", ".git")


def find_repo_root(anchor: Path) -> Path | None:
    for parent in (anchor, *anchor.parents):
        for marker in _REPO_MARKERS:
            if (parent / marker).exists():
                return parent
    return None


def resolve_cruncher_root(anchor: Path) -> Path | None:
    repo_root = find_repo_root(anchor)
    if repo_root is None:
        return None
    cruncher_root = repo_root / "src" / "dnadesign" / "cruncher"
    if cruncher_root.is_dir():
        return cruncher_root
    return None


def resolve_catalog_root(config_path: Path, catalog_root: Path | str) -> Path:
    root = Path(catalog_root)
    if root.is_absolute():
        return root.resolve()
    cruncher_root = resolve_cruncher_root(config_path.parent)
    if cruncher_root is None:
        raise ValueError(
            "Unable to resolve cruncher cache root from config location. Set cruncher.catalog.root to an absolute path."
        )
    resolved = (cruncher_root / root).resolve()
    try:
        resolved.relative_to(cruncher_root.resolve())
    except ValueError as exc:
        raise ValueError("cruncher.catalog.root must stay within the cruncher root.") from exc
    return resolved


def workspace_state_root(config_path: Path) -> Path:
    return config_path.parent / ".cruncher"


def resolve_lock_path(config_path: Path, *, name: str | None = None) -> Path:
    stem = name or config_path.stem
    return workspace_state_root(config_path) / "locks" / f"{stem}.lock.json"


def resolve_run_index_path(config_path: Path) -> Path:
    return workspace_state_root(config_path) / "run_index.json"
