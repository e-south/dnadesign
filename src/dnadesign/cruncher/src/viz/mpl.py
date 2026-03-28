"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/viz/mpl.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from pathlib import Path

_NOISY_FONT_LOGGERS = (
    "matplotlib.font_manager",
    "matplotlib.category",
    "fontTools",
    "fontTools.subset",
)


def _quiet_font_logs() -> None:
    """Reduce noisy font parsing chatter from Matplotlib/fontTools."""
    for logger_name in _NOISY_FONT_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def _default_mpl_cache_dir() -> Path:
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is None:
        raise RuntimeError("Unable to determine repository root for Matplotlib cache. Set MPLCONFIGDIR explicitly.")
    return repo_root / ".cache" / "matplotlib" / "cruncher"


def _ensure_writable_dir(dest: Path) -> None:
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to create matplotlib cache dir: {dest}") from exc
    probe = dest / ".write_test"
    try:
        probe.write_text("")
    except Exception as exc:
        raise RuntimeError(f"Matplotlib cache dir is not writable: {dest}") from exc
    else:
        probe.unlink(missing_ok=True)


def bind_mpl_config_dir(
    cache_dir: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Bind `MPLCONFIGDIR` to a writable location if the caller has not set one."""
    _quiet_font_logs()
    target_env = os.environ if environ is None else environ
    env_dir = str(target_env.get("MPLCONFIGDIR", "")).strip()
    if env_dir:
        resolved = Path(env_dir).expanduser()
        _ensure_writable_dir(resolved)
        return resolved

    resolved = cache_dir.expanduser().resolve()
    _ensure_writable_dir(resolved)
    target_env["MPLCONFIGDIR"] = str(resolved)
    return resolved


def workspace_mpl_cache_dir(workspace_root: Path) -> Path:
    return workspace_root.expanduser().resolve() / ".cruncher" / ".runtime_mplconfig"


def ensure_workspace_mpl_cache(
    workspace_root: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    return bind_mpl_config_dir(workspace_mpl_cache_dir(workspace_root), environ=environ)


def infer_workspace_root_from_output_artifact(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if candidate.name == "outputs":
            return candidate.parent
    return None


def ensure_mpl_cache(catalog_root: Path) -> Path:
    """Ensure Matplotlib writes its cache under a repository-shared Cruncher cache root."""
    _ = catalog_root
    return bind_mpl_config_dir(_default_mpl_cache_dir())
