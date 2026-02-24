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


def ensure_mpl_cache(catalog_root: Path) -> Path:
    """Ensure Matplotlib writes its cache under a repository-shared Cruncher cache root."""
    _quiet_font_logs()
    env_dir = os.environ.get("MPLCONFIGDIR")
    if env_dir:
        cache_dir = Path(env_dir).expanduser()
        _ensure_writable_dir(cache_dir)
        return cache_dir

    cache_dir = _default_mpl_cache_dir()
    _ensure_writable_dir(cache_dir)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    return cache_dir
