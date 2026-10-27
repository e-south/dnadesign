"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/utils/mpl_utils.py

Helpers for configuring Matplotlib cache behavior in DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


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
    return repo_root / ".cache" / "matplotlib" / "densegen"


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


def ensure_mpl_cache_dir(target: Path | str | None = None) -> Path:
    from .logging_utils import _configure_fonttools_logger

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    _configure_fonttools_logger()
    env_value = os.environ.get("MPLCONFIGDIR", "").strip()
    if env_value:
        dest = Path(env_value).expanduser()
        _ensure_writable_dir(dest)
        return dest
    dest = Path(target).expanduser() if target else _default_mpl_cache_dir()
    _ensure_writable_dir(dest)
    os.environ["MPLCONFIGDIR"] = str(dest)
    return dest
