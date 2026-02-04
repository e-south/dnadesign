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


def _default_mpl_cache_dir() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        base = Path(cache_root)
    else:
        base = Path.home() / ".cache"
    return base / "matplotlib" / "densegen"


def ensure_mpl_cache_dir(target: Path | str | None = None) -> Path:
    from .logging_utils import _configure_fonttools_logger

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    _configure_fonttools_logger()
    if os.environ.get("MPLCONFIGDIR"):
        return Path(os.environ["MPLCONFIGDIR"])
    dest = Path(target).expanduser() if target else _default_mpl_cache_dir()
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
    os.environ["MPLCONFIGDIR"] = str(dest)
    return dest
