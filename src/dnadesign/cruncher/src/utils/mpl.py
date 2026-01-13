"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/mpl.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_NOISY_FONT_LOGGERS = ("matplotlib.font_manager", "fontTools", "fontTools.subset")


def _quiet_font_logs() -> None:
    """Reduce noisy font parsing chatter from Matplotlib/fontTools."""
    for logger_name in _NOISY_FONT_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def ensure_mpl_cache(catalog_root: Path) -> Path:
    """Ensure Matplotlib writes its cache under the project-local catalog root."""
    _quiet_font_logs()
    env_dir = os.environ.get("MPLCONFIGDIR")
    if env_dir:
        cache_dir = Path(env_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    cache_dir = catalog_root / ".mplcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    return cache_dir
