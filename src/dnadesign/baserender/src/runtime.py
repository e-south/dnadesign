"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/runtime.py

Explicit runtime bootstrap for baserender contracts and effect drawers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .core import register_builtin_contracts


def _ensure_mpl_config_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return

    default_dir = Path.home() / ".matplotlib"
    if default_dir.exists() and os.access(default_dir, os.W_OK):
        return

    cache_dir = Path(tempfile.gettempdir()) / "baserender-mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def initialize_runtime() -> None:
    # Ensure Matplotlib can write cache artifacts in sandboxed/workspace environments.
    _ensure_mpl_config_dir()

    from .render.effects import register_builtin_effect_drawers

    register_builtin_contracts()
    register_builtin_effect_drawers()
