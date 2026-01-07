"""Matplotlib cache helpers."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_mpl_cache(catalog_root: Path) -> Path:
    """Ensure Matplotlib writes its cache under the project-local catalog root."""
    env_dir = os.environ.get("MPLCONFIGDIR")
    if env_dir:
        cache_dir = Path(env_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    cache_dir = catalog_root / ".mplcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    return cache_dir
