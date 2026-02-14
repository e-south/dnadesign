"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/arviz_cache.py

Configure a writable ArviZ data directory for runtime imports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()
        return True
    except Exception:
        return False


def _suppress_arviz_refactor_future_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"ArviZ is undergoing a major refactor.*",
        module=r"^arviz(\..*)?$",
    )


def ensure_arviz_data_dir(catalog_root: Path) -> Path:
    """Ensure ArviZ can write its daily warning stamp before import."""
    catalog_root = Path(catalog_root).expanduser().resolve()
    _suppress_arviz_refactor_future_warning()
    runtime_home = (catalog_root / ".runtime_home").resolve()
    home_warning_dir = Path.home() / "arviz_data"
    if not _is_writable_dir(home_warning_dir):
        logger.warning("HOME is not writable for ArviZ cache; overriding HOME with %s.", runtime_home)
        runtime_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(runtime_home)
        home_warning_dir = runtime_home / "arviz_data"
        if not _is_writable_dir(home_warning_dir):
            raise RuntimeError(f"ArviZ home cache directory is not writable: {home_warning_dir}")

    target = (catalog_root / ".arviz_data").resolve()
    env_dir = os.environ.get("ARVIZ_DATA")
    if env_dir:
        env_path = Path(env_dir).expanduser().resolve()
        if env_path != target:
            logger.warning("ARVIZ_DATA is set to %s; overriding with %s.", env_path, target)
    if not _is_writable_dir(target):
        raise RuntimeError(f"ArviZ data directory is not writable: {target}")
    os.environ["ARVIZ_DATA"] = str(target)
    return home_warning_dir


__all__ = ["ensure_arviz_data_dir"]
