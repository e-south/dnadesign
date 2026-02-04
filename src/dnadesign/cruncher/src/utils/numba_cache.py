"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/numba_cache.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

_NUMBA_CACHE_ENV = "NUMBA_CACHE_DIR"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


@contextlib.contextmanager
def temporary_numba_cache_dir(*, env_var: str = _NUMBA_CACHE_ENV) -> Iterator[Path]:
    prior_env = os.environ.get(env_var)
    numba_config = None
    prior_cache_dir = None
    if "numba" in sys.modules:
        try:
            from numba.core import config as numba_config
        except Exception as exc:
            raise RuntimeError("Failed to access numba config") from exc
        prior_cache_dir = numba_config.CACHE_DIR

    with tempfile.TemporaryDirectory(prefix="cruncher-numba-cache-") as raw_cache_dir:
        cache_dir = Path(raw_cache_dir).resolve()
        os.environ[env_var] = str(cache_dir)
        if numba_config is not None:
            try:
                numba_config.CACHE_DIR = str(cache_dir)
            except Exception as exc:
                raise RuntimeError(f"Failed to configure numba cache dir: {cache_dir}") from exc
        try:
            yield cache_dir
        finally:
            if prior_env is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = prior_env
            if numba_config is not None:
                numba_config.CACHE_DIR = prior_cache_dir


def ensure_numba_cache_dir(
    anchor: Path,
    *,
    cache_dir: Path | None = None,
    env_var: str = _NUMBA_CACHE_ENV,
) -> Path:
    """Ensure Numba cache directory is set and writable.

    If cache_dir is provided, use it (relative paths are resolved against anchor).
    Otherwise, env_var must be set to a writable directory.
    """
    if anchor is None:
        raise ValueError("anchor is required")
    anchor = Path(anchor).expanduser().resolve()
    if not anchor.is_dir():
        raise ValueError(f"anchor must be a directory: {anchor}")

    env_path = _env_path(env_var)
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = anchor / cache_dir
        cache_dir = cache_dir.resolve()
        if env_path is not None and env_path != cache_dir:
            raise ValueError(f"{env_var} is set to {env_path}, which conflicts with cache_dir={cache_dir}.")
    else:
        if env_path is None:
            raise RuntimeError(f"{env_var} must be set or cache_dir must be provided.")
        cache_dir = env_path
    os.environ[env_var] = str(cache_dir)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=cache_dir).close()
    except Exception as exc:
        raise RuntimeError(
            f"Numba cache directory is not writable: {cache_dir}. Set {env_var} to a writable path."
        ) from exc

    if "numba" in sys.modules:
        try:
            from numba.core import config as numba_config

            numba_config.CACHE_DIR = str(cache_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to configure numba cache dir: {cache_dir}") from exc

    return cache_dir


__all__ = ["ensure_numba_cache_dir", "temporary_numba_cache_dir"]
