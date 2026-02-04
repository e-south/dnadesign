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

from dnadesign.cruncher.utils.paths import find_repo_root, resolve_cruncher_root

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


def ensure_numba_cache_dir(anchor: Path, *, env_var: str = _NUMBA_CACHE_ENV) -> Path:
    """Ensure Numba cache directory is set and writable.

    If env_var is set, it must point to a writable directory (created if missing).
    Otherwise, use <repo>/.cruncher/numba_cache, where repo is found by walking
    up from anchor looking for pyproject.toml or .git. If no repo root is found,
    raise an error instructing the user to set env_var explicitly.
    """
    if anchor is None:
        raise ValueError("anchor is required")
    anchor = Path(anchor).expanduser().resolve()

    cache_dir = _env_path(env_var)
    if cache_dir is None:
        repo_root = find_repo_root(anchor)
        if repo_root is None:
            raise RuntimeError(
                f"Numba cache dir not set and repo root not found from {anchor}. Set {env_var} to a writable path."
            )
        cache_root = resolve_cruncher_root(anchor) or repo_root
        cache_dir = (cache_root / ".cruncher" / "numba_cache").resolve()
        os.environ[env_var] = str(cache_dir)
        logger.debug("NUMBA cache dir not set; using %s", cache_dir)

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
