"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/conftest.py

Shared pytest fixtures for cruncher tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Unable to locate repository root for Cruncher test cache.")


_SESSION_HOME = Path(tempfile.mkdtemp(prefix="cruncher-test-home-session-"))
_SESSION_MPLCONFIGDIR = _repo_root() / ".cache" / "matplotlib" / "cruncher"
_SESSION_NUMBA_CACHE_DIR = _SESSION_HOME / "numba_cache"
_SESSION_ORIGINAL_ENV = {
    name: os.environ.get(name) for name in ("HOME", "MPLCONFIGDIR", "ARVIZ_DATA", "NUMBA_CACHE_DIR")
}
_SESSION_ENV_CLEANED = False
_SESSION_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_SESSION_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_SESSION_HOME)
os.environ["MPLCONFIGDIR"] = str(_SESSION_MPLCONFIGDIR)
os.environ["ARVIZ_DATA"] = str(_SESSION_HOME / "arviz_data")
os.environ["NUMBA_CACHE_DIR"] = str(_SESSION_NUMBA_CACHE_DIR)


def _cleanup_session_environment() -> None:
    global _SESSION_ENV_CLEANED
    if _SESSION_ENV_CLEANED:
        return
    for name, value in _SESSION_ORIGINAL_ENV.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if "numba" in sys.modules:
        from numba.core import config as numba_config

        numba_config.CACHE_DIR = _SESSION_ORIGINAL_ENV["NUMBA_CACHE_DIR"] or ""
    shutil.rmtree(_SESSION_HOME, ignore_errors=True)
    _SESSION_ENV_CLEANED = True


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    _ = config
    _cleanup_session_environment()


atexit.register(_cleanup_session_environment)


@pytest.fixture(autouse=True, scope="function")
def _cruncher_test_environment() -> None:
    import pandas as pd

    prior_env: dict[str, str | None] = {}
    env_vars = (
        "CRUNCHER_WORKSPACE",
        "CRUNCHER_WORKSPACE_ROOTS",
        "CRUNCHER_DEFAULT_WORKSPACE",
        "CRUNCHER_CONFIG",
        "CRUNCHER_NONINTERACTIVE",
        "CRUNCHER_CWD",
        "HOME",
        "MPLCONFIGDIR",
        "ARVIZ_DATA",
        "NUMBA_CACHE_DIR",
    )
    for name in env_vars:
        prior_env[name] = os.environ.get(name)
        if name in os.environ:
            del os.environ[name]

    prior_string_storage = getattr(pd.options.mode, "string_storage", None)
    if prior_string_storage is not None:
        pd.options.mode.string_storage = "python"
    try:
        with tempfile.TemporaryDirectory(prefix="cruncher-test-home-") as tmp_home:
            home_path = Path(tmp_home)
            os.environ["HOME"] = str(home_path)
            os.environ["MPLCONFIGDIR"] = str(_SESSION_MPLCONFIGDIR)
            os.environ["ARVIZ_DATA"] = str(home_path / "arviz_data")
            os.environ["NUMBA_CACHE_DIR"] = str(_SESSION_NUMBA_CACHE_DIR)
            if "numba" in sys.modules:
                from numba.core import config as numba_config

                numba_config.CACHE_DIR = str(_SESSION_NUMBA_CACHE_DIR)
            yield
    finally:
        if prior_string_storage is not None:
            pd.options.mode.string_storage = prior_string_storage
        if "numba" in sys.modules:
            from numba.core import config as numba_config

            numba_config.CACHE_DIR = str(_SESSION_NUMBA_CACHE_DIR)
        for name, value in prior_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
