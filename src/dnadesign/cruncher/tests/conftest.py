"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/conftest.py

Shared pytest fixtures for cruncher tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


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
    )
    for name in env_vars:
        prior_env[name] = os.environ.get(name)
        if name in os.environ:
            del os.environ[name]

    prior_numba_cache = os.environ.get("NUMBA_CACHE_DIR")
    numba_cache_dir = Path(tempfile.mkdtemp(prefix="cruncher-numba-cache-"))
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)

    prior_string_storage = getattr(pd.options.mode, "string_storage", None)
    if prior_string_storage is not None:
        pd.options.mode.string_storage = "python"
    try:
        yield
    finally:
        if prior_string_storage is not None:
            pd.options.mode.string_storage = prior_string_storage
        if prior_numba_cache is None:
            os.environ.pop("NUMBA_CACHE_DIR", None)
        else:
            os.environ["NUMBA_CACHE_DIR"] = prior_numba_cache
        for name, value in prior_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
