"""
Shared pytest fixtures for cruncher tests.
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.cli.config_resolver import (
    CONFIG_ENV_VAR,
    DEFAULT_WORKSPACE_ENV_VAR,
    NONINTERACTIVE_ENV_VAR,
    WORKSPACE_ENV_VAR,
    WORKSPACE_ROOTS_ENV_VAR,
)


@pytest.fixture(autouse=True)
def _clear_cruncher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        CONFIG_ENV_VAR,
        WORKSPACE_ENV_VAR,
        DEFAULT_WORKSPACE_ENV_VAR,
        WORKSPACE_ROOTS_ENV_VAR,
        NONINTERACTIVE_ENV_VAR,
    ):
        monkeypatch.delenv(var, raising=False)
