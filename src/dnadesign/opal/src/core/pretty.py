"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/pretty.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from functools import lru_cache

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val == "":
        return default
    return val in _TRUTHY


@lru_cache(maxsize=1)
def rich_available() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def console_out():
    if not _truthy("OPAL_CLI_RICH", True) or not rich_available():
        return None
    from rich.console import Console

    return Console(markup=True, emoji=True, soft_wrap=False)


@lru_cache(maxsize=1)
def console_err():
    if not _truthy("OPAL_CLI_RICH", True) or not rich_available():
        return None
    from rich.console import Console

    return Console(markup=True, emoji=True, soft_wrap=False, stderr=True)


def maybe_install_rich_traceback() -> None:
    if not _truthy("OPAL_CLI_RICH", True) or not rich_available():
        return
    # Lightweight pretty tracebacks when enabled
    try:
        from rich.traceback import install

        install(show_locals=False, suppress=[__name__.rsplit(".", 1)[0]])
    except Exception:
        pass
