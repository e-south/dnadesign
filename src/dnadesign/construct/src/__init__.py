"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/__init__.py

Internal construct package exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces.api import PreflightResult, RunResult, load_job_config, preflight_from_config, run_from_config

__all__ = ["PreflightResult", "RunResult", "load_job_config", "preflight_from_config", "run_from_config"]

_API_EXPORTS = frozenset(__all__)


def __getattr__(name: str):
    if name not in _API_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".interfaces.api", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
