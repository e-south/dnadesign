"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/__init__.py

Public construct package exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .src.api import (
        LinearSsdnaCompositionResult,
        LinearSsdnaCompositionSummary,
        PreflightResult,
        RunResult,
        load_job_config,
        load_linear_ssdna_composition_config,
        preflight_from_config,
        publish_composition_review_svg,
        run_from_config,
        run_linear_ssdna_composition,
        summarize_linear_ssdna_composition,
    )

__all__ = [
    "LinearSsdnaCompositionResult",
    "LinearSsdnaCompositionSummary",
    "PreflightResult",
    "RunResult",
    "load_job_config",
    "load_linear_ssdna_composition_config",
    "preflight_from_config",
    "publish_composition_review_svg",
    "run_from_config",
    "run_linear_ssdna_composition",
    "summarize_linear_ssdna_composition",
]

_API_EXPORTS = frozenset(__all__)


def __getattr__(name: str):
    if name not in _API_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".src.api", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
