"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/__init__.py

Public secondary-structure folding package exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .src import (
        FoldingConfigError,
        FoldingError,
        FoldingExecutionError,
        FoldingPreflightResult,
        enrich_prediction_pairing_qa,
        load_prediction_request,
        parse_rnafold_stdout,
        preflight_request,
        publish_viennarna_structure_svg,
        run_prediction_request,
    )

__all__ = [
    "FoldingConfigError",
    "FoldingError",
    "FoldingExecutionError",
    "FoldingPreflightResult",
    "enrich_prediction_pairing_qa",
    "load_prediction_request",
    "parse_rnafold_stdout",
    "preflight_request",
    "publish_viennarna_structure_svg",
    "run_prediction_request",
]

_API_EXPORTS = frozenset(__all__)


def __getattr__(name: str):
    if name not in _API_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".src", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
