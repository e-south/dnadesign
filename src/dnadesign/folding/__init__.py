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
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FoldingConfigError": (".src.errors", "FoldingConfigError"),
    "FoldingError": (".src.errors", "FoldingError"),
    "FoldingExecutionError": (".src.errors", "FoldingExecutionError"),
    "FoldingPreflightResult": (".src.api", "FoldingPreflightResult"),
    "enrich_prediction_pairing_qa": (".src.viennarna_plot", "enrich_prediction_pairing_qa"),
    "load_prediction_request": (".src.api", "load_prediction_request"),
    "parse_rnafold_stdout": (".src.rnafold", "parse_rnafold_stdout"),
    "preflight_request": (".src.api", "preflight_request"),
    "publish_viennarna_structure_svg": (".src.viennarna_plot", "publish_viennarna_structure_svg"),
    "run_prediction_request": (".src.api", "run_prediction_request"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
