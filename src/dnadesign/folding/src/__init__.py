"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/__init__.py

Internal secondary-structure folding package facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        FoldingPreflightResult,
        load_prediction_request,
        preflight_request,
        run_prediction_request,
    )
    from .errors import FoldingConfigError, FoldingError, FoldingExecutionError  # noqa: F401
    from .rnafold import parse_rnafold_stdout  # noqa: F401
    from .viennarna_plot import (  # noqa: F401
        enrich_prediction_pairing_qa,
        publish_viennarna_structure_svg,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FoldingConfigError": (".errors", "FoldingConfigError"),
    "FoldingError": (".errors", "FoldingError"),
    "FoldingExecutionError": (".errors", "FoldingExecutionError"),
    "FoldingPreflightResult": (".api", "FoldingPreflightResult"),
    "enrich_prediction_pairing_qa": (".viennarna_plot", "enrich_prediction_pairing_qa"),
    "load_prediction_request": (".api", "load_prediction_request"),
    "parse_rnafold_stdout": (".rnafold", "parse_rnafold_stdout"),
    "preflight_request": (".api", "preflight_request"),
    "publish_viennarna_structure_svg": (".viennarna_plot", "publish_viennarna_structure_svg"),
    "run_prediction_request": (".api", "run_prediction_request"),
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
