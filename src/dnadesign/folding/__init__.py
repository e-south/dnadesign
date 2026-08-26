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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .src.api import (  # noqa: F401
        FoldingPreflightResult,
        load_prediction_request,
        preflight_request,
        run_prediction_request,
    )
    from .src.assessment import (  # noqa: F401
        PublishedStructureAssessment,
        load_published_assessment,
        publish_structure_assessment,
    )
    from .src.errors import FoldingConfigError, FoldingError, FoldingExecutionError  # noqa: F401
    from .src.rnafold import parse_rnafold_stdout  # noqa: F401
    from .src.viennarna_plot import (  # noqa: F401
        enrich_prediction_pairing_qa,
        publish_viennarna_structure_svg,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FoldingConfigError": (".src.errors", "FoldingConfigError"),
    "FoldingError": (".src.errors", "FoldingError"),
    "FoldingExecutionError": (".src.errors", "FoldingExecutionError"),
    "FoldingPreflightResult": (".src.api", "FoldingPreflightResult"),
    "PublishedStructureAssessment": (".src.assessment", "PublishedStructureAssessment"),
    "enrich_prediction_pairing_qa": (".src.viennarna_plot", "enrich_prediction_pairing_qa"),
    "load_prediction_request": (".src.api", "load_prediction_request"),
    "load_published_assessment": (".src.assessment", "load_published_assessment"),
    "parse_rnafold_stdout": (".src.rnafold", "parse_rnafold_stdout"),
    "preflight_request": (".src.api", "preflight_request"),
    "publish_viennarna_structure_svg": (".src.viennarna_plot", "publish_viennarna_structure_svg"),
    "publish_structure_assessment": (".src.assessment", "publish_structure_assessment"),
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
