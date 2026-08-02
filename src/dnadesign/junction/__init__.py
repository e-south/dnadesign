"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/__init__.py

Lazy imports for the public three-way-junction planning API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import PlanSummary, build, plan, preflight, verify  # noqa: F401
    from .contracts.plan import JunctionPlan  # noqa: F401
    from .contracts.request import (  # noqa: F401
        ComplementEndPreparation,
        JunctionRequest,
        OrderPolicy,
        PlanningProfile,
        Primer,
        RecoveryPrimerMode,
        RecoveryPrimerPair,
        Target,
        parse_request,
    )
    from .errors import (  # noqa: F401
        JunctionBundleError,
        JunctionConfigError,
        JunctionDesignError,
        JunctionError,
    )
    from .publication import BundleVerification, PublishedJunctionBundle  # noqa: F401

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BundleVerification": (".publication", "BundleVerification"),
    "ComplementEndPreparation": (".contracts.request", "ComplementEndPreparation"),
    "OrderPolicy": (".contracts.request", "OrderPolicy"),
    "PlanSummary": (".api", "PlanSummary"),
    "PlanningProfile": (".contracts.request", "PlanningProfile"),
    "Primer": (".contracts.request", "Primer"),
    "PublishedJunctionBundle": (".publication", "PublishedJunctionBundle"),
    "RecoveryPrimerMode": (".contracts.request", "RecoveryPrimerMode"),
    "RecoveryPrimerPair": (".contracts.request", "RecoveryPrimerPair"),
    "Target": (".contracts.request", "Target"),
    "JunctionBundleError": (".errors", "JunctionBundleError"),
    "JunctionConfigError": (".errors", "JunctionConfigError"),
    "JunctionDesignError": (".errors", "JunctionDesignError"),
    "JunctionError": (".errors", "JunctionError"),
    "JunctionPlan": (".contracts.plan", "JunctionPlan"),
    "JunctionRequest": (".contracts.request", "JunctionRequest"),
    "build": (".api", "build"),
    "parse_request": (".contracts.request", "parse_request"),
    "plan": (".api", "plan"),
    "preflight": (".api", "preflight"),
    "verify": (".api", "verify"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
