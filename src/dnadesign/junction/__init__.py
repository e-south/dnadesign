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
    from .api import (  # noqa: F401
        PlanSummary,
        SequenceRecord,
        build,
        load_sequence_records,
        plan,
        plot_sequence_dissimilarity,
        preflight,
        render_sequence_dissimilarity_svg,
        request_from_sequences,
        sequence_record,
        verify,
    )
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
        load_request,
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
    "SequenceRecord": (".api", "SequenceRecord"),
    "Target": (".contracts.request", "Target"),
    "JunctionBundleError": (".errors", "JunctionBundleError"),
    "JunctionConfigError": (".errors", "JunctionConfigError"),
    "JunctionDesignError": (".errors", "JunctionDesignError"),
    "JunctionError": (".errors", "JunctionError"),
    "JunctionPlan": (".contracts.plan", "JunctionPlan"),
    "JunctionRequest": (".contracts.request", "JunctionRequest"),
    "build": (".api", "build"),
    "load_request": (".contracts.request", "load_request"),
    "load_sequence_records": (".api", "load_sequence_records"),
    "parse_request": (".contracts.request", "parse_request"),
    "plan": (".api", "plan"),
    "plot_sequence_dissimilarity": (".api", "plot_sequence_dissimilarity"),
    "preflight": (".api", "preflight"),
    "render_sequence_dissimilarity_svg": (".api", "render_sequence_dissimilarity_svg"),
    "request_from_sequences": (".api", "request_from_sequences"),
    "sequence_record": (".api", "sequence_record"),
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
