"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/sequence/__init__.py

Neutral sequence-contract exports, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .linear_ssdna_composition_v1 import LinearSsdnaCompositionV1  # noqa: F401
    from .msd_design_reference_v1 import MsdDesignCatalogV1, MsdDesignReferenceV1  # noqa: F401

_LAZY_EXPORTS = {
    "LinearSsdnaCompositionV1": (".linear_ssdna_composition_v1", "LinearSsdnaCompositionV1"),
    "MsdDesignCatalogV1": (".msd_design_reference_v1", "MsdDesignCatalogV1"),
    "MsdDesignReferenceV1": (".msd_design_reference_v1", "MsdDesignReferenceV1"),
}

__all__ = ["LinearSsdnaCompositionV1", "MsdDesignCatalogV1", "MsdDesignReferenceV1"]


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
