"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/methods/__init__.py

Public clustering-method surface.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import ClusteringMethod  # noqa: F401
    from .params import parse_method_param_assignments  # noqa: F401
    from .registry import (  # noqa: F401
        MethodRegistry,
        default_method_registry,
        get_method,
        register_method,
        registered_methods,
        supported_method_ids,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "ClusteringMethod": (".contracts", "ClusteringMethod"),
    "MethodRegistry": (".registry", "MethodRegistry"),
    "default_method_registry": (".registry", "default_method_registry"),
    "get_method": (".registry", "get_method"),
    "parse_method_param_assignments": (".params", "parse_method_param_assignments"),
    "register_method": (".registry", "register_method"),
    "registered_methods": (".registry", "registered_methods"),
    "supported_method_ids": (".registry", "supported_method_ids"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
