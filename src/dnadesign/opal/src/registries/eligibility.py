"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/registries/eligibility.py

Registry for OPAL candidate eligibility rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .loader import load_builtin_modules

_REGISTRY: dict[str, Callable[..., Any]] = {}
_REQUIRED_COLUMN_RESOLVERS: dict[str, Callable[[Mapping[str, Any]], Iterable[str]]] = {}
_BUILTINS_LOADED = False


def _dbg(message: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[opal.debug.eligibility] {message}", file=sys.stderr)


def register_candidate_eligibility(
    name: str,
    *,
    required_columns: Callable[[Mapping[str, Any]], Iterable[str]],
):
    """Register a candidate eligibility rule."""

    if not callable(required_columns):
        raise TypeError(f"candidate eligibility rule '{name}' required_columns must be callable")

    def _wrap(fn: Callable[..., Any]):
        if name in _REGISTRY:
            raise ValueError(f"candidate eligibility rule '{name}' already registered")
        _REGISTRY[name] = fn
        _REQUIRED_COLUMN_RESOLVERS[name] = required_columns
        return fn

    return _wrap


def _ensure_builtins_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    load_builtin_modules("dnadesign.opal.src.eligibility", label="candidate eligibility", debug=_dbg)
    _BUILTINS_LOADED = True


def _validate_rule_signature(name: str, fn: Callable[..., Any]) -> None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"candidate eligibility rule '{name}' has unreadable callable signature") from exc
    params = sig.parameters
    if "frame" not in params:
        raise TypeError(f"candidate eligibility rule '{name}' must accept keyword parameter 'frame'")
    if "params" not in params:
        raise TypeError(f"candidate eligibility rule '{name}' must accept keyword parameter 'params'")


def get_candidate_eligibility_rule(name: str) -> Callable[..., Any]:
    """Return a registered candidate eligibility rule."""

    _ensure_builtins_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"candidate eligibility rule '{name}' not found. Available: {sorted(_REGISTRY)}")
    fn = _REGISTRY[name]
    _validate_rule_signature(name, fn)
    return fn


def get_candidate_eligibility_required_columns(name: str, params: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the candidate-table columns declared by one eligibility plugin."""

    _ensure_builtins_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"candidate eligibility rule '{name}' not found. Available: {sorted(_REGISTRY)}")
    resolver = _REQUIRED_COLUMN_RESOLVERS[name]
    resolved = resolver(params)
    columns: set[str] = set()
    for value in resolved:
        column = str(value).strip()
        if not column:
            raise ValueError(f"candidate eligibility rule '{name}' declared an empty required column")
        columns.add(column)
    return tuple(sorted(columns))


def list_candidate_eligibility_rules() -> list[str]:
    """List registered candidate eligibility rules."""

    _ensure_builtins_loaded()
    return sorted(_REGISTRY)
