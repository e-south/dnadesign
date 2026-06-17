"""Registry for OPAL candidate eligibility rules."""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Callable
from typing import Any

from .loader import load_builtin_modules

_REGISTRY: dict[str, Callable[..., Any]] = {}
_BUILTINS_LOADED = False


def _dbg(message: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[opal.debug.eligibility] {message}", file=sys.stderr)


def register_candidate_eligibility(name: str):
    """Register a candidate eligibility rule."""

    def _wrap(fn: Callable[..., Any]):
        if name in _REGISTRY:
            raise ValueError(f"candidate eligibility rule '{name}' already registered")
        _REGISTRY[name] = fn
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


def list_candidate_eligibility_rules() -> list[str]:
    """List registered candidate eligibility rules."""

    _ensure_builtins_loaded()
    return sorted(_REGISTRY)
