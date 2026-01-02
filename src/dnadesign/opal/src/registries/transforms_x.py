"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/transforms_x.py

X-transform registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from ..core.round_context import Contract, PluginCtx

_REGISTRY: Dict[str, Callable[..., Callable[[pd.Series], np.ndarray]]] = {}
_BUILTINS_LOADED = False


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.transforms_x] {msg}", file=sys.stderr)


def _ensure_builtins_loaded() -> None:
    """Import package-shipped transforms_x modules that self-register via @register_transform_x."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    try:
        pkg = importlib.import_module("dnadesign.opal.src.transforms_x")
        _dbg(f"imported package: {pkg.__name__} ({getattr(pkg, '__file__', '?')})")
        try:
            pkg_path = pkg.__path__  # type: ignore[attr-defined]
        except Exception:
            pkg_path = []
        for mod in pkgutil.iter_modules(pkg_path):
            if mod.name.startswith("_"):
                continue
            fq = f"{pkg.__name__}.{mod.name}"
            try:
                importlib.import_module(fq)
                _dbg(f"imported built-in transform_x module: {fq}")
            except Exception as e:
                _dbg(f"FAILED importing {fq}: {e!r}")
                continue
    except Exception as e:
        _dbg(f"FAILED importing package dnadesign.opal.src.transforms_x: {e!r}")
    _BUILTINS_LOADED = True


def register_transform_x(name: str):
    """Decorator to register a factory under a stable string name."""

    def _wrap(factory: Callable[..., Callable[[pd.Series], np.ndarray]]):
        if name in _REGISTRY:
            raise ValueError(f"Duplicate transform_x: {name!r}")
        _REGISTRY[name] = factory
        return factory

    return _wrap


def list_transforms_x() -> list[str]:
    _ensure_builtins_loaded()
    return sorted(_REGISTRY.keys())


def _assert_and_wrap(
    name: str,
    fn: Callable[[pd.Series, Optional[PluginCtx]], Any],
    contract: Optional[Contract],
) -> Callable[[pd.Series, Optional[PluginCtx]], np.ndarray]:
    """Wrap returned callable to enforce ndarray[float] with shape (N,F)."""

    def _wrapped(series: pd.Series, *, ctx: Optional[PluginCtx] = None) -> np.ndarray:
        if contract is not None and ctx is None:
            raise ValueError(f"transform_x[{name}] requires ctx for contract enforcement.")
        if contract is not None and ctx is not None:
            ctx.precheck_requires()
        X = fn(series, ctx=ctx)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != len(series):
            raise ValueError(f"transform_x[{name}] returned shape {tuple(X.shape)} for input length {len(series)}")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"transform_x[{name}] produced non-finite values.")
        if contract is not None and ctx is not None:
            ctx.postcheck_produces()
        return X

    return _wrapped


def get_transform_x(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Callable[[pd.Series, Optional[PluginCtx]], np.ndarray]:
    """
    Build a configured transform by name. `params` is optional for back-compat.
    Returns a callable that accepts a pandas Series and returns ndarray (N,F).
    """
    _ensure_builtins_loaded()
    if name not in _REGISTRY:
        available = ", ".join(list_transforms_x()) or "<none registered>"
        raise ValueError(f"Unknown transform_x: {name!r}. Available: {available}")
    factory = _REGISTRY[name]
    try:
        fn = factory(params or {})
    except TypeError as e:
        raise ValueError(f"transform_x factory '{name}' must accept a params dict.") from e
    contract = getattr(factory, "__opal_contract__", None)
    wrapped = _assert_and_wrap(name, fn, contract)
    if contract is not None:
        setattr(wrapped, "__opal_contract__", contract)
    return wrapped
