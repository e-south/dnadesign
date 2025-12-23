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

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

_REGISTRY: Dict[str, Callable[..., Callable[[pd.Series], np.ndarray]]] = {}


def register_transform_x(name: str):
    """Decorator to register a factory under a stable string name."""

    def _wrap(factory: Callable[..., Callable[[pd.Series], np.ndarray]]):
        if name in _REGISTRY:
            raise ValueError(f"Duplicate transform_x: {name!r}")
        _REGISTRY[name] = factory
        return factory

    return _wrap


def list_transforms_x() -> list[str]:
    return sorted(_REGISTRY.keys())


def _assert_and_wrap(name: str, fn: Callable[[pd.Series], Any]) -> Callable[[pd.Series], np.ndarray]:
    """Wrap returned callable to enforce ndarray[float] with shape (N,F)."""

    def _wrapped(series: pd.Series) -> np.ndarray:
        X = fn(series)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != len(series):
            raise ValueError(f"transform_x[{name}] returned shape {tuple(X.shape)} for input length {len(series)}")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"transform_x[{name}] produced non-finite values.")
        return X

    return _wrapped


def get_transform_x(name: str, params: Optional[Dict[str, Any]] = None) -> Callable[[pd.Series], np.ndarray]:
    """
    Build a configured transform by name. `params` is optional for back-compat.
    Returns a callable that accepts a pandas Series and returns ndarray (N,F).
    """
    if name not in _REGISTRY:
        available = ", ".join(list_transforms_x()) or "<none registered>"
        raise ValueError(f"Unknown transform_x: {name!r}. Available: {available}")
    factory = _REGISTRY[name]
    try:
        fn = factory(params or {})
    except TypeError:
        fn = factory()  # type: ignore[misc]
    return _assert_and_wrap(name, fn)
