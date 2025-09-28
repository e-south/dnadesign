"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/transforms_x.py

A minimal, explicit registry for X (representation) transforms.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Dict, List

# Registry: name -> callable(df: pd.DataFrame, ids: list[str], *, params: dict) -> (X: np.ndarray, meta: dict)
_REG_X: Dict[str, Callable] = {}


def register_transform_x(name: str):
    """Decorator to register an X transform by name."""

    def _wrap(func: Callable):
        if name in _REG_X:
            raise ValueError(f"transform_x '{name}' already registered")
        _REG_X[name] = func
        return func

    return _wrap


def get_transform_x(name: str) -> Callable:
    """Return the registered X transform callable."""
    try:
        return _REG_X[name]
    except KeyError:
        raise KeyError(f"transform_x '{name}' not found. Available: {sorted(_REG_X)}")


def list_transforms_x() -> List[str]:
    return sorted(_REG_X)
