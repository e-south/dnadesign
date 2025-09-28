"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/objectives.py

Objective registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Dict, List

# Registry: name -> callable(y_pred, *, params, round_ctx) -> result(score=np.ndarray, diagnostics=dict)
_REG_O: Dict[str, Callable] = {}


def register_objective(name: str):
    """Decorator to register an objective by name."""

    def _wrap(func: Callable):
        if name in _REG_O:
            raise ValueError(f"objective '{name}' already registered")
        _REG_O[name] = func
        return func

    return _wrap


def get_objective(name: str) -> Callable:
    try:
        return _REG_O[name]
    except KeyError:
        raise KeyError(f"objective '{name}' not found. Available: {sorted(_REG_O)}")


def list_objectives() -> List[str]:
    return sorted(_REG_O)
