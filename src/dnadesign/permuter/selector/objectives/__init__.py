"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/objectives/__init__.py

Objective plugin registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Objective
from .weighted_sum import WeightedSumObjective

_REGISTRY: Dict[str, Type[Objective]] = {
    "weighted_sum": WeightedSumObjective,
}


def get(name: str) -> Type[Objective]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown objective: {name!r}") from exc
