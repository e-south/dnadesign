"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/strategies/__init__.py

Strategy plugin registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Strategy
from .threshold import ThresholdStrategy
from .top_k import TopKStrategy

_REGISTRY: Dict[str, Type[Strategy]] = {
    "top_k": TopKStrategy,
    "threshold": ThresholdStrategy,
}


def get(name: str) -> Type[Strategy]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown strategy: {name!r}") from exc
