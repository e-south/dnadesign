"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluators/__init__.py

Registry and factory for sequence scorers (“evaluators”).
You can add new classes under `dnadesign.permuter.evaluators` and register them here.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Type

from .base import Evaluator
from .placeholder import PlaceholderEvaluator

# Map short names → Evaluator subclasses
_EVAL_REGISTRY: Dict[str, Type[Evaluator]] = {
    "placeholder": PlaceholderEvaluator,
    # future evaluators go here, e.g. "evo2_7b": Evo2Evaluator,
}


@lru_cache(maxsize=None)
def get_evaluator(name: str, **params) -> Evaluator:
    """
    Instantiate (and cache) an evaluator by name.

    Args:
      name: must be a key in _EVAL_REGISTRY.
      params: passed to the evaluator's constructor.

    Raises:
      ValueError if name is unknown.
    """
    try:
        cls = _EVAL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluator: {name}") from exc
    return cls(**params)
