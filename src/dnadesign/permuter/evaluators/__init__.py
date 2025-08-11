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
from .evo2_ll import Evo2LogLikelihoodEvaluator
from .evo2_llr import Evo2LogLikelihoodRatioEvaluator
from .placeholder import PlaceholderEvaluator

# Map short names → Evaluator subclasses
_EVAL_REGISTRY: Dict[str, Type[Evaluator]] = {
    "placeholder": PlaceholderEvaluator,
    "evo2_ll": Evo2LogLikelihoodEvaluator,
    "evo2_llr": Evo2LogLikelihoodRatioEvaluator,
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
