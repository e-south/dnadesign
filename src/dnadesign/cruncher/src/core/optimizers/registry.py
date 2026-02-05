"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from dnadesign.cruncher.core.optimizers.base import Optimizer
from dnadesign.cruncher.core.optimizers.pt import PTGibbsOptimizer

OptimizerFactory = Callable[..., Optimizer]


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    description: str


_REGISTRY: Dict[str, OptimizerFactory] = {}
_DESCRIPTIONS: Dict[str, str] = {}


def register_optimizer(name: str, factory: OptimizerFactory, description: str = "") -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("optimizer name must be non-empty")
    if key in _REGISTRY:
        raise ValueError(f"optimizer '{key}' is already registered")
    _REGISTRY[key] = factory
    _DESCRIPTIONS[key] = description


def get_optimizer(name: str) -> OptimizerFactory:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def list_optimizers() -> list[str]:
    return sorted(_REGISTRY)


def list_optimizer_specs() -> list[OptimizerSpec]:
    return [OptimizerSpec(name=key, description=_DESCRIPTIONS.get(key, "")) for key in sorted(_REGISTRY)]


# Built-ins
register_optimizer("pt", PTGibbsOptimizer, "Parallel-tempered Gibbs sampler with swap diagnostics.")
