"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/core/optimizers/base.py

Abstract base class for optimizers (Gibbs & PT share the same interface).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dnadesign.cruncher.core.state import SequenceState


class Optimizer(ABC):
    """
    Abstract interface for all optimizers in `core/optimizers/`.
    """

    def __init__(self, evaluator: Any, cfg: Dict, rng):
        """
        evaluator:  sequence evaluator used to score sequences
        cfg:     a flattened dict of all parameters (draws, tune, moves, cooling, etc.)
        rng:     numpy.random.Generator
        """
        self.scorer = evaluator
        self.cfg = cfg
        self.rng = rng

    @abstractmethod
    def optimise(self) -> List[SequenceState]:
        """
        Run the MCMC procedure and return a list of SequenceState objects,
        sorted by descending fitness.
        """
        ...
