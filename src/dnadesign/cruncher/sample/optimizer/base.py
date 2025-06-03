"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/optimizer/base.py

Abstract base class for optimizers (Gibbs & PT share the same interface).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState


class Optimizer(ABC):
    """
    Abstract interface for all optimizers in `sample/optimizer/`.
    """

    def __init__(self, scorer: Scorer, cfg: Dict, rng):
        """
        scorer:  a fully-initialized Scorer object
        cfg:     a flattened dict of all parameters (draws, tune, moves, cooling, etc.)
        rng:     numpy.random.Generator
        """
        self.scorer = scorer
        self.cfg = cfg
        self.rng = rng

    @abstractmethod
    def optimise(self, initial: SequenceState) -> List[SequenceState]:
        """
        Run the MCMC procedure and return a list of SequenceState objects,
        sorted by descending fitness.
        """
        ...
