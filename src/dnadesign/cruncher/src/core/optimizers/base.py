"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/base.py

Author(s): Eric J. South
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

    def final_softmin_beta(self) -> float | None:
        """Return the optimizer's final soft-min beta, if applicable."""
        return None

    def final_mcmc_beta(self) -> float | None:
        """Return the optimizer's final MCMC beta, if applicable."""
        return None

    def objective_schedule_summary(self) -> Dict[str, object]:
        """Summarize objective schedule parameters for downstream consumers."""
        return {}
