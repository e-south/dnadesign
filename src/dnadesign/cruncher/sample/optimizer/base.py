"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/optimizer/base.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
from dnadesign.cruncher.sample.state import SequenceState
from dnadesign.cruncher.sample.scorer import Scorer

class Optimizer(ABC):
    def __init__(self, scorer: Scorer, cfg: Dict, rng):
        self.scorer, self.cfg, self.rng = scorer, cfg, rng

    @abstractmethod
    def optimise(self, initial: SequenceState) -> List[SequenceState]:
        """Return a ranked list (best first) of unique OverlapState objects."""
