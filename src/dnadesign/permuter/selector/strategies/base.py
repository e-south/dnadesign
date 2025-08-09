"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/strategies/base.py

Strategy base class.

A Strategy reads `objective_score` and returns elites.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
from typing import Dict, List


class Strategy(abc.ABC):
    @abc.abstractmethod
    def select(self, variants: List[Dict], *, cfg: Dict) -> List[Dict]:
        """
        Returns a new list of selected variant dicts based on `objective_score`.
        Must be deterministic and stable among ties unless explicitly configured otherwise.
        """
        ...
