"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/evaluators/base.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class Evaluator(ABC):
    def __init__(self, **_: object) -> None:
        # Accept arbitrary kwargs so concrete evaluators can pass their knobs
        # without coupling the base to implementation details.
        pass

    @abstractmethod
    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence: Optional[str] = None,
        ref_embedding: Optional[np.ndarray] = None,
    ) -> List[float]: ...
