"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/base.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np


class Protocol(ABC):
    @abstractmethod
    def validate_cfg(self, *, params: Dict) -> None: ...

    @abstractmethod
    def generate(
        self,
        *,
        ref_entry: Dict,
        params: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Dict]:
        """
        Yield dicts with at least:
          - 'sequence': str
          - 'modifications': list[str]
        Additional flat keys are allowed and will be namespaced by caller.
        """
        ...


def assert_dna(seq: str):
    if not isinstance(seq, str) or not re.fullmatch(r"[ACGTacgt]+", seq or ""):
        raise ValueError("Sequence must be DNA (A/C/G/T)")
