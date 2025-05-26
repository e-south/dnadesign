"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/state.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal
import numpy as np

from dnadesign.cruncher.motif.model import PWM

_ALPH = np.array(["A", "C", "G", "T"], dtype="<U1")

@dataclass(frozen=True)
class SequenceState:
    seq: np.ndarray  # array of ints in {0,1,2,3}

    @staticmethod
    def random(length: int, rng) -> SequenceState:
        assert length > 0
        return SequenceState(seq=rng.integers(0, 4, size=length, dtype=int))

    @staticmethod
    def from_consensus(
        pwms: Dict[str, PWM],
        mode: Literal["shortest", "longest"],
        target_length: int,
        pad_with: Literal["background", "A", "C", "G", "T"],
        rng,
    ) -> SequenceState:
        if mode == "shortest":
            pwm = min(pwms.values(), key=lambda m: m.length)
        else:
            pwm = max(pwms.values(), key=lambda m: m.length)

        consensus = np.argmax(pwm.matrix, axis=1).astype(int)
        L0 = consensus.size
        if L0 > target_length:
            raise ValueError(f"motif length {L0} > target_length {target_length}")

        pad_total = target_length - L0
        left = pad_total // 2
        # build pad
        if pad_with == "background":
            pad = rng.integers(0, 4, size=pad_total, dtype=int)
        else:
            idx = int(np.where(_ALPH == pad_with)[0])
            pad = np.full(pad_total, idx, dtype=int)

        seq = np.concatenate([pad[:left], consensus, pad[left:]])
        return SequenceState(seq=seq)

    def seq_str(self) -> str:
        return "".join(_ALPH[self.seq])

    def seq_array(self) -> np.ndarray:
        return self.seq.copy()