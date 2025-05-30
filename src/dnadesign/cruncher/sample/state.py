"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/state.py

SequenceState handles how we initialize and represent candidate DNA sequences
for sampler algorithms. It supports:
  - random: completely random background sequences

Each SequenceState stores its sequence as a numpy array of ints [0,1,2,3]
corresponding to A, C, G, T.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from dnadesign.cruncher.parse.model import PWM

# Index ↔ base mapping for sequence representation
_ALPH = np.array(["A", "C", "G", "T"], dtype="<U1")


@dataclass(frozen=True, slots=True)
class SequenceState:
    """
    Lightweight container for candidate sequences used by the sampler.
    Stores sequences as integer arrays (0=A,1=C,2=G,3=T) and provides helpers.
    """

    seq: np.ndarray  # 1-D int array, values in {0,1,2,3}

    @staticmethod
    def random(length: int, rng) -> "SequenceState":
        """Return a new random sequence of *length* bases drawn i.i.d. uniform."""
        if length <= 0:
            raise ValueError("length must be > 0")
        return SequenceState(seq=rng.integers(0, 4, size=length, dtype=np.int8))

    @staticmethod
    def from_consensus(
        pwms: Dict[str, PWM],
        mode: str,
        target_length: int,
        pad_with: str,
        rng,
    ) -> "SequenceState":
        """
        Construct a consensus-based sequence from a PWM.

        Parameters
        ----------
        pwms : dict of {name: PWM}
            At least one PWM must be provided; the first is used.
        mode : str
            'longest' to take the full PWM length (truncating if > target_length).
            Other values treated similarly.
        target_length : int
            Desired final sequence length; consensus is padded or truncated to match.
        pad_with : str
            How to pad if consensus shorter than target_length:
            - 'background' or 'background_pwm': random uniform draws of A/C/G/T
            - one of 'A','C','G','T': repeat that base
        rng : numpy.random.Generator
            Random number generator for padding.

        Returns
        -------
        SequenceState
        """
        # Select the first PWM
        pwm = next(iter(pwms.values()))
        # Derive consensus by argmax across PWM probability matrix rows
        consensus_ints = np.argmax(pwm.matrix, axis=1).astype(np.int8)
        L0 = consensus_ints.size

        # Truncate or pad consensus to match target_length
        if L0 >= target_length:
            seq_arr = consensus_ints[:target_length]
        else:
            pad_n = target_length - L0
            if pad_with.lower() in ("background", "background_pwm"):
                pad_arr = rng.integers(0, 4, size=pad_n, dtype=np.int8)
            else:
                base = pad_with.upper()
                idxs = np.where(_ALPH == base)[0]
                if idxs.size == 0:
                    raise ValueError(f"Unknown pad_with value '{pad_with}'")
                pad_arr = np.full(pad_n, idxs[0], dtype=np.int8)
            seq_arr = np.concatenate([consensus_ints, pad_arr])
        return SequenceState(seq=seq_arr)

    def seq_str(self) -> str:
        """Return ACGT string representation of the sequence."""
        return "".join(_ALPH[self.seq])

    def seq_array(self) -> np.ndarray:
        """Return a *copy* of the underlying integer array."""
        return self.seq.copy()
