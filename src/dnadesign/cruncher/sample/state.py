"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/state.py

SequenceState handles how we initialize and represent candidate DNA sequences
for sampler algorithms. It supports:
  - random: completely random background sequences
  - consensus_shortest / consensus_longest:
      - embed a PWM consensus at a random location
      - pad the flanks with either
          - uniform random bases
          - PWM-derived base frequencies (reflecting GC/AT bias)
          - a fixed nucleotide (e.g. "A" repeat)

Each SequenceState stores its sequence as a numpy array of ints [0,1,2,3]
corresponding to A, C, G, T.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from dnadesign.cruncher.motif.model import PWM

# Map integer codes back to bases for human-readable output
_ALPH = np.array(["A", "C", "G", "T"], dtype="<U1")


@dataclass(frozen=True)
class SequenceState:
    seq: np.ndarray  # 1D array of ints in {0,1,2,3}

    @staticmethod
    def random(length: int, rng) -> SequenceState:
        """
        Create a completely random sequence of given length.
        Each position is drawn iid from {A,C,G,T} uniformly.

        Args:
            length: desired sequence length (must be >0)
            rng: numpy.random.Generator for reproducibility
        Returns:
            SequenceState wrapping the random array
        """
        assert length > 0, "Length must be positive"
        # rng.integers(0,4) yields values 0,1,2,3 corresponding to A,C,G,T
        random_seq = rng.integers(0, 4, size=length, dtype=int)
        return SequenceState(seq=random_seq)

    @staticmethod
    def from_consensus(
        pwms: Dict[str, PWM],
        mode: Literal["shortest", "longest"],
        target_length: int,
        pad_with: Literal["background", "background_pwm", "A", "C", "G", "T"],
        rng,
    ) -> SequenceState:
        """
        Initialize a sequence by embedding the consensus of one PWM,
        then padding both sides to reach `target_length`.

        Steps:
          1) Pick the PWM with shortest or longest motif length.
          2) Compute its consensus: at each column, take the base with max probability.
          3) Choose a random insertion point: uniformly pick left padding size.
          4) Pad flanks:
             - "background": uniform iid bases
             - "background_pwm": sample according to overall PWM base frequencies
             - fixed base ("A"/"C"/"G"/"T"): repeat that base

        Args:
            pwms: dict mapping TF name to PWM object
            mode: "shortest" or "longest" to select which PWM
            target_length: final sequence length (>= motif length)
            pad_with: how to fill outside the consensus region
            rng: numpy.random.Generator for reproducibility

        Returns:
            A new SequenceState with the embedded consensus.
        """
        # 1) choose PWM by length
        pwm = (min if mode == "shortest" else max)(pwms.values(), key=lambda m: m.length)
        # consensus: int codes for most likely base at each position
        consensus = np.argmax(pwm.matrix, axis=1).astype(int)
        motif_len = consensus.size

        if motif_len > target_length:
            raise ValueError(f"Motif length {motif_len} > target_length {target_length}")

        # 2) determine how many bases remain for padding
        total_pad = target_length - motif_len
        # choose left pad size uniformly from [0, total_pad]
        left_pad = rng.integers(0, total_pad + 1)
        right_pad = total_pad - left_pad

        # 3) build padding arrays
        if pad_with == "background":
            # uniform random among {A,C,G,T}
            left_arr = rng.integers(0, 4, size=left_pad)
            right_arr = rng.integers(0, 4, size=right_pad)

        elif pad_with == "background_pwm":
            # compute overall base frequencies from all PWMs
            counts = np.zeros(4, dtype=float)
            total_positions = 0
            for motif in pwms.values():
                # sum probabilities per column
                counts += motif.matrix.sum(axis=0)
                total_positions += motif.length
            # compute overall base frequencies and renormalize
            freqs = counts / total_positions
            freqs = freqs / freqs.sum()  # ensure sum==1
            left_arr = rng.choice(4, size=left_pad, p=freqs)
            right_arr = rng.choice(4, size=right_pad, p=freqs)

        else:
            # fixed base padding
            base_idx = int(np.where(_ALPH == pad_with)[0])
            left_arr = np.full(left_pad, base_idx, dtype=int)
            right_arr = np.full(right_pad, base_idx, dtype=int)

        # 4) concatenate padding + consensus + padding
        full_seq = np.concatenate([left_arr, consensus, right_arr])
        return SequenceState(seq=full_seq)

    def seq_str(self) -> str:
        """
        Convert the internal int array back to a string of ACGT.
        """
        return "".join(_ALPH[self.seq])

    def seq_array(self) -> np.ndarray:
        """
        Return a copy of the internal numpy array for downstream processing.
        """
        return self.seq.copy()
