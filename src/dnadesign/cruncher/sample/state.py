"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/state.py

SequenceState: represents one candidate DNA sequence in integer form (0=A,1=C,2=G,3=T).
This module also provides a simple “consensus” seeding factory.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from dnadesign.cruncher.parse.model import PWM

# Fixed array of characters, used to convert numeric-encoded sequences back to strings.
_ALPH = np.array(["A", "C", "G", "T"], dtype="<U1")


@dataclass(frozen=True, slots=True)
class SequenceState:
    """
    Immutable container for a DNA sequence encoded as a 1-D numpy array of ints (0=A,1=C,2=G,3=T).

    Attributes:
      seq: 1-D numpy.ndarray of shape (L,), dtype=np.int8, with values in {0,1,2,3}.
    """

    seq: np.ndarray  # shape = (L,), dtype = np.int8

    @staticmethod
    def random(length: int, rng: np.random.Generator) -> "SequenceState":
        """
        Create a completely random DNA sequence of given length, sampling each base uniformly.

        Args:
          length: Desired sequence length (must be ≥ 1).
          rng:     numpy.random.Generator instance.

        Returns:
          SequenceState containing `length` bases sampled uniformly from A,C,G,T.
        """
        if length < 1:
            raise ValueError(f"Cannot create a SequenceState of non-positive length {length}")
        arr = rng.integers(0, 4, size=length, dtype=np.int8)
        return SequenceState(arr)

    @staticmethod
    def from_consensus(
        pwm: PWM, length: int, rng: np.random.Generator, *, pad_with: str = "background"
    ) -> "SequenceState":
        """
        Create a sequence by embedding the PWM's “consensus” (argmax at each column)
        somewhere inside a total-length = `length`. Pads/truncates as needed.
        """
        if length < 1:
            raise ValueError(f"Cannot create a SequenceState of non-positive length {length}")

        # 1) Build consensus_vec: argmax over each row of pwm.matrix → shape = (w,)
        cons = np.argmax(pwm.matrix, axis=1).astype(np.int8)
        w = cons.size

        # 2) If PWM width ≥ desired length, center-truncate consensus
        if w >= length:
            start = (w - length) // 2
            truncated = cons[start : start + length]
            return SequenceState(truncated)

        # 3) Else: embed consensus_vec in a background of total size = length
        pad_n = length - w
        if pad_with == "background":
            full_pad = rng.integers(0, 4, size=pad_n, dtype=np.int8)
        else:
            pad_base = pad_with.upper()
            if pad_base not in ("A", "C", "G", "T"):
                raise ValueError(f"Invalid pad_with '{pad_with}'; must be 'background' or one of 'A','C','G','T'")
            idx_base = int(np.where(_ALPH == pad_base)[0])
            full_pad = np.full(pad_n, idx_base, dtype=np.int8)

        # 3a) Choose random insertion offset
        offs = rng.integers(0, pad_n + 1)
        prefix_pad = full_pad[:offs]
        suffix_pad = full_pad[offs:]

        seq_arr = np.concatenate([prefix_pad, cons, suffix_pad])
        return SequenceState(seq_arr)

    def to_string(self) -> str:
        """
        Render this SequenceState as a string of “A/C/G/T” for human readability.
        """
        return "".join(_ALPH[self.seq])

    def copy(self) -> SequenceState:
        """
        Return a new SequenceState with a copied underlying numpy array.
        """
        return SequenceState(self.seq.copy())

    def __len__(self) -> int:
        """
        Return the integer length L of this sequence.
        """
        return int(self.seq.size)


def make_seed(cfg_init: Any, pwms: Dict[str, PWM], rng: np.random.Generator) -> SequenceState:
    """
    Return an initial SequenceState based on cfg_init.kind. Three modes:

      • “random”: uniform-random DNA sequence of length = cfg_init.length.

      • “consensus”: embed one PWM's consensus (argmax) into background of length = cfg_init.length.

      • “consensus_mix”: randomly choose “random” or choose one PWM's consensus to embed.
    """
    kind = getattr(cfg_init, "kind", None)
    length = int(getattr(cfg_init, "length", 0))
    if length < 1:
        raise ValueError(f"cfg_init.length must be ≥ 1, got {length}")

    if kind == "random":
        return SequenceState.random(length, rng)

    if kind == "consensus":
        pwm_name = getattr(cfg_init, "regulator", None)
        if pwm_name is None:
            raise ValueError("For init.kind=='consensus', you must supply init.regulator=<PWM_name>.")
        if pwm_name not in pwms:
            raise KeyError(
                f"PWM '{pwm_name}' not found in loaded regulator_sets. Available PWMs: {sorted(pwms.keys())}"
            )
        pwm = pwms[pwm_name]
        pad_with = getattr(cfg_init, "pad_with", "background")
        return SequenceState.from_consensus(pwm, length, rng, pad_with=pad_with)

    if kind == "consensus_mix":
        choices: list[str] = ["random"] + list(pwms.keys())
        pick = rng.choice(choices)
        if pick == "random":
            return SequenceState.random(length, rng)
        pwm = pwms[pick]
        pad_with = getattr(cfg_init, "pad_with", "background")
        return SequenceState.from_consensus(pwm, length, rng, pad_with=pad_with)

    raise ValueError(f"Unknown init.kind '{kind}'. Must be 'random' | 'consensus' | 'consensus_mix'.")
