"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_encoding.py

Stage-A core sequence encoding helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def encode_cores(cores: Sequence[str]) -> np.ndarray:
    if not cores:
        return np.empty((0, 0), dtype=np.int8)
    length = len(cores[0])
    if length <= 0 or any(len(core) != length for core in cores):
        raise ValueError("Core sequences must share a non-zero length for encoding.")
    joined = "".join(cores).encode("ascii")
    raw = np.frombuffer(joined, dtype=np.uint8)
    raw = raw.reshape(len(cores), length)
    lookup = np.full(256, 4, dtype=np.int8)
    lookup[ord("A")] = 0
    lookup[ord("C")] = 1
    lookup[ord("G")] = 2
    lookup[ord("T")] = 3
    encoded = lookup[raw]
    if np.any(encoded == 4):
        raise ValueError("Unsupported base in TFBS core; expected A/C/G/T.")
    return encoded


def encode_core(core: str) -> np.ndarray:
    return encode_cores([core])[0]
