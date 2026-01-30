"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_encoding.py

Stage-A core sequence encoding helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
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


def _encoding_key(cores: Sequence[str]) -> str:
    hasher = hashlib.md5()
    hasher.update(str(len(cores)).encode("utf-8"))
    for core in cores:
        hasher.update(b"\0")
        hasher.update(str(core).encode("utf-8"))
    return hasher.hexdigest()


@dataclass
class CoreEncodingStore:
    max_entries: int = 8
    _cache: OrderedDict[str, np.ndarray] = field(default_factory=OrderedDict, init=False)

    def encode(self, cores: Sequence[str]) -> np.ndarray:
        if not cores:
            return encode_cores(cores)
        key = _encoding_key(cores)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        encoded = encode_cores(cores)
        self._cache[key] = encoded
        self._cache.move_to_end(key)
        if len(self._cache) > int(self.max_entries):
            self._cache.popitem(last=False)
        return encoded
