"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/design/randomness.py

Tool-owned deterministic random streams for reproducible search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableSequence, Sequence
from typing import TypeVar

_MASK_64 = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB

Item = TypeVar("Item")


def derive_seed(seed: int, *, pool_id: str, stage: str) -> int:
    """Derive an ordering-independent 64-bit seed for one pool stage."""

    payload = f"dnadesign.trijunction.seed.v1\0{seed}\0{pool_id}\0{stage}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class StablePrng:
    """SplitMix64 stream with repository-owned sampling semantics.

    Python's :mod:`random` module does not promise that helper algorithms such
    as ``choices`` and ``sample`` remain byte-for-byte stable across Python
    releases.  TriJunction owns this small stream and the Fisher-Yates/rejection
    algorithms below so a seed has one inspectable meaning on every supported
    runtime.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._state = seed & _MASK_64

    def next_u64(self) -> int:
        """Return the next unsigned 64-bit value from the stable stream."""

        self._state = (self._state + _SPLITMIX_INCREMENT) & _MASK_64
        value = self._state
        value = ((value ^ (value >> 30)) * _SPLITMIX_MULTIPLIER_1) & _MASK_64
        value = ((value ^ (value >> 27)) * _SPLITMIX_MULTIPLIER_2) & _MASK_64
        return (value ^ (value >> 31)) & _MASK_64

    def randbelow(self, upper_bound: int) -> int:
        """Return an unbiased integer in ``range(upper_bound)``."""

        if isinstance(upper_bound, bool) or not isinstance(upper_bound, int) or upper_bound < 1:
            raise ValueError("upper_bound must be a positive integer")
        if upper_bound > 1 << 64:
            raise ValueError("upper_bound must fit in an unsigned 64-bit draw")
        limit = (1 << 64) - ((1 << 64) % upper_bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper_bound

    def choice(self, values: Sequence[Item]) -> Item:
        """Choose one value from a non-empty finite sequence."""

        if not values:
            raise ValueError("choice requires a non-empty sequence")
        return values[self.randbelow(len(values))]

    def shuffle(self, values: MutableSequence[Item]) -> None:
        """Shuffle a mutable sequence with stable Fisher-Yates semantics."""

        for index in range(len(values) - 1, 0, -1):
            swap_index = self.randbelow(index + 1)
            values[index], values[swap_index] = values[swap_index], values[index]

    def sample(self, values: Sequence[Item], count: int) -> list[Item]:
        """Sample without replacement using a partial Fisher-Yates shuffle."""

        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(values):
            raise ValueError("sample count must be between zero and the population size")
        available = list(values)
        for index in range(count):
            swap_index = index + self.randbelow(len(available) - index)
            available[index], available[swap_index] = available[swap_index], available[index]
        return available[:count]

    def weighted_choice(self, values: Sequence[Item], weights: Sequence[int]) -> Item:
        """Choose from nonnegative integer weights without floating-point draws."""

        if not values or len(values) != len(weights):
            raise ValueError("weighted_choice requires equally sized non-empty values and weights")
        if any(isinstance(weight, bool) or not isinstance(weight, int) or weight < 0 for weight in weights):
            raise ValueError("weights must be nonnegative integers")
        total = sum(weights)
        if total < 1:
            raise ValueError("at least one weight must be positive")
        draw = self.randbelow(total)
        cumulative = 0
        for value, weight in zip(values, weights, strict=True):
            cumulative += weight
            if draw < cumulative:
                return value
        raise AssertionError("integer weighted choice exhausted a validated distribution")


__all__ = ["StablePrng", "derive_seed"]
