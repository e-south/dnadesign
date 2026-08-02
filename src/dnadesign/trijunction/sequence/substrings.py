"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/sequence/substrings.py

Contiguous-substring and k-mer primitives for strict DNA sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .alphabet import reverse_complement, validate_dna


def longest_common_substring_length(left: str, right: str) -> int:
    """Return the length of the longest contiguous substring shared by both inputs."""

    left = validate_dna(left, name="left")
    right = validate_dna(right, name="right")
    if len(left) < len(right):
        left, right = right, left

    longest = 0
    previous = [0] * (len(right) + 1)
    for left_base in left:
        current = [0]
        for right_index, right_base in enumerate(right, start=1):
            match_length = previous[right_index - 1] + 1 if left_base == right_base else 0
            current.append(match_length)
            longest = max(longest, match_length)
        previous = current
    return longest


def kmer_set(sequence: str, k: int) -> set[str]:
    """Return the unique length-*k* contiguous substrings of *sequence*."""

    sequence = validate_dna(sequence)
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 1 or k > len(sequence):
        raise ValueError(f"k must be between 1 and the sequence length; got k={k} for sequence length {len(sequence)}")
    return {sequence[start : start + k] for start in range(len(sequence) - k + 1)}


def kmer_set_with_reverse_complements(sequence: str, k: int) -> set[str]:
    """Return sequence k-mers together with each k-mer's reverse complement."""

    forward = kmer_set(sequence, k)
    return forward | {reverse_complement(kmer) for kmer in forward}


__all__ = [
    "kmer_set",
    "kmer_set_with_reverse_complements",
    "longest_common_substring_length",
]
