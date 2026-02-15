"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/sequence.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

_BASES = np.array(["A", "C", "G", "T"], dtype="<U1")
_REVCOMP_MAP = np.array([3, 2, 1, 0], dtype=np.int8)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Return Hamming distance between two 1-D arrays, allowing unequal lengths.

    The distance is the mismatches over the shared prefix plus the length
    difference (i.e., extra positions count as mismatches).
    """
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.ndim != 1 or arr_b.ndim != 1:
        raise ValueError("hamming_distance requires 1-D arrays")
    if arr_a.size == arr_b.size:
        return int((arr_a != arr_b).sum())
    min_len = min(arr_a.size, arr_b.size)
    return int((arr_a[:min_len] != arr_b[:min_len]).sum()) + abs(int(arr_a.size) - int(arr_b.size))


def revcomp_int(seq: np.ndarray) -> np.ndarray:
    """
    Return the reverse-complement of an integer-encoded DNA sequence.

    Encoding is assumed to be: A=0, C=1, G=2, T=3.
    """
    arr = np.asarray(seq, dtype=np.int8)
    if arr.ndim != 1:
        raise ValueError("revcomp_int requires a 1-D array")
    return _REVCOMP_MAP[arr[::-1]]


def canon_string(seq: str) -> str:
    """
    Canonicalize a DNA string by choosing the lexicographically smaller
    of the sequence and its reverse complement.
    """
    clean = seq.strip().upper()
    if not clean:
        return ""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    try:
        revcomp = "".join(comp[ch] for ch in reversed(clean))
    except KeyError as exc:
        raise ValueError(f"Invalid base in sequence '{seq}'") from exc
    return min(clean, revcomp)


def canon_int(seq: np.ndarray) -> np.ndarray:
    """
    Canonicalize an integer-encoded DNA sequence by comparing to its reverse
    complement and returning the lexicographically smaller representation.
    """
    arr = np.asarray(seq, dtype=np.int8)
    if arr.ndim != 1:
        raise ValueError("canon_int requires a 1-D array")
    rev = revcomp_int(arr)
    seq_str = "".join(_BASES[arr])
    rev_str = "".join(_BASES[rev])
    return arr.copy() if seq_str <= rev_str else rev


def identity_key(seq: str, *, bidirectional: bool) -> str:
    """
    Return the deterministic identity key for a DNA sequence.

    When bidirectional is True, this is the canonical sequence (lexicographically
    smaller of the sequence and its reverse complement). Otherwise it is the
    cleaned uppercase sequence.
    """
    if not isinstance(seq, str):
        raise TypeError("identity_key requires a DNA string")
    clean = seq.strip().upper()
    if not clean:
        return ""
    if bidirectional:
        return canon_string(clean)
    return clean


def dsdna_hamming(a: np.ndarray, b: np.ndarray) -> int:
    """
    dsDNA-aware Hamming distance: minimum of forward and reverse-complement matches.
    """
    arr_a = np.asarray(a, dtype=np.int8)
    arr_b = np.asarray(b, dtype=np.int8)
    if arr_a.ndim != 1 or arr_b.ndim != 1:
        raise ValueError("dsdna_hamming requires 1-D arrays")
    return min(hamming_distance(arr_a, arr_b), hamming_distance(revcomp_int(arr_a), arr_b))
