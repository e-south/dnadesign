"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/bio/iupac.py

Shared DNA/IUPAC normalization and matching helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

_DNA_RE = re.compile(r"^[ACGT]+$")
_IUPAC_RE = re.compile(r"^[ACGTRYSWKMBDHVN]+$")
_IUPAC_MAP: dict[str, set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}
_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_IUPAC_COMPLEMENT = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "D": "H",
        "H": "D",
        "V": "B",
        "N": "N",
    }
)


def normalize_dna(value: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip().upper()
    if not text:
        if allow_empty:
            return ""
        raise ValueError("DNA sequence cannot be empty.")
    if not _DNA_RE.fullmatch(text):
        raise ValueError(f"DNA sequence must contain only A/C/G/T: {value!r}")
    return text


def normalize_iupac(value: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip().upper()
    if not text:
        if allow_empty:
            return ""
        raise ValueError("DNA motif cannot be empty.")
    if not _IUPAC_RE.fullmatch(text):
        raise ValueError(f"DNA motif must contain only IUPAC nucleotide symbols: {value!r}")
    return text


def reverse_complement(sequence: str) -> str:
    return normalize_dna(sequence).translate(_DNA_COMPLEMENT)[::-1]


def reverse_complement_iupac(sequence: str) -> str:
    return normalize_iupac(sequence).translate(_IUPAC_COMPLEMENT)[::-1]


def iupac_bases_for_symbol(symbol: str) -> set[str]:
    text = str(symbol or "").strip().upper()
    if len(text) != 1 or text not in _IUPAC_MAP:
        raise ValueError(f"Unknown IUPAC nucleotide symbol: {symbol!r}")
    return set(_IUPAC_MAP[text])


def iupac_symbols_compatible(left: str, right: str) -> bool:
    left_text = normalize_iupac(left)
    right_text = normalize_iupac(right)
    if len(left_text) != 1 or len(right_text) != 1:
        raise ValueError("IUPAC symbol compatibility requires single-character symbols.")
    return bool(_IUPAC_MAP[left_text] & _IUPAC_MAP[right_text])


def motif_matches(sequence: str, motif: str) -> bool:
    sequence_text = normalize_iupac(sequence)
    motif_text = normalize_iupac(motif)
    if len(sequence_text) != len(motif_text):
        return False
    return all(iupac_symbols_compatible(base, symbol) for base, symbol in zip(sequence_text, motif_text, strict=True))


def sequence_contains_iupac(sequence: str, motif: str) -> bool:
    sequence_text = normalize_iupac(sequence)
    motif_text = normalize_iupac(motif)
    window = len(motif_text)
    if window == 0 or window > len(sequence_text):
        return False
    return any(
        motif_matches(sequence_text[idx : idx + window], motif_text)
        for idx in range(0, len(sequence_text) - window + 1)
    )


def longest_reverse_complement_overlap(left: str, right: str) -> int:
    left_text = normalize_iupac(left, allow_empty=True)
    right_text = normalize_iupac(right, allow_empty=True)
    if not left_text or not right_text:
        return 0
    rc_right = reverse_complement_iupac(right_text)
    max_overlap = min(len(left_text), len(rc_right))
    for length in range(max_overlap, 0, -1):
        if motif_matches(left_text[-length:], rc_right[:length]):
            return length
    return 0
