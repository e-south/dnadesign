"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/sequence_constraints/kmers.py

Kmer helpers for strand-aware matching on final forward-sequence coordinates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

_RC_TABLE = str.maketrans("ACGT", "TGCA")


def _normalize_dna(value: str, *, label: str) -> str:
    seq = str(value).strip().upper()
    if not seq:
        raise ValueError(f"{label} must be a non-empty DNA string.")
    if any(ch not in {"A", "C", "G", "T"} for ch in seq):
        raise ValueError(f"{label} must contain only A/C/G/T characters.")
    return seq


def reverse_complement(sequence: str) -> str:
    seq = _normalize_dna(sequence, label="sequence")
    return seq.translate(_RC_TABLE)[::-1]


def _find_all(sequence: str, pattern: str) -> list[int]:
    starts: list[int] = []
    start = 0
    while start < len(sequence):
        idx = sequence.find(pattern, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts


def find_kmer_matches(*, sequence: str, pattern: str, strands: str = "forward") -> list[dict]:
    seq = _normalize_dna(sequence, label="sequence")
    motif = _normalize_dna(pattern, label="pattern")
    mode = str(strands).strip().lower()
    if mode not in {"forward", "both"}:
        raise ValueError("strands must be 'forward' or 'both'.")

    matches: list[dict] = []
    for idx in _find_all(seq, motif):
        matches.append(
            {
                "pattern": motif,
                "strand": "+",
                "position": int(idx),
                "matched_seq": seq[idx : idx + len(motif)],
            }
        )

    if mode == "both":
        rc_seq = reverse_complement(seq)
        for idx_rc in _find_all(rc_seq, motif):
            start = len(seq) - (idx_rc + len(motif))
            matches.append(
                {
                    "pattern": motif,
                    "strand": "-",
                    "position": int(start),
                    "matched_seq": seq[start : start + len(motif)],
                }
            )

    deduped: dict[tuple[str, str, int], dict] = {}
    for match in matches:
        key = (str(match["pattern"]), str(match["strand"]), int(match["position"]))
        deduped[key] = match
    return sorted(deduped.values(), key=lambda item: (int(item["position"]), str(item["strand"]), str(item["pattern"])))
