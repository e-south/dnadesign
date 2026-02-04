"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/ingest/validators.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

from ..errors import ValidationError

_DNA_ACGT = set("ACGTacgt")
_DNA_IUPAC = set("RYSWKMBDHVNryswkmbdhvn")

_AA_20 = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")
_AA_EXT = set("XBZOUxbzou")  # extended tokens allowed optionally


def validate_dna(seqs: Iterable[str], *, allow_iupac: bool = False) -> None:
    for i, s in enumerate(seqs):
        if not isinstance(s, str) or not s:
            raise ValidationError(f"Invalid DNA at index {i}: must be non-empty string")
        letters = set(s)
        if not letters.issubset(_DNA_ACGT | (_DNA_IUPAC if allow_iupac else set())):
            raise ValidationError(f"Invalid DNA alphabet at index {i}")


def validate_protein(seqs: Iterable[str], *, allow_extended_aas: bool = False) -> None:
    for i, s in enumerate(seqs):
        if not isinstance(s, str) or not s:
            raise ValidationError(f"Invalid protein at index {i}: must be non-empty string")
        letters = set(s)
        allowed = _AA_20 | (_AA_EXT if allow_extended_aas else set())
        if not letters.issubset(allowed):
            raise ValidationError(f"Invalid protein alphabet at index {i}")
