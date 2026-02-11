"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/src/normalize.py

Sequence normalization and deterministic ID hashing rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import hashlib
import re

ID_DELIMITER = "|"
ALLOWED_BIO_TYPES = {"dna", "rna", "protein"}
ALPHABETS_BY_BIO_TYPE = {
    "dna": {"dna_4", "dna_5"},
    "rna": {"rna_4", "rna_5"},
    "protein": {"protein_20", "protein_21"},
}
ALPHABET_SYMBOLS = {
    "dna_4": "ACGT",
    "dna_5": "ACGTN",
    "rna_4": "ACGU",
    "rna_5": "ACGUN",
    "protein_20": "ACDEFGHIKLMNPQRSTVWY",
    "protein_21": "ACDEFGHIKLMNPQRSTVWYX",
}
ALPHABET_PATTERNS = {name: re.compile(rf"[^{letters}{letters.lower()}]") for name, letters in ALPHABET_SYMBOLS.items()}


def validate_bio_type(bio_type: str) -> str:
    bt = str(bio_type).strip()
    if not bt:
        raise ValueError("bio_type cannot be empty.")
    if ID_DELIMITER in bt:
        raise ValueError(f"bio_type must not contain delimiter '{ID_DELIMITER}'.")
    if bt not in ALLOWED_BIO_TYPES:
        allowed = ", ".join(sorted(ALLOWED_BIO_TYPES))
        raise ValueError(f"Unsupported bio_type '{bt}'. Allowed: {allowed}.")
    return bt


def validate_alphabet(bio_type: str, alphabet: str) -> str:
    ab = str(alphabet).strip()
    if not ab:
        raise ValueError("alphabet cannot be empty.")
    allowed = ALPHABETS_BY_BIO_TYPE.get(str(bio_type))
    if not allowed:
        raise ValueError(f"Unsupported bio_type '{bio_type}'.")
    if ab not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported alphabet '{ab}' for bio_type '{bio_type}'. Allowed: {allowed_str}.")
    return ab


def normalize_sequence(seq: str, bio_type: str, alphabet: str, *, validate: bool = True) -> str:
    """
    Trim only. Preserve case (lowercase may carry information).
    Enforce alphabet constraints without coercing case.
    """
    s = (seq or "").strip()
    if validate:
        bio_type = validate_bio_type(bio_type)
        alphabet = validate_alphabet(bio_type, alphabet)
    pat = ALPHABET_PATTERNS.get(str(alphabet))
    if not pat:
        raise ValueError(f"Unsupported alphabet '{alphabet}'.")
    if pat.search(s):
        raise ValueError(f"Sequence contains invalid characters for {alphabet}.")
    return s


def compute_id(bio_type: str, sequence_norm: str) -> str:
    """
    Deterministic id: sha1(bio_type|sequence_norm).
    Case is preserved (no uppercasing).
    """
    bio_type = validate_bio_type(bio_type)
    h = hashlib.sha1()
    h.update(f"{bio_type}{ID_DELIMITER}{sequence_norm}".encode("utf-8"))
    return h.hexdigest()
