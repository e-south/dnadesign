from __future__ import annotations

import hashlib

import pytest

from dnadesign.densegen.src.core.canonical import compute_id, normalize_sequence


def test_compute_id_matches_expected() -> None:
    seq = "AaTG"
    bio_type = "dna"
    norm = normalize_sequence(seq, bio_type, "dna_4")
    expected = hashlib.sha1(f"{bio_type}|{norm}".encode("utf-8")).hexdigest()
    assert compute_id(bio_type, norm) == expected


def test_normalize_rejects_non_acgt() -> None:
    with pytest.raises(ValueError):
        normalize_sequence("ATGX", "dna", "dna_4")
