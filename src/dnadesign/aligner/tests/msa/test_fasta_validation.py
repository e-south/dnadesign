"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/msa/test_fasta_validation.py

Module support for dnadesign.aligner.tests.msa.test_fasta_validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.aligner.msa import load_fasta_records, validate_aligned_fasta_records


def test_load_fasta_records_rejects_duplicate_ids(tmp_path: Path) -> None:
    fasta = tmp_path / "duplicate.fasta"
    fasta.write_text(">target\nACD\n>target\nACD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate FASTA record id"):
        load_fasta_records(fasta, alphabet="protein")


def test_load_fasta_records_rejects_invalid_protein_characters(tmp_path: Path) -> None:
    fasta = tmp_path / "invalid.fasta"
    fasta.write_text(">target\nACZ\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid protein character"):
        load_fasta_records(fasta, alphabet="protein")


def test_validate_aligned_fasta_records_requires_target_row() -> None:
    with pytest.raises(ValueError, match="missing target row"):
        validate_aligned_fasta_records({"other": "ACD"}, target_row_id="target")


def test_validate_aligned_fasta_records_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="one alignment length"):
        validate_aligned_fasta_records({"target": "ACD", "other": "ACDE"}, target_row_id="target")
