"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_import_strict.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.usr import AlphabetError, Dataset, SchemaError
from dnadesign.usr.tests.registry_helpers import ensure_registry


def _init_dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "datasets"
    ensure_registry(root)
    ds = Dataset(root, "demo")
    ds.init(source="test")
    return ds


def test_import_missing_sequence_is_error(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    rows = [
        {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4"},
        {"sequence": "", "bio_type": "dna", "alphabet": "dna_4"},
    ]
    with pytest.raises(SchemaError):
        ds.import_rows(rows)


def test_import_missing_bio_type_is_error(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    df = pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA"],
            "bio_type": ["dna", None],
            "alphabet": ["dna_4", "dna_4"],
        }
    )
    with pytest.raises(SchemaError):
        ds.import_rows(df)


def test_import_missing_alphabet_is_error(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    df = pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", ""],
        }
    )
    with pytest.raises(SchemaError):
        ds.import_rows(df)


def test_import_invalid_sequence_reports_alphabet_error(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    rows = [{"sequence": "NNNN", "bio_type": "dna", "alphabet": "dna_4"}]
    with pytest.raises(AlphabetError):
        ds.import_rows(rows)


def test_import_invalid_bio_type_is_error(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    rows = [{"sequence": "ACGT", "bio_type": "DNA", "alphabet": "dna_4"}]
    with pytest.raises(SchemaError):
        ds.import_rows(rows)


def test_import_invalid_alphabet_for_bio_type_is_error(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    rows = [{"sequence": "ACGT", "bio_type": "dna", "alphabet": "protein_20"}]
    with pytest.raises(AlphabetError):
        ds.import_rows(rows)


def test_import_rna_and_protein_sequences(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    rows = [
        {"sequence": "ACGU", "bio_type": "rna", "alphabet": "rna_4"},
        {"sequence": "ACDEFGHIKLMNPQRSTVWY", "bio_type": "protein", "alphabet": "protein_20"},
    ]
    n = ds.import_rows(rows)
    assert n == 2


def test_import_rna_rejects_thymine(tmp_path: Path) -> None:
    ds = _init_dataset(tmp_path)
    rows = [{"sequence": "ACGT", "bio_type": "rna", "alphabet": "rna_4"}]
    with pytest.raises(AlphabetError):
        ds.import_rows(rows)
