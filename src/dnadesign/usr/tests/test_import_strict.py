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


def _init_dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "datasets"
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
