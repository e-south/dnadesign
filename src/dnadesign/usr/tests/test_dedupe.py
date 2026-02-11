"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_dedupe.py

Tests for streaming dedupe behavior and key validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.errors import SchemaError
from dnadesign.usr.tests.registry_helpers import ensure_registry


def _make_dataset(tmp_path: Path) -> Dataset:
    ensure_registry(tmp_path)
    ds = Dataset(tmp_path, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "acgt", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return ds


def test_dedupe_sequence_ci_keep_first(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    with ds.maintenance(reason="dedupe"):
        stats = ds.dedupe(key="sequence_ci", keep="keep-first")
    assert stats.rows_dropped == 1
    pf = pq.ParquetFile(str(ds.records_path))
    assert pf.metadata.num_rows == 2


def test_dedupe_sequence_ci_keep_last(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    with ds.maintenance(reason="dedupe"):
        stats = ds.dedupe(key="sequence_ci", keep="keep-last")
    assert stats.rows_dropped == 1
    pf = pq.ParquetFile(str(ds.records_path))
    assert pf.metadata.num_rows == 2


def test_dedupe_sequence_ci_requires_dna4(tmp_path: Path) -> None:
    ensure_registry(tmp_path)
    ds = Dataset(tmp_path, "prot")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACDE", "bio_type": "protein", "alphabet": "protein_20", "source": "unit"},
            {"sequence": "ACDF", "bio_type": "protein", "alphabet": "protein_20", "source": "unit"},
        ]
    )
    with ds.maintenance(reason="dedupe"):
        with pytest.raises(SchemaError):
            ds.dedupe(key="sequence_ci", keep="keep-first")


def test_dedupe_invalid_key_is_error(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    with ds.maintenance(reason="dedupe"):
        with pytest.raises(SchemaError):
            ds.dedupe(key="bogus", keep="keep-first")


def test_dedupe_requires_maintenance_context(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    with pytest.raises(SchemaError):
        ds.dedupe(key="sequence_ci", keep="keep-first")
