"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_data_ingestor_strict.py

Input table validation tests for DenseGen sources.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.adapters.sources import BindingSitesDataSource, SequenceLibraryDataSource


def test_binding_sites_rejects_empty_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "tfbs.csv"
    csv_path.write_text("tf,tfbs\nLexA,\n")
    ds = BindingSitesDataSource(path=str(csv_path), cfg_path=tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="Null regulator/sequence|Empty regulator/sequence"):
        ds.load_data()


def test_binding_sites_allows_duplicates(tmp_path: Path, caplog) -> None:
    csv_path = tmp_path / "tfbs.csv"
    csv_path.write_text("tf,tfbs\nLexA,AAA\nLexA,AAA\n")
    ds = BindingSitesDataSource(path=str(csv_path), cfg_path=tmp_path / "config.yaml")
    with caplog.at_level("WARNING"):
        entries, df, _summaries = ds.load_data()
    assert len(entries) == 2
    assert df.shape[0] == 2
    assert "duplicate regulator/binding-site pairs" in caplog.text.lower()


def test_binding_sites_rejects_invalid_bases(tmp_path: Path) -> None:
    csv_path = tmp_path / "tfbs.csv"
    csv_path.write_text("tf,tfbs\nLexA,AAAN\n")
    ds = BindingSitesDataSource(path=str(csv_path), cfg_path=tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="Invalid binding-site motifs"):
        ds.load_data()


def test_sequence_library_rejects_nulls(tmp_path: Path) -> None:
    csv_path = tmp_path / "seqs.csv"
    csv_path.write_text('sequence\nAAA\n""\n')
    ds = SequenceLibraryDataSource(path=str(csv_path), cfg_path=tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="Null sequences|Empty sequences"):
        ds.load_data()


def test_sequence_library_rejects_invalid_bases(tmp_path: Path) -> None:
    csv_path = tmp_path / "seqs.csv"
    csv_path.write_text("sequence\nAAAN\n")
    ds = SequenceLibraryDataSource(path=str(csv_path), cfg_path=tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="Invalid sequences"):
        ds.load_data()
