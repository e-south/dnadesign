"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_attach_strict.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.usr import Dataset, SchemaError


def _make_dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "datasets"
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
            {"sequence": "GGGG", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
        ],
        source="test",
    )
    return ds


def test_attach_missing_id_requires_explicit_allow(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    existing_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [existing_id, "missing_id"], "score": [1.0, 2.0]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)

    with pytest.raises(SchemaError):
        ds.attach_columns(path, namespace="mock", id_col="id", columns=["score"])

    n = ds.attach_columns(path, namespace="mock", id_col="id", columns=["score"], allow_missing=True)
    assert n == 1


def test_attach_sequence_missing_requires_explicit_allow(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    df = pd.DataFrame({"sequence": ["ACGT", "TTTT"], "score": [1.0, 2.0]})
    path = tmp_path / "attach_seq.csv"
    df.to_csv(path, index=False)

    with pytest.raises(SchemaError):
        ds.attach_columns(path, namespace="mock", id_col="sequence", columns=["score"])

    n = ds.attach_columns(path, namespace="mock", id_col="sequence", columns=["score"], allow_missing=True)
    assert n == 1


def test_attach_invalid_json_is_error_by_default(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    existing_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [existing_id], "vec": ["[1,2"]})
    path = tmp_path / "attach_bad_json.csv"
    df.to_csv(path, index=False)

    with pytest.raises(SchemaError):
        ds.attach_columns(path, namespace="mock", id_col="id", columns=["vec"])

    n = ds.attach_columns(path, namespace="mock", id_col="id", columns=["vec"], parse_json=False)
    assert n == 1


def test_attach_duplicate_ids_is_error(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    existing_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [existing_id, existing_id], "score": [1.0, 2.0]})
    path = tmp_path / "attach_dup.csv"
    df.to_csv(path, index=False)

    with pytest.raises(SchemaError):
        ds.attach_columns(path, namespace="mock", id_col="id", columns=["score"])
