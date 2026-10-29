"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/security/test_opal_demo_fixture.py

Verify that OPAL's packaged demo evidence is synthetic and path-free.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.devtools.fixtures.opal_scalar_demo import FEATURE_COUNT, LABEL_COUNT, RECORD_COUNT


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[3] / "opal" / "campaigns" / "_fixtures" / "scalar-regression"


def test_opal_demo_fixture_is_small_synthetic_and_path_free() -> None:
    root = _fixture_root()
    records_path = root / "records.parquet"
    labels_path = root / "labels.csv"

    table = pq.read_table(records_path)
    labels = pd.read_csv(labels_path)

    assert table.num_rows == RECORD_COUNT
    assert table.column_names == ["id", "sequence", "bio_type", "alphabet", "fixture_kind", "X"]
    assert pa.types.is_fixed_size_list(table.schema.field("X").type)
    assert table.schema.field("X").type.list_size == FEATURE_COUNT
    assert set(table.column("fixture_kind").to_pylist()) == {"synthetic"}
    assert records_path.stat().st_size < 1_000_000

    assert labels.columns.tolist() == ["sequence", "y"]
    assert len(labels) == LABEL_COUNT
    assert all("ACCA" not in sequence for sequence in table.column("sequence").to_pylist())
    combined = records_path.read_bytes() + labels_path.read_bytes()
    assert b"/Users/" not in combined
    assert b"/scratch/" not in combined
    assert b"Dropbox" not in combined
