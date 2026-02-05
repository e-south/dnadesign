"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_attach_sequence_keys.py

Attach sequence key constraints for bio_type and uniqueness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset
from dnadesign.usr.src.errors import SchemaError
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def test_attach_sequence_requires_single_bio_type(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit-test"},
            {"sequence": "ACGU", "bio_type": "rna", "alphabet": "rna_4", "source": "unit-test"},
        ],
        source="unit-test",
    )

    attach_path = tmp_path / "attach.parquet"
    pq.write_table(pa.table({"sequence": ["ACGT"], "score": [0.1]}), attach_path)

    with pytest.raises(SchemaError, match="single bio_type"):
        ds.attach(
            attach_path,
            namespace="mock",
            key="sequence",
            backend="duckdb",
            parse_json=False,
        )


def test_attach_sequence_ci_requires_unique_base_keys(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit-test"},
            {"sequence": "acgt", "bio_type": "dna", "alphabet": "dna_4", "source": "unit-test"},
        ],
        source="unit-test",
    )

    attach_path = tmp_path / "attach.parquet"
    pq.write_table(pa.table({"sequence": ["ACGT"], "score": [0.1]}), attach_path)

    with pytest.raises(SchemaError, match="duplicate base keys"):
        ds.attach(
            attach_path,
            namespace="mock",
            key="sequence_ci",
            backend="duckdb",
            parse_json=False,
        )
