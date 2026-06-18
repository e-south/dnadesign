"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/scripts/test_refresh_overlay_registry_metadata.py

Regression tests for refresh overlay registry metadata USR scripts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset
from dnadesign.usr.scripts.refresh_overlay_registry_metadata import refresh_overlay_registry_metadata
from dnadesign.usr.src.contracts import compute_id
from dnadesign.usr.src.overlays import overlay_metadata
from dnadesign.usr.src.registry import registry_hash


def _usr_root(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")
    return usr_root


def test_refresh_overlay_registry_metadata_restamps_existing_compact_overlay(tmp_path: Path) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "demo")
    record_id = compute_id("dna", "ACGTACGT")
    with dataset.write_session() as session:
        session.init(source="fixture", notes="fixture")
        session.import_rows(
            [
                {
                    "id": record_id,
                    "bio_type": "dna",
                    "sequence": "ACGTACGT",
                    "alphabet": "dna_4",
                    "length": 8,
                    "source": "fixture",
                }
            ],
            source="fixture",
        )
        session.write_overlay(
            "usr_label",
            pd.DataFrame([{"id": record_id, "usr_label__primary": "demo", "usr_label__aliases": ["alias"]}]),
        )

    overlay_path = dataset.dir / "_derived" / "usr_label.parquet"
    table = pq.read_table(overlay_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"usr:registry_hash"] = b"stale"
    stale_table = table.replace_schema_metadata(metadata)
    pq.write_table(stale_table, overlay_path)
    assert overlay_metadata(overlay_path)["registry_hash"] == "stale"

    result = refresh_overlay_registry_metadata(usr_root=usr_root, dataset_name="demo", namespace="usr_label")

    assert result.rows_refreshed == 1
    assert overlay_metadata(overlay_path)["registry_hash"] == registry_hash(usr_root, required=True)
    dataset.validate(strict=True)


def test_refresh_overlay_registry_metadata_fails_for_missing_overlay(tmp_path: Path) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "demo")
    with dataset.write_session() as session:
        session.init(source="fixture", notes="fixture")
        session.import_rows(
            [
                {
                    "id": compute_id("dna", "ACGT"),
                    "bio_type": "dna",
                    "sequence": "ACGT",
                    "alphabet": "dna_4",
                    "length": 4,
                    "source": "fixture",
                }
            ],
            source="fixture",
        )

    with pytest.raises(FileNotFoundError, match="Compact overlay not found"):
        refresh_overlay_registry_metadata(usr_root=usr_root, dataset_name="demo", namespace="usr_label")
