"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_output_store.py

Direct tests for construct output-store helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.construct.src.output_store import (
    _construct_metadata_table,
    _ensure_construct_registry,
    _existing_output_ids,
    _usr_label_table,
)
from dnadesign.usr import Dataset


def test_ensure_construct_registry_writes_required_namespaces(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"

    _ensure_construct_registry(root)

    payload = yaml.safe_load((root / "registry.yaml").read_text(encoding="utf-8"))
    namespaces = payload["namespaces"]

    assert set(namespaces) >= {"construct", "construct_seed", "usr_label", "usr_state"}
    construct_columns = {column["name"]: column["type"] for column in namespaces["construct"]["columns"]}
    assert construct_columns["construct__parts"].startswith("list<struct<")
    assert construct_columns["construct__context_id"] == "string"
    assert construct_columns["construct__context_kind"] == "string"
    assert construct_columns["construct__anchor_id"] == "string"
    assert construct_columns["construct__anchor_start"] == "int64"
    assert construct_columns["construct__anchor_end"] == "int64"
    assert construct_columns["construct__resolved_length"] == "int64"


def test_existing_output_ids_returns_ids_for_initialized_dataset(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    _ensure_construct_registry(root)

    assert _existing_output_ids(root, "anchors_demo") == set()

    dataset = Dataset(root, "anchors_demo")
    dataset.init(source="test", notes="output-store test")
    result = dataset.add_sequences(["ACGT", "GGGG"], bio_type="dna", alphabet="dna_4", source="test")

    assert _existing_output_ids(root, "anchors_demo") == set(result.ids)


def test_construct_metadata_table_uses_construct_registry_schema() -> None:
    table = _construct_metadata_table(
        [
            {
                "id": "out-1",
                "construct__job": "job-1",
                "construct__parts": [
                    {
                        "name": "core",
                        "role": "payload",
                        "sequence_source": "input_field",
                        "sequence_field": "sequence",
                        "placement_kind": "template_span",
                        "orientation": "forward",
                        "template_start": 0,
                        "template_end": 4,
                        "realized_start": 0,
                        "realized_end": 4,
                        "length": 4,
                    }
                ],
            }
        ]
    )

    assert table.num_rows == 1
    assert "construct__job" in table.schema.names
    assert str(table.schema.field("construct__parts").type).startswith("list<item: struct<")


def test_usr_label_table_uses_usr_label_schema() -> None:
    table = _usr_label_table(
        [
            {
                "id": "out-1",
                "usr_label__primary": "J23105",
                "usr_label__aliases": ["BBa_J23105"],
            }
        ]
    )

    assert table.num_rows == 1
    assert table.column("usr_label__primary").to_pylist() == ["J23105"]
    assert table.column("usr_label__aliases").to_pylist() == [["BBa_J23105"]]
