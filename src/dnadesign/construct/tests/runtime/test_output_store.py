"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_output_store.py

Direct tests for construct output-store helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.construct.src.persistence.usr_registry import (
    _CONSTRUCT_COLUMNS,
    _CONSTRUCT_SEED_COLUMNS,
    _USR_LABEL_COLUMNS,
    _USR_STATE_COLUMNS,
    _construct_metadata_table,
    _ensure_construct_registry,
    _existing_output_ids,
    _usr_label_table,
)
from dnadesign.usr import Dataset
from dnadesign.usr.src.registry.models import DERIVED_COLUMNS, SEQ_ANNOT_COLUMNS


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _column_pairs(columns: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(str(column["name"]), str(column["type"])) for column in columns]


def _registry_column_dicts(columns: list[object]) -> list[dict[str, str]]:
    return [{"name": str(column.name), "type": str(column.type)} for column in columns]


def test_ensure_construct_registry_writes_required_namespaces(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"

    _ensure_construct_registry(root)

    payload = yaml.safe_load((root / "registry.yaml").read_text(encoding="utf-8"))
    namespaces = payload["namespaces"]

    assert set(namespaces) >= {"construct", "construct_seed", "usr_label", "usr_state", "seq_annot", "derived"}
    assert _column_pairs(namespaces["construct"]["columns"]) == _column_pairs(_CONSTRUCT_COLUMNS)
    assert _column_pairs(namespaces["construct_seed"]["columns"]) == _column_pairs(_CONSTRUCT_SEED_COLUMNS)
    assert _column_pairs(namespaces["usr_label"]["columns"]) == _column_pairs(_USR_LABEL_COLUMNS)


def test_checked_in_shared_usr_registry_matches_construct_contract() -> None:
    payload = yaml.safe_load((_repo_root() / "src/dnadesign/usr/datasets/registry.yaml").read_text(encoding="utf-8"))
    namespaces = payload["namespaces"]

    assert _column_pairs(namespaces["construct"]["columns"]) == _column_pairs(_CONSTRUCT_COLUMNS)
    assert _column_pairs(namespaces["construct_seed"]["columns"]) == _column_pairs(_CONSTRUCT_SEED_COLUMNS)
    assert _column_pairs(namespaces["usr_label"]["columns"]) == _column_pairs(_USR_LABEL_COLUMNS)


def test_ensure_construct_registry_is_idempotent_for_checked_in_shared_registry(tmp_path: Path) -> None:
    source = _repo_root() / "src/dnadesign/usr/datasets/registry.yaml"
    root = tmp_path / "usr_root"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "registry.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    before = target.read_text(encoding="utf-8")

    _ensure_construct_registry(root)

    assert target.read_text(encoding="utf-8") == before


def test_ensure_construct_registry_preserves_existing_valid_construct_column_order(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    stale_names = {
        "construct__context_id",
        "construct__context_kind",
        "construct__anchor_id",
        "construct__anchor_orientation",
        "construct__anchor_start",
        "construct__anchor_end",
        "construct__resolved_length",
    }
    base_columns = [dict(column) for column in _CONSTRUCT_COLUMNS if column["name"] not in stale_names]
    late_columns = [dict(column) for column in _CONSTRUCT_COLUMNS if column["name"] in stale_names]
    parts_index = next(index for index, column in enumerate(base_columns) if column["name"] == "construct__parts")
    stale_columns = base_columns[: parts_index + 1] + late_columns + base_columns[parts_index + 1 :]
    payload = {
        "namespaces": {
            "audit": {
                "owner": "usr",
                "description": "audit namespace preserved during construct repair",
                "columns": [{"name": "audit__score", "type": "float64"}],
            },
            "construct": {
                "owner": "construct",
                "description": "Construct lineage overlays for realized DNA sequences.",
                "columns": stale_columns,
            },
            "construct_seed": {
                "owner": "construct",
                "description": "Construct bootstrap/import metadata for seeded input datasets.",
                "columns": [dict(column) for column in _CONSTRUCT_SEED_COLUMNS],
            },
            "derived": {
                "owner": "usr",
                "description": "Derived-product lineage, focal selection, and feature-retention overlays.",
                "columns": _registry_column_dicts(DERIVED_COLUMNS),
            },
            "seq_annot": {
                "owner": "usr",
                "description": "Imported source annotation overlays with preserved GenBank location fidelity.",
                "columns": _registry_column_dicts(SEQ_ANNOT_COLUMNS),
            },
            "usr_label": {
                "owner": "usr",
                "description": "Human-readable labels and aliases for canonical sequence records.",
                "columns": [dict(column) for column in _USR_LABEL_COLUMNS],
            },
            "usr_state": {
                "owner": "usr",
                "description": "Reserved record-state overlay (masked/qc/split/lineage).",
                "columns": [dict(column) for column in _USR_STATE_COLUMNS],
            },
        }
    }
    (root / "registry.yaml").parent.mkdir(parents=True, exist_ok=True)
    registry_path = root / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    _ensure_construct_registry(root)

    repaired = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    namespaces = repaired["namespaces"]
    assert _column_pairs(namespaces["construct"]["columns"]) == _column_pairs(stale_columns)
    assert namespaces["audit"]["columns"] == [{"name": "audit__score", "type": "float64"}]
    assert namespaces["seq_annot"]["owner"] == "usr"
    assert namespaces["derived"]["owner"] == "usr"


def test_ensure_construct_registry_appends_missing_construct_columns_without_reordering(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    missing_names = {
        "construct__context_id",
        "construct__context_kind",
        "construct__anchor_id",
        "construct__anchor_orientation",
        "construct__anchor_start",
        "construct__anchor_end",
        "construct__resolved_length",
    }
    existing_columns = [dict(column) for column in _CONSTRUCT_COLUMNS if column["name"] not in missing_names]
    payload = {
        "namespaces": {
            "construct": {
                "owner": "construct",
                "description": "Construct lineage overlays for realized DNA sequences.",
                "columns": existing_columns,
            },
        }
    }
    (root / "registry.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "registry.yaml").write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    _ensure_construct_registry(root)

    repaired = yaml.safe_load((root / "registry.yaml").read_text(encoding="utf-8"))
    appended = repaired["namespaces"]["construct"]["columns"]
    assert _column_pairs(appended[: len(existing_columns)]) == _column_pairs(existing_columns)
    assert _column_pairs(appended[len(existing_columns) :]) == _column_pairs(
        [column for column in _CONSTRUCT_COLUMNS if column["name"] in missing_names]
    )


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
