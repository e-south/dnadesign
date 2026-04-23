"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_merge_overlay_carry.py

Contracts for explicit overlay carry during USR maintenance merges.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset
from dnadesign.usr.src.datasets.merge import MergeColumnsMode, MergePolicy, merge_usr_to_usr
from dnadesign.usr.src.errors import SchemaError
from dnadesign.usr.src.overlays import with_overlay_metadata
from dnadesign.usr.src.storage.parquet import now_utc
from dnadesign.usr.tests.registry_helpers import ensure_registry, register_test_namespace


def _row(sequence: str, *, record_id: str | None = None, source: str = "test") -> dict[str, object]:
    row = {
        "sequence": sequence,
        "bio_type": "dna",
        "alphabet": "dna_4",
        "source": source,
    }
    if record_id is not None:
        row["id"] = record_id
    return row


def _make_merge_datasets(tmp_path: Path) -> tuple[Path, Dataset, Dataset]:
    root = tmp_path / "datasets"
    ensure_registry(root)
    dest = Dataset(root, "dest")
    src = Dataset(root, "src")
    dest.init(source="unit-test")
    src.init(source="unit-test")
    return root, dest, src


def _register_usr_label_namespace(root: Path) -> None:
    register_test_namespace(
        root,
        namespace="usr_label",
        columns_spec="usr_label__primary:string,usr_label__aliases:list<string>",
    )


def _write_usr_labels(dataset: Dataset, rows: dict[str, tuple[str, list[str]]]) -> None:
    with dataset.write_session() as session:
        session.write_overlay(
            "usr_label",
            pa.table(
                {
                    "id": list(rows.keys()),
                    "usr_label__primary": [value[0] for value in rows.values()],
                    "usr_label__aliases": [value[1] for value in rows.values()],
                }
            ),
            key="id",
            note="merge overlay carry test",
        )


def test_merge_carry_namespace_preserves_labels_for_surviving_source_rows(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    _register_usr_label_namespace(root)
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")

    dest_id = dest.head(1)["id"].iloc[0]
    src_id = src.head(1)["id"].iloc[0]
    _write_usr_labels(dest, {dest_id: ("dest-label", ["dest-alias"])})
    _write_usr_labels(src, {src_id: ("src-label", ["src-alias"])})

    with dest.maintenance(reason="merge"):
        preview = merge_usr_to_usr(
            root=root,
            dest="dest",
            src="src",
            columns_mode=MergeColumnsMode.UNION,
            duplicate_policy=MergePolicy.SKIP,
            carry_namespaces=["usr_label"],
        )

    rows = dest.head(10, columns=["sequence", "usr_label__primary", "usr_label__aliases"]).to_dict(orient="records")
    by_sequence = {str(row["sequence"]): row for row in rows}
    assert by_sequence["ACGT"]["usr_label__primary"] == "dest-label"
    assert by_sequence["TGCA"]["usr_label__primary"] == "src-label"
    assert by_sequence["TGCA"]["usr_label__aliases"] == ["src-alias"]
    assert preview.carried_namespace_counts == (("usr_label", 1),)
    events = [json.loads(line) for line in dest.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    merge_event = [row for row in events if row.get("action") == "merge_datasets"][-1]
    assert merge_event["args"]["carried_namespaces"] == ["usr_label"]
    assert merge_event["args"]["carried_namespace_rows"] == {"usr_label": 1}
    assert merge_event["args"]["carried_overlay_rows"] == 1


def test_merge_carry_namespace_overwrites_duplicate_ids_when_prefer_src(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    _register_usr_label_namespace(root)
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("ACGT")], source="unit-test")
    shared_id = dest.head(1)["id"].iloc[0]
    _write_usr_labels(dest, {shared_id: ("old-label", ["old-alias"])})
    _write_usr_labels(src, {shared_id: ("new-label", ["new-alias"])})

    with dest.maintenance(reason="merge"):
        preview = merge_usr_to_usr(
            root=root,
            dest="dest",
            src="src",
            columns_mode=MergeColumnsMode.UNION,
            duplicate_policy=MergePolicy.PREFER_SRC,
            avoid_casefold_dups=False,
            carry_namespaces=["usr_label"],
        )

    row = dest.head(5, columns=["id", "sequence", "usr_label__primary", "usr_label__aliases"]).iloc[0]
    assert row["id"] == shared_id
    assert row["sequence"] == "ACGT"
    assert row["usr_label__primary"] == "new-label"
    assert row["usr_label__aliases"] == ["new-alias"]
    assert preview.carried_namespace_counts == (("usr_label", 1),)


def test_merge_carry_namespace_dry_run_reports_counts_without_mutating_dest(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    _register_usr_label_namespace(root)
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")
    src_id = src.head(1)["id"].iloc[0]
    _write_usr_labels(src, {src_id: ("src-label", ["src-alias"])})

    with dest.maintenance(reason="merge"):
        preview = merge_usr_to_usr(
            root=root,
            dest="dest",
            src="src",
            columns_mode=MergeColumnsMode.UNION,
            duplicate_policy=MergePolicy.SKIP,
            carry_namespaces=["usr_label"],
            dry_run=True,
        )

    assert preview.carried_namespace_counts == (("usr_label", 1),)
    assert len(dest.head(10)) == 1
    assert not (dest.dir / "_derived" / "usr_label.parquet").exists()


def test_merge_carry_namespace_requires_namespace_in_source_dataset(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")

    with dest.maintenance(reason="merge"):
        with pytest.raises(SchemaError, match="Requested carry namespace 'usr_label' was not found"):
            merge_usr_to_usr(
                root=root,
                dest="dest",
                src="src",
                columns_mode=MergeColumnsMode.UNION,
                duplicate_policy=MergePolicy.SKIP,
                carry_namespaces=["usr_label"],
                dry_run=True,
            )


def test_merge_carry_namespace_rejects_non_id_keyed_source_overlay(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    register_test_namespace(root, namespace="infer", columns_spec="infer__score:float64")
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")
    attach_path = tmp_path / "infer.csv"
    pd.DataFrame({"sequence": ["TGCA"], "infer__score": [0.5]}).to_csv(attach_path, index=False)
    src.attach(
        attach_path,
        namespace="infer",
        key="sequence",
        key_col="sequence",
        columns=["infer__score"],
        parse_json=False,
    )

    with dest.maintenance(reason="merge"):
        with pytest.raises(SchemaError, match="only supports id-keyed overlays"):
            merge_usr_to_usr(
                root=root,
                dest="dest",
                src="src",
                columns_mode=MergeColumnsMode.UNION,
                duplicate_policy=MergePolicy.SKIP,
                carry_namespaces=["infer"],
                dry_run=True,
            )


def test_merge_carry_namespace_requires_compact_source_overlay(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")
    src_id = src.head(1)["id"].iloc[0]
    src.write_overlay_part("mock", pa.table({"id": [src_id], "mock__score": [1.0]}), key="id")

    with dest.maintenance(reason="merge"):
        with pytest.raises(SchemaError, match="requires a compact source overlay file"):
            merge_usr_to_usr(
                root=root,
                dest="dest",
                src="src",
                columns_mode=MergeColumnsMode.UNION,
                duplicate_policy=MergePolicy.SKIP,
                carry_namespaces=["mock"],
                dry_run=True,
            )


def test_merge_carry_namespace_rejects_duplicate_source_overlay_keys_before_base_rewrite(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    _register_usr_label_namespace(root)
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")
    src_id = src.head(1)["id"].iloc[0]

    overlay_path = src.dir / "_derived" / "usr_label.parquet"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    table = with_overlay_metadata(
        pa.table(
            {
                "id": [src_id, src_id],
                "usr_label__primary": ["dup-a", "dup-b"],
                "usr_label__aliases": [["dup-a"], ["dup-b"]],
            }
        ),
        namespace="usr_label",
        key="id",
        created_at=now_utc(),
        registry_hash=src._registry_hash(required=True),  # noqa: SLF001
    )
    pq.write_table(table, overlay_path)

    before_rows = len(dest.head(10))
    with dest.maintenance(reason="merge"):
        with pytest.raises(SchemaError, match="duplicate keys in source overlay"):
            merge_usr_to_usr(
                root=root,
                dest="dest",
                src="src",
                columns_mode=MergeColumnsMode.UNION,
                duplicate_policy=MergePolicy.SKIP,
                carry_namespaces=["usr_label"],
            )

    assert len(dest.head(10)) == before_rows


def test_merge_carry_namespace_rejects_reserved_namespace(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")

    with dest.maintenance(reason="merge"):
        with pytest.raises(SchemaError, match="reserved and cannot be transferred"):
            merge_usr_to_usr(
                root=root,
                dest="dest",
                src="src",
                columns_mode=MergeColumnsMode.UNION,
                duplicate_policy=MergePolicy.SKIP,
                carry_namespaces=["usr_state"],
                dry_run=True,
            )


def test_merge_carry_namespace_ignores_reserved_state_overlays(tmp_path: Path) -> None:
    root, dest, src = _make_merge_datasets(tmp_path)
    _register_usr_label_namespace(root)
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")

    dest_id = dest.head(1)["id"].iloc[0]
    src_id = src.head(1)["id"].iloc[0]
    dest.set_state([dest_id], qc_status="pass")
    src.set_state([src_id], qc_status="fail")
    _write_usr_labels(src, {src_id: ("src-label", ["src-alias"])})

    with dest.maintenance(reason="merge"):
        preview = merge_usr_to_usr(
            root=root,
            dest="dest",
            src="src",
            columns_mode=MergeColumnsMode.UNION,
            duplicate_policy=MergePolicy.SKIP,
            carry_namespaces=["usr_label"],
        )

    rows = dest.head(10, columns=["sequence", "usr_label__primary"]).to_dict(orient="records")
    by_sequence = {str(row["sequence"]): row for row in rows}
    assert by_sequence["TGCA"]["usr_label__primary"] == "src-label"
    assert preview.carried_namespace_counts == (("usr_label", 1),)
