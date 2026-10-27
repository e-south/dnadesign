"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/views/test_sequence_view_alias_repair.py

Tests for explicit sequence-view alias repair.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.registry import ensure_sequence_contract_namespaces
from dnadesign.usr.src.sequence_views import (
    SequenceViewRecord,
    load_sequence_views,
    repair_sequence_view_alias_conflicts,
)
from dnadesign.usr.src.sequence_views import maintenance as sequence_view_maintenance
from dnadesign.usr.src.sequence_views.store import _rows_to_table, _write_sequence_views_atomic, sequence_views_path


def _make_dataset(root: Path, name: str, rows: list[dict[str, object]]) -> Dataset:
    ensure_sequence_contract_namespaces(root)
    dataset = Dataset(root, name)
    dataset.init(source="unit-test")
    dataset.import_rows(rows, source="unit-test")
    return dataset


def _record(sequence_id: str, view_name: str, aliases: list[str] | None) -> SequenceViewRecord:
    return SequenceViewRecord(
        sequence_id=sequence_id,
        view_name=view_name,
        aliases=aliases,
        product_kind="source_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id="legacy_alias_views",
        created_at="2026-04-25T00:00:00.000000Z",
    )


def test_sequence_view_alias_repair_drops_all_non_unique_aliases(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "legacy_alias_views",
        [
            {"sequence": "ACGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
            {"sequence": "TGCA" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
            {"sequence": "GATTACA" * 12, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
        ],
    )
    ids = [str(row_id) for row_id in dataset.head(3)["id"].tolist()]
    rows = [
        _record(ids[0], "legacy_a", ["shared_alias", "a_only"]),
        _record(ids[1], "legacy_b", ["Shared_Alias", "b_only"]),
        _record(ids[2], "legacy_c", ["c_only"]),
    ]
    _write_sequence_views_atomic(sequence_views_path(dataset), _rows_to_table(rows))

    dry_run = repair_sequence_view_alias_conflicts(dataset, write=False)
    assert dry_run.duplicate_alias_keys == 1
    assert dry_run.conflicting_view_rows == 2
    assert dry_run.aliases_removed == 2
    assert dry_run.rows_touched == 2
    assert dry_run.written is False
    assert [row.aliases for row in load_sequence_views(dataset)] == [
        ["shared_alias", "a_only"],
        ["Shared_Alias", "b_only"],
        ["c_only"],
    ]

    written = repair_sequence_view_alias_conflicts(dataset, write=True)

    assert written.written is True
    assert written.aliases_removed == 2
    assert [row.aliases for row in load_sequence_views(dataset)] == [["a_only"], ["b_only"], ["c_only"]]


def test_sequence_view_alias_repair_noops_when_aliases_are_unique(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "unique_alias_views",
        [{"sequence": "ACGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = str(dataset.head(1)["id"].tolist()[0])
    rows = [_record(sequence_id, "native", ["unique_alias"])]
    _write_sequence_views_atomic(sequence_views_path(dataset), _rows_to_table(rows))

    result = repair_sequence_view_alias_conflicts(dataset, write=True)

    assert result.duplicate_alias_keys == 0
    assert result.aliases_removed == 0
    assert result.written is False
    assert load_sequence_views(dataset)[0].aliases == ["unique_alias"]


def test_sequence_view_alias_repair_write_reloads_rows_after_lock(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "late_sequence_view_writer",
        [
            {"sequence": "ACGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
            {"sequence": "TGCA" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
            {"sequence": "GATTACA" * 12, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
            {"sequence": "CAGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
        ],
    )
    ids = [str(row_id) for row_id in dataset.head(4)["id"].tolist()]
    initial_rows = [
        _record(ids[0], "legacy_a", ["shared_alias", "a_only"]),
        _record(ids[1], "legacy_b", ["Shared_Alias", "b_only"]),
        _record(ids[2], "legacy_c", ["c_only"]),
    ]
    late_row = _record(ids[3], "late_view", ["late_only"])
    _write_sequence_views_atomic(sequence_views_path(dataset), _rows_to_table(initial_rows))
    injected = {"done": False}

    @contextmanager
    def _injecting_lock(_dataset_dir: Path):
        if not injected["done"]:
            current_rows = load_sequence_views(dataset)
            _write_sequence_views_atomic(sequence_views_path(dataset), _rows_to_table([*current_rows, late_row]))
            injected["done"] = True
        yield

    monkeypatch.setattr(sequence_view_maintenance, "dataset_write_lock", _injecting_lock)

    result = repair_sequence_view_alias_conflicts(dataset, write=True)

    assert result.written is True
    stored_aliases = {row.view_name: row.aliases for row in load_sequence_views(dataset)}
    assert stored_aliases == {
        "legacy_a": ["a_only"],
        "legacy_b": ["b_only"],
        "legacy_c": ["c_only"],
        "late_view": ["late_only"],
    }
