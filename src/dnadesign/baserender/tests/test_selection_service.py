"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_selection_service.py

Selection service tests for strict filtering and keep_order behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.baserender.src.config import SelectionCfg
from dnadesign.baserender.src.core import Record, SchemaError
from dnadesign.baserender.src.core.record import Display
from dnadesign.baserender.src.pipeline import apply_selection


def _record(record_id: str, sequence: str, row_index: int) -> Record:
    return Record(
        id=record_id,
        alphabet="DNA",
        sequence=sequence,
        features=(),
        effects=(),
        display=Display(),
        meta={"row_index": row_index},
    ).validate()


def _write_selection(path: Path, *, header: str, values: list[str]) -> Path:
    lines = [header] + values
    path.write_text("\n".join(lines) + "\n")
    return path


def test_apply_selection_id_keep_order_false_filters_to_csv_keys(tmp_path: Path) -> None:
    records = [
        _record("a", "AAAA", 10),
        _record("b", "CCCC", 5),
        _record("c", "GGGG", 7),
    ]
    csv_path = _write_selection(tmp_path / "sel_id.csv", header="id", values=["b", "a"])
    cfg = SelectionCfg(
        path=csv_path,
        match_on="id",
        column="id",
        overlay_column=None,
        keep_order=False,
        on_missing="error",
    )

    selected, missing = apply_selection(records, cfg)
    assert missing == []
    assert [r.id for r in selected] == ["a", "b"]


def test_apply_selection_sequence_keep_order_false_filters_to_csv_keys(tmp_path: Path) -> None:
    records = [
        _record("a", "AAAA", 10),
        _record("b", "CCCC", 5),
        _record("c", "GGGG", 7),
    ]
    csv_path = _write_selection(tmp_path / "sel_seq.csv", header="sequence", values=["CCCC", "AAAA"])
    cfg = SelectionCfg(
        path=csv_path,
        match_on="sequence",
        column="sequence",
        overlay_column=None,
        keep_order=False,
        on_missing="error",
    )

    selected, missing = apply_selection(records, cfg)
    assert missing == []
    assert [r.id for r in selected] == ["a", "b"]


def test_apply_selection_row_keep_order_false_filters_to_csv_keys(tmp_path: Path) -> None:
    records = [
        _record("a", "AAAA", 10),
        _record("b", "CCCC", 5),
        _record("c", "GGGG", 7),
    ]
    csv_path = _write_selection(tmp_path / "sel_row.csv", header="row", values=["7", "10"])
    cfg = SelectionCfg(
        path=csv_path,
        match_on="row",
        column="row",
        overlay_column=None,
        keep_order=False,
        on_missing="error",
    )

    selected, missing = apply_selection(records, cfg)
    assert missing == []
    assert [r.id for r in selected] == ["c", "a"]


def test_apply_selection_row_requires_integer_row_index_metadata(tmp_path: Path) -> None:
    record = Record(
        id="bad",
        alphabet="DNA",
        sequence="AAAA",
        features=(),
        effects=(),
        display=Display(),
        meta={"row_index": "not_int"},
    ).validate()
    csv_path = _write_selection(tmp_path / "sel_row.csv", header="row", values=["1"])
    cfg = SelectionCfg(
        path=csv_path,
        match_on="row",
        column="row",
        overlay_column=None,
        keep_order=False,
        on_missing="error",
    )

    with pytest.raises(SchemaError, match="record.meta.row_index must be int"):
        apply_selection([record], cfg)


def test_apply_selection_id_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    records = [
        _record("dup", "AAAA", 0),
        _record("dup", "CCCC", 1),
    ]
    csv_path = _write_selection(tmp_path / "sel_id_dup.csv", header="id", values=["dup"])
    cfg = SelectionCfg(
        path=csv_path,
        match_on="id",
        column="id",
        overlay_column=None,
        keep_order=False,
        on_missing="error",
    )

    with pytest.raises(SchemaError, match="duplicate record keys"):
        apply_selection(records, cfg)


def test_apply_selection_sequence_rejects_duplicate_record_sequences(tmp_path: Path) -> None:
    records = [
        _record("a", "AAAA", 0),
        _record("b", "AAAA", 1),
    ]
    csv_path = _write_selection(tmp_path / "sel_seq_dup.csv", header="sequence", values=["AAAA"])
    cfg = SelectionCfg(
        path=csv_path,
        match_on="sequence",
        column="sequence",
        overlay_column=None,
        keep_order=False,
        on_missing="error",
    )

    with pytest.raises(SchemaError, match="duplicate record keys"):
        apply_selection(records, cfg)
