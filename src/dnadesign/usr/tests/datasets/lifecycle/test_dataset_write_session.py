"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/lifecycle/test_dataset_write_session.py

Behavior tests for the explicit Dataset write-session contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pytest

from dnadesign.devtools.tests.support.usr import ensure_registry
from dnadesign.usr import Dataset, compute_id
from dnadesign.usr.src.datasets.lifecycle import write_session as write_session_module


def test_write_session_requires_with_block(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    ds = Dataset(root, "demo")

    with pytest.raises(RuntimeError, match="with"):
        ds.write_session().init_if_missing(source="test")


def test_write_session_holds_one_lock_and_reuses_lockless_helpers(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    ds = Dataset(root, "demo")
    observed: dict[str, object] = {}
    lock_calls: list[Path] = []

    @contextmanager
    def _counting_lock(path: Path):
        lock_calls.append(path)
        yield

    real_add_sequences = write_session_module.add_sequences_dataset
    real_import_rows = write_session_module.import_rows_dataset
    real_write_overlay = write_session_module.write_overlay_dataset

    def _spy_add_sequences(*args, **kwargs):
        observed["add_sequences_write_lock"] = kwargs.get("write_lock")
        return real_add_sequences(*args, **kwargs)

    def _spy_import_rows(*args, **kwargs):
        observed["import_write_lock"] = kwargs.get("write_lock")
        return real_import_rows(*args, **kwargs)

    def _spy_write_overlay(*args, **kwargs):
        observed["overlay_write_lock"] = kwargs.get("write_lock")
        return real_write_overlay(*args, **kwargs)

    monkeypatch.setattr(write_session_module, "dataset_write_lock", _counting_lock)
    monkeypatch.setattr(write_session_module, "add_sequences_dataset", _spy_add_sequences)
    monkeypatch.setattr(write_session_module, "import_rows_dataset", _spy_import_rows)
    monkeypatch.setattr(write_session_module, "write_overlay_dataset", _spy_write_overlay)

    with ds.write_session() as session:
        assert session.init_if_missing(source="test", notes="write-session") is True
        added = session.add_sequences(["GGGG"], bio_type="dna", alphabet="dna_4", source="test")
        assert added.added == 1
        assert session.import_rows([{"sequence": "ACGT"}], source="test") == 1
        assert (
            session.write_overlay(
                "mock",
                pa.table(
                    {
                        "id": [compute_id("dna", "GGGG"), compute_id("dna", "ACGT")],
                        "mock__score": [2.5, 1.5],
                    }
                ),
                note="test overlay write",
            )
            == 2
        )

    assert lock_calls == [ds.dir]
    assert observed["add_sequences_write_lock"] is write_session_module._held_write_lock
    assert observed["import_write_lock"] is write_session_module._held_write_lock
    assert observed["overlay_write_lock"] is write_session_module._held_write_lock

    head = ds.head(n=10, include_derived=True).sort_values("sequence").reset_index(drop=True)
    assert list(head["sequence"]) == ["ACGT", "GGGG"]
    assert list(head["mock__score"]) == pytest.approx([1.5, 2.5])
