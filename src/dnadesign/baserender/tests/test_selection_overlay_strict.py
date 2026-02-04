"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_selection_overlay_strict.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.baserender.src.contracts import SchemaError
from dnadesign.baserender.src.model import Guide, SeqRecord
from dnadesign.baserender.src.selection import apply_overlay_label, read_selection_csv


def test_missing_overlay_column_raises(tmp_path):
    path = tmp_path / "sel.csv"
    path.write_text("id,foo\nrec_1,x\n")
    with pytest.raises(SchemaError):
        read_selection_csv(path, key_col="id", overlay_col="details")


def test_explicit_overlay_column_null_no_read(tmp_path):
    path = tmp_path / "sel.csv"
    path.write_text("id,details\nrec_1,hello\n")
    spec = read_selection_csv(path, key_col="id", overlay_col=None)
    assert spec.overlays == [None]


def test_csv_overlay_overwrites_dataset_overlay():
    rec = SeqRecord(
        id="rec",
        alphabet="DNA",
        sequence="ACGT",
        annotations=(),
        guides=(Guide(kind="overlay_label", start=0, end=0, label="old"),),
    ).validate()
    out = apply_overlay_label(rec, "new", source="csv")
    labels = [g.label for g in out.guides if g.kind == "overlay_label"]
    assert labels == ["new"]
