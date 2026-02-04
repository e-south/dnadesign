"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_legend.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.baserender.src.legend import legend_entries_for_record
from dnadesign.baserender.src.model import Annotation, SeqRecord


def _record(seq: str, annotations):
    return SeqRecord(
        id="rec",
        alphabet="DNA",
        sequence=seq,
        annotations=annotations,
    ).validate()


def test_legend_entries_for_tfs():
    seq = "ACGTACGT"
    anns = [
        Annotation(start=0, length=3, strand="fwd", label="ACG", tag="tf:cpxr"),
        Annotation(start=4, length=3, strand="fwd", label="ACG", tag="tf:lexa"),
    ]
    rec = _record(seq, anns)
    assert legend_entries_for_record(rec) == [("tf:cpxr", "cpxr"), ("tf:lexa", "lexa")]


def test_legend_sigma_from_dataset_tag():
    seq = "TTGACA"
    anns = [
        Annotation(
            start=0,
            length=6,
            strand="fwd",
            label="TTGACA",
            tag="tf:sigma70_high",
        )
    ]
    rec = _record(seq, anns)
    assert legend_entries_for_record(rec) == [("sigma", "σ70 high")]


def test_legend_sigma_from_plugin_payload():
    seq = "AAAAAA"
    anns = [
        Annotation(
            start=0,
            length=3,
            strand="fwd",
            label="AAA",
            tag="sigma",
            payload={"strength": "mid"},
        )
    ]
    rec = _record(seq, anns)
    assert legend_entries_for_record(rec) == [("sigma", "σ70 mid")]
