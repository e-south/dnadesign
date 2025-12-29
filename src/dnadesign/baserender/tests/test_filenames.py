"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_filenames.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.baserender.src.api import render_images
from dnadesign.baserender.src.model import SeqRecord


def test_duplicate_ids_do_not_overwrite(tmp_path):
    recs = [
        SeqRecord(id="dup", alphabet="DNA", sequence="ACGTAC").validate(),
        SeqRecord(id="dup", alphabet="DNA", sequence="TTTTAA").validate(),
    ]
    out_dir = tmp_path / "imgs"
    render_images(recs, out_dir=out_dir, fmt="png")
    names = sorted(p.name for p in out_dir.iterdir())
    assert names == ["dup.png", "dup_1.png"]


def test_id_with_slash_sanitized(tmp_path):
    recs = [SeqRecord(id="a/b", alphabet="DNA", sequence="ACGTAC").validate()]
    out_dir = tmp_path / "imgs"
    render_images(recs, out_dir=out_dir, fmt="png")
    names = [p.name for p in out_dir.iterdir()]
    assert names == ["a_b.png"]
