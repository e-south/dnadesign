"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_figures.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from dnadesign.baserender.src.api import render_images
from dnadesign.baserender.src.model import SeqRecord


def test_render_images_closes_figures(tmp_path):
    plt.close("all")
    rec = SeqRecord(id="r1", alphabet="DNA", sequence="ACGTAC").validate()
    out_dir = tmp_path / "imgs"
    render_images([rec], out_dir=out_dir, fmt="png")
    assert plt.get_fignums() == []
