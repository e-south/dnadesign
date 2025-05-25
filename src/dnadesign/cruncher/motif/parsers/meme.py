"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/motif/parser.py

MEME-suite motif file → PWM.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from Bio import motifs

from ..backend import register
from ..model import PWM


@register("MEME")
def parse_meme(path: Path) -> PWM:
    with path.open() as handle:
        mlist = motifs.parse(handle, "MEME")
    if not mlist:
        raise ValueError(f"No motifs found in MEME file {path}")
    m = mlist[0]  # pick the first motif
    mat = np.array(m.pwm).T.astype("float32")  # BioPython gives list[dict]
    return PWM(name=m.name or path.stem, matrix=mat)
