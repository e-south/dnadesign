"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/parse/parsers/meme.py

MEME-suite motif file → PWM.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import numpy as np
from Bio import motifs

from dnadesign.cruncher.parse.backend import register
from dnadesign.cruncher.parse.model import PWM


@register("MEME")
def parse_meme(path: Path) -> PWM:
    """
    Parse a MEME-format PWM (XML or text) and always
    use the file's own stem (preserving case) as PWM.name.
    """
    lines = path.read_text().splitlines()
    logodds_rows: list[list[float]] = []
    prob_rows: list[list[float]] = []
    nsites: Optional[int] = None
    evalue: Optional[float] = None

    # Try the Biopython XML path (but ignore m.name entirely)
    try:
        with path.open() as handle:
            mlist = motifs.parse(handle, "MEME")
        if mlist:
            m = mlist[0]
            # Biopython's m.pwm is a dict of columns, so transpose to get L×4
            prob_mat = np.array(m.pwm).T.astype("float32")
            # Use the file stem for name:
            return PWM(name=path.stem, matrix=prob_mat)
    except (ValueError, ElementTree.ParseError):
        pass

    # Fallback to manual text parsing
    i = 0
    while i < len(lines):
        header = lines[i].lower()
        if header.startswith("log-odds matrix"):
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("---"):
                parts = lines[i].strip().split()
                logodds_rows.append([float(x) for x in parts])
                i += 1
            continue

        if header.startswith("letter-probability matrix"):
            # pull out nsites & E
            m_meta = re.search(r"nsites=\s*(\d+).*E=\s*([0-9.eE+-]+)", header)
            if m_meta:
                nsites = int(m_meta.group(1))
                evalue = float(m_meta.group(2))
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("---"):
                parts = lines[i].strip().split()
                prob_rows.append([float(x) for x in parts[-4:]])
                i += 1
            continue

        i += 1

    if not prob_rows:
        raise ValueError(f"Could not parse PWM from MEME file {path}")

    prob_mat = np.array(prob_rows, dtype="float32")
    logodds_mat = np.array(logodds_rows, dtype="float32") if logodds_rows else None

    return PWM(name=path.stem, matrix=prob_mat, nsites=nsites, evalue=evalue, log_odds_matrix=logodds_mat)
