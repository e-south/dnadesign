"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/motif/parsers/meme.py

MEME-suite motif file → PWM.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from Bio import motifs
from xml.etree import ElementTree
from dnadesign.cruncher.motif.backend import register
from dnadesign.cruncher.motif.model import PWM
from typing import Optional

@register("MEME")
def parse_meme(path: Path) -> PWM:
    lines = path.read_text().splitlines()
    logodds_rows: list[list[float]] = []
    prob_rows:    list[list[float]] = []
    nsites: Optional[int] = None
    evalue: Optional[float] = None

    # First, try Biopython XML parser for probability matrix only
    try:
        with path.open() as handle:
            mlist = motifs.parse(handle, "MEME")
        if mlist:
            m = mlist[0]
            prob_mat = np.array(m.pwm).T.astype("float32")
            return PWM(
                name=m.name or path.stem,
                matrix=prob_mat
            )
    except (ValueError, ElementTree.ParseError):
        pass

    # Fallback: plain‐text parsing, collect both log-odds and letter-probability blocks
    i = 0
    while i < len(lines):
        header = lines[i].lower()
        if header.startswith("log-odds matrix"):
            # skip the header line, then read numeric rows
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("---"):
                parts = lines[i].strip().split()
                logodds_rows.append([float(x) for x in parts])
                i += 1
            continue

        if header.startswith("letter-probability matrix"):
            # extract nsites & E-value
            m_meta = re.search(r"nsites=\s*(\d+).*E=\s*([0-9.eE+-]+)", header)
            if m_meta:
                nsites = int(m_meta.group(1))
                evalue = float(m_meta.group(2))
            # read the probability rows
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("---"):
                parts = lines[i].strip().split()
                # last four columns are the probabilities for A,C,G,T
                prob_rows.append([float(x) for x in parts[-4:]])
                i += 1
            continue

        i += 1

    if not prob_rows:
        raise ValueError(f"Could not parse PWM from MEME file {path}")

    prob_mat = np.array(prob_rows, dtype="float32")
    logodds_mat = np.array(logodds_rows, dtype="float32") if logodds_rows else None

    return PWM(
        name=path.stem,
        matrix=prob_mat,
        nsites=nsites,
        evalue=evalue,
        log_odds_matrix=logodds_mat
    )