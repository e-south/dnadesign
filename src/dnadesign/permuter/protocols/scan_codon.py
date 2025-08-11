"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/scan_codon.py

Codon-level saturation. For each codon in the selected region, replace with
the most frequent codon of every other amino acid from a usage table.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from . import register
from .base import Protocol

_TRIPLE = 3
_REQUIRED_COLUMNS = {"codon", "amino_acid", "frequency"}


def _load_codon_table(path: str | Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    aa2entries: Dict[str, List[Tuple[str, float]]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=",")
        if not _REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(
                f"scan_codon: codon_table must include columns {_REQUIRED_COLUMNS}, got {reader.fieldnames}"
            )
        for row in reader:
            aa = row["amino_acid"]
            codon = row["codon"].upper()
            freq = float(row["frequency"])
            aa2entries.setdefault(aa, []).append((codon, freq))
    aa2codons: Dict[str, List[str]] = {}
    codon2aa: Dict[str, str] = {}
    for aa, entries in aa2entries.items():
        sorted_codons = sorted(entries, key=lambda cf: cf[1], reverse=True)
        aa2codons[aa] = [codon for codon, _ in sorted_codons]
        for c, _ in sorted_codons:
            codon2aa[c] = aa
    return aa2codons, codon2aa


@register
class ScanCodon(Protocol):
    id = "scan_codon"
    version = "1.0"

    def validate_cfg(self, *, params: Dict) -> None:
        # Required: codon_table path
        table = params.get("codon_table")
        if not table or not Path(table).expanduser().exists():
            raise ValueError(
                "scan_codon: params.codon_table is required and must exist"
            )
        # Optional: region_codons must be [start,end]
        rc = params.get("region_codons")
        if rc is not None:
            if not (isinstance(rc, (list, tuple)) and len(rc) == 2):
                raise ValueError(
                    "scan_codon: params.region_codons must be [start,end] in codon units"
                )

    def generate(
        self, *, ref_entry: Dict, params: Dict, rng: np.random.Generator
    ) -> Iterable[Dict]:
        table = params.get("codon_table")
        aa2codons, codon2aa = _load_codon_table(Path(table).expanduser())

        seq = str(ref_entry["sequence"]).upper()
        name = ref_entry.get("ref_name", "<unknown>")
        if not re.fullmatch(r"[ACGT]+", seq or ""):
            raise ValueError(f"[{name}] invalid characters in sequence")

        if len(seq) % _TRIPLE != 0:
            raise ValueError(
                "scan_codon: sequence length must be divisible by 3 for codon scanning"
            )

        codons = [seq[i : i + _TRIPLE] for i in range(0, len(seq), _TRIPLE)]
        total = len(codons)

        rc = params.get("region_codons")
        if rc:
            start, end = int(rc[0]), int(rc[1])
            if not (0 <= start <= end <= total):
                raise ValueError(
                    f"scan_codon: region_codons out of bounds for {total} codons: {rc}"
                )
            scan_ix = range(start, end)
        else:
            scan_ix = range(total)

        for ci in scan_ix:
            wt = codons[ci]
            wt_aa = codon2aa.get(wt)
            if wt_aa is None:
                # unknown / stop codon not represented in table; skip
                continue
            for aa, codon_list in aa2codons.items():
                if aa == wt_aa or not codon_list:
                    continue
                new_codon = codon_list[0]
                mutated = codons.copy()
                mutated[ci] = new_codon
                yield {
                    "sequence": "".join(mutated),
                    "modifications": [f"codon i={ci} wt={wt} new={new_codon} aa={aa}"],
                    "codon_index": ci,
                    "codon_wt": wt,
                    "codon_new": new_codon,
                    "codon_aa": aa,
                }
