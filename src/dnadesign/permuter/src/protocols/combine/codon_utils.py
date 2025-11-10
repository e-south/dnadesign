"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/combine/codon_utils.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_BASE_REQUIRED = {"codon", "amino_acid"}
_WEIGHT_ALIASES = ("frequency", "fraction")


@dataclass(frozen=True)
class CodonTable:
    aa2codons: Dict[str, List[str]]
    codon2aa: Dict[str, str]
    aa2weights: Dict[
        str, List[float]
    ]  # normalized (sum to 1.0), same order as aa2codons[aa]


def _parse_float_maybe(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except Exception:
        return None


def load_codon_table(path: str | Path) -> CodonTable:
    """
    Strict CSV loader for codon usage.
    Requires columns: codon, amino_acid and one of {frequency|fraction}.
    Returns:
      - aa2codons[AA]  = list of codons ranked by weight desc
      - codon2aa[CODON] = AA
      - aa2weights[AA] = normalized weights aligned to aa2codons[AA]
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"combine_aa: codon_table not found at {p}")
    aa2entries: Dict[str, List[Tuple[str, float]]] = {}
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = [f.strip() for f in (reader.fieldnames or [])]
        if not _BASE_REQUIRED.issubset(fields):
            raise ValueError(
                f"combine_aa: codon table must include columns {_BASE_REQUIRED}, got {fields}"
            )
        if not any(w in fields for w in _WEIGHT_ALIASES):
            raise ValueError(
                f"combine_aa: codon table must include at least one of {_WEIGHT_ALIASES}, got {fields}"
            )
        for row in reader:
            aa = (row.get("amino_acid") or "").strip().upper()
            if not aa or aa == "*":
                continue  # ignore stops here
            codon = (row.get("codon") or "").strip().upper()
            if not re.fullmatch(r"[ACGT]{3}", codon):
                continue
            w = _parse_float_maybe(row.get("frequency")) or _parse_float_maybe(
                row.get("fraction")
            )
            if w is None:
                continue
            aa2entries.setdefault(aa, []).append((codon, float(w)))
    if not aa2entries:
        raise ValueError("combine_aa: codon table yielded no usable entries")

    aa2codons: Dict[str, List[str]] = {}
    aa2weights: Dict[str, List[float]] = {}
    codon2aa: Dict[str, str] = {}

    for aa, entries in aa2entries.items():
        ranked = sorted(entries, key=lambda t: t[1], reverse=True)
        codons = [c for c, _ in ranked]
        weights = np.array([w for _, w in ranked], dtype=float)
        if weights.sum() <= 0:
            raise ValueError(f"combine_aa: non-positive weights for amino acid {aa}")
        weights = (weights / weights.sum()).tolist()
        aa2codons[aa] = codons
        aa2weights[aa] = weights
        for c in codons:
            codon2aa[c] = aa

    return CodonTable(aa2codons=aa2codons, codon2aa=codon2aa, aa2weights=aa2weights)


def aa_to_best_codon(tbl: CodonTable, aa: str) -> str:
    aa = (aa or "").upper()
    if aa not in tbl.aa2codons or not tbl.aa2codons[aa]:
        raise ValueError(f"combine_aa: no codon available for amino acid {aa!r}")
    return tbl.aa2codons[aa][0]


def aa_to_weighted_codon(tbl: CodonTable, aa: str, rng: np.random.Generator) -> str:
    aa = (aa or "").upper()
    if aa not in tbl.aa2codons or not tbl.aa2codons[aa]:
        raise ValueError(f"combine_aa: no codon available for amino acid {aa!r}")
    codons = tbl.aa2codons[aa]
    weights = np.array(tbl.aa2weights[aa], dtype=float)
    idx = int(rng.choice(len(codons), p=weights))
    return codons[idx]
