"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/protocols/scan_codon.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .base import Protocol

_TRIPLE = 3
_BASE_REQUIRED = {"codon", "amino_acid"}
_WEIGHT_ALIASES = ("frequency", "fraction")


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


def _load_codon_table(path: str | Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    path = Path(path).expanduser().resolve()
    aa2entries: Dict[str, List[Tuple[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = [f.strip() for f in (reader.fieldnames or [])]
        if not _BASE_REQUIRED.issubset(fields):
            raise ValueError(
                f"scan_codon: table must include columns {_BASE_REQUIRED}, got {fields}"
            )
        if not any(w in fields for w in _WEIGHT_ALIASES):
            raise ValueError(
                f"scan_codon: table must include at least one of {_WEIGHT_ALIASES}, got {fields}"
            )
        for row in reader:
            aa = (row.get("amino_acid") or "").strip()
            if aa == "*":
                continue
            codon = (row.get("codon") or "").strip().upper()
            if not re.fullmatch(r"[ACGT]{3}", codon):
                continue
            weight: Optional[float] = None
            if "frequency" in row:
                weight = _parse_float_maybe(row.get("frequency"))
            if weight is None and "fraction" in row:
                weight = _parse_float_maybe(row.get("fraction"))
            if weight is None:
                continue
            aa2entries.setdefault(aa, []).append((codon, weight))
    aa2codons: Dict[str, List[str]] = {}
    codon2aa: Dict[str, str] = {}
    for aa, entries in aa2entries.items():
        if not entries:
            continue
        ranked = sorted(entries, key=lambda t: t[1], reverse=True)
        aa2codons[aa] = [c for c, _ in ranked]
        for c, _ in ranked:
            codon2aa[c] = aa
    if not aa2codons:
        raise ValueError("scan_codon: no usable codon entries after filtering table")
    return aa2codons, codon2aa


class ScanCodon(Protocol):
    def validate_cfg(self, *, params: Dict) -> None:
        table = params.get("codon_table")
        if not table:
            raise ValueError("scan_codon: params.codon_table is required")
        p = Path(str(table)).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"scan_codon: codon_table not found at {p}")
        rc = params.get("region_codons")
        if rc is not None:
            if not (isinstance(rc, (list, tuple)) and len(rc) == 2):
                raise ValueError(
                    "scan_codon: region_codons must be [start,end] in codon units"
                )

    def generate(
        self,
        *,
        ref_entry: Dict,
        params: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Dict]:
        aa2codons, codon2aa = _load_codon_table(params["codon_table"])
        # Preserve original casing in outputs; operate case-insensitively.
        orig = str(ref_entry["sequence"])
        if not re.fullmatch(r"[ACGTacgt]+", orig or ""):
            raise ValueError("scan_codon: sequence contains non-ACGT symbols")
        if len(orig) % _TRIPLE != 0:
            raise ValueError("scan_codon: sequence length must be divisible by 3")
        seq_upper = orig.upper()
        codons_upper = [
            seq_upper[i : i + _TRIPLE] for i in range(0, len(seq_upper), _TRIPLE)
        ]
        total = len(codons_upper)
        rc = params.get("region_codons")
        scan_ix = range(int(rc[0]), int(rc[1])) if rc else range(total)
        region_start = int(rc[0]) if rc else 0

        for ci in scan_ix:
            wt_codon = codons_upper[ci]
            wt_aa = codon2aa.get(wt_codon)
            if wt_aa is None:
                continue
            for aa, variants in aa2codons.items():
                if aa == wt_aa or not variants:
                    continue
                new_codon = variants[0]  # uppercase
                # Apply codon swap while preserving per-char case from original
                chars = list(orig)
                codon_start = ci * 3
                for off in range(3):
                    orig_ch = chars[codon_start + off]
                    new_up = new_codon[off]
                    chars[codon_start + off] = (
                        new_up if orig_ch.isupper() else new_up.lower()
                    )
                new_seq = "".join(chars)
                codon_start_1b = ci * 3 + 1

                nt_tokens: List[str] = []
                nt_changes: List[Tuple[int, str, str]] = []
                for off in range(3):
                    wt_nt = wt_codon[off]
                    new_nt = new_codon[off]
                    if wt_nt != new_nt:
                        pos_1b = codon_start_1b + off
                        nt_tokens.append(f"nt pos={pos_1b} wt={wt_nt} alt={new_nt}")
                        nt_changes.append((pos_1b, wt_nt, new_nt))

                # AA positions:
                #  - aa_pos (absolute, 1-based across full protein)  → used by plots
                #  - aa_pos_rel (relative to scan window)            → kept for convenience
                aa_pos_abs_1b = ci + 1
                aa_pos_rel_1b = (ci - region_start) + 1
                aa_token = f"aa pos={aa_pos_abs_1b} wt={wt_aa} alt={aa}"

                out = {
                    "sequence": new_seq,
                    "modifications": [
                        f"codon i={ci} wt={wt_codon} new={new_codon} aa={aa}",
                        aa_token,
                        *nt_tokens,
                    ],
                    "codon_index": ci,
                    "codon_wt": wt_codon,
                    "codon_new": new_codon,
                    "codon_aa": aa,
                    "aa_index": ci,
                    "aa_pos": aa_pos_abs_1b,
                    "aa_pos_rel": aa_pos_rel_1b,
                    "aa_wt": wt_aa,
                    "aa_alt": aa,
                }
                if len(nt_changes) == 1:
                    pos_1b, nt_wt, nt_alt = nt_changes[0]
                    out["nt_pos"] = pos_1b
                    out["nt_wt"] = nt_wt
                    out["nt_alt"] = nt_alt
                yield out
