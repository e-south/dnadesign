"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/scan_codon.py

Codon-level saturation. For each codon in the selected region, replace with
the most frequent codon of every other amino acid from a usage table.

Sensible/robust CSV handling:
- Accepts columns: codon, amino_acid, and at least one of {frequency, fraction}.
- Uses `frequency` preferentially; falls back to `fraction` if `frequency` is missing/invalid.
- Skips stop codons (amino_acid == "*").
- Skips rows with invalid codons or non-numeric weights (e.g., "-").
- Builds:
    aa2codons: amino_acid -> list of codons sorted by descending weight
    codon2aa:  codon -> amino_acid
- Mutation rule: for each position, for every other amino acid present in the table,
  substitute the **most frequent** codon for that amino acid (one variant per AA).

Also emits AA-level metadata so downstream plots can operate in amino-acid space:
    aa_pos (1-based), aa_index (0-based), aa_wt, aa_alt,
and an "aa pos=.. wt=X alt=Y" token. Nucleotide tokens are also emitted for
backward compatibility.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

from . import register
from .base import Protocol

_TRIPLE = 3
_BASE_REQUIRED = {"codon", "amino_acid"}
_WEIGHT_ALIASES = ("frequency", "fraction")  # prefer frequency, fallback to fraction


def _parse_float_maybe(x: Optional[str]) -> Optional[float]:
    """
    Return float(x) if possible; otherwise None (handles '-', '', None).
    """
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
    """
    Load a codon usage table with columns:
        codon, amino_acid, [frequency], [fraction]

    Rules:
      - At least one of {frequency, fraction} must be present in the header.
      - We skip stop codons (amino_acid == "*").
      - We skip rows with invalid codons or non-numeric weights.
      - If both frequency and fraction exist, frequency is preferred.
    """
    path = Path(path).expanduser().resolve()
    aa2entries: Dict[str, List[Tuple[str, float]]] = {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [f.strip() for f in (reader.fieldnames or [])]

        if not _BASE_REQUIRED.issubset(fieldnames):
            raise ValueError(
                f"scan_codon: table must include columns {_BASE_REQUIRED}, got {fieldnames}"
            )
        if not any(w in fieldnames for w in _WEIGHT_ALIASES):
            raise ValueError(
                f"scan_codon: table must include at least one of {_WEIGHT_ALIASES}, got {fieldnames}"
            )

        # Normalize header access: DictReader keys use the original names
        for row in reader:
            aa = (row.get("amino_acid") or "").strip()
            if aa == "*":
                # skip stops
                continue

            codon = (row.get("codon") or "").strip().upper()
            if not re.fullmatch(r"[ACGT]{3}", codon):
                # skip malformed/ambiguous codons
                continue

            # prefer frequency, fallback to fraction
            weight: Optional[float] = None
            if "frequency" in row:
                weight = _parse_float_maybe(row.get("frequency"))
            if weight is None and "fraction" in row:
                weight = _parse_float_maybe(row.get("fraction"))
            if weight is None:
                # skip rows like "-" or missing numeric weight
                continue

            aa2entries.setdefault(aa, []).append((codon, weight))

    # Rank codons within each amino acid by descending weight
    aa2codons: Dict[str, List[str]] = {}
    codon2aa: Dict[str, str] = {}
    for aa, entries in aa2entries.items():
        if not entries:
            continue
        sorted_codons = sorted(entries, key=lambda cf: cf[1], reverse=True)
        aa2codons[aa] = [codon for codon, _ in sorted_codons]
        for c, _ in sorted_codons:
            codon2aa[c] = aa

    if not aa2codons:
        raise ValueError("scan_codon: no usable codon entries after filtering table")

    return aa2codons, codon2aa


@register
class ScanCodon(Protocol):
    id = "scan_codon"
    version = "1.2"

    def validate_cfg(self, *, params: Dict) -> None:
        # Required: codon_table path (must exist)
        table = params.get("codon_table")
        if not table:
            raise ValueError("scan_codon: params.codon_table is required")
        p = Path(str(table)).expanduser().resolve()
        if not p.exists():
            raise ValueError(
                f"scan_codon: params.codon_table must exist (checked: {p})"
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
        table = Path(str(params.get("codon_table"))).expanduser().resolve()
        aa2codons, codon2aa = _load_codon_table(table)

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
            wt_codon = codons[ci]
            wt_aa = codon2aa.get(wt_codon)
            if wt_aa is None:
                # unknown / not represented in table; skip this position
                continue

            # For each **other** amino acid, substitute its most frequent codon
            for aa, codon_list in aa2codons.items():
                if aa == wt_aa or not codon_list:
                    continue
                new_codon = codon_list[0]
                mutated = codons.copy()
                mutated[ci] = new_codon
                new_seq = "".join(mutated)

                # 1-based sequence index of first nt in this codon
                codon_start_1b = ci * 3 + 1

                # Build NT tokens for each base difference (keeps old plots working)
                nt_tokens: List[str] = []
                nt_changes: List[Tuple[int, str, str]] = []  # (pos_1b, wt_nt, new_nt)
                for off in range(3):
                    wt_nt = wt_codon[off]
                    new_nt = new_codon[off]
                    if wt_nt != new_nt:
                        pos_1b = codon_start_1b + off
                        nt_tokens.append(f"nt pos={pos_1b} wt={wt_nt} alt={new_nt}")
                        nt_changes.append((pos_1b, wt_nt, new_nt))

                # AA metadata (this is what the AA plot will use)
                aa_pos_1b = ci + 1
                aa_token = f"aa pos={aa_pos_1b} wt={wt_aa} alt={aa}"

                modifications = [
                    f"codon i={ci} wt={wt_codon} new={new_codon} aa={aa}",
                    aa_token,
                    *nt_tokens,
                ]

                out = {
                    "sequence": new_seq,
                    "modifications": modifications,
                    # codon-level fields
                    "codon_index": ci,       # 0-based codon index
                    "codon_wt": wt_codon,
                    "codon_new": new_codon,
                    "codon_aa": aa,          # target AA
                    # AA-level fields (preferred by AA plots)
                    "aa_index": ci,          # 0-based
                    "aa_pos": aa_pos_1b,     # 1-based
                    "aa_wt": wt_aa,
                    "aa_alt": aa,
                }

                # If exactly one nt changed, also surface structured nt_* columns
                if len(nt_changes) == 1:
                    pos_1b, nt_wt, nt_alt = nt_changes[0]
                    out["nt_pos"] = pos_1b
                    out["nt_wt"] = nt_wt
                    out["nt_alt"] = nt_alt

                yield out
