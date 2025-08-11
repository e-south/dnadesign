"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/scan_dna.py

Nucleotide saturation scanning. For each position in the selected region(s),
emit variants that substitute the base with each of the other three nucleotides.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np

from . import register
from .base import Protocol

_ALPHABET = ("A", "C", "G", "T")


def _normalize_regions(seq_len: int, params: Dict) -> List[Tuple[int, int]]:
    """Resolve params.region / params.regions into a list of [start,end) pairs."""
    regions: List[Tuple[int, int]] = []
    if "regions" in params and params["regions"]:
        raw = params["regions"]
        if not isinstance(raw, list):
            raise ValueError(
                "scan_dna: params.regions must be a list of [start,end] pairs"
            )
        for r in raw:
            if not (isinstance(r, (list, tuple)) and len(r) == 2):
                raise ValueError(f"scan_dna: region must be [start,end], got {r!r}")
            s, e = int(r[0]), int(r[1])
            if not (0 <= s <= e <= seq_len) or s == e:
                raise ValueError(
                    f"scan_dna: region out of bounds for sequence length {seq_len}: {r}"
                )
            regions.append((s, e))
    else:
        r = params.get("region")
        if r is None:
            regions = [(0, seq_len)]
        else:
            if not (isinstance(r, (list, tuple)) and len(r) == 2):
                raise ValueError(f"scan_dna: region must be [start,end], got {r!r}")
            s, e = int(r[0]), int(r[1])
            if not (0 <= s <= e <= seq_len) or s == e:
                raise ValueError(
                    f"scan_dna: region out of bounds for sequence length {seq_len}: {r}"
                )
            regions = [(s, e)]
    return regions


@register
class ScanDNA(Protocol):
    id = "scan_dna"
    version = "1.0"

    def validate_cfg(self, *, params: Dict) -> None:
        # Type-shape checks only; bounds are validated in generate() with sequence length.
        if "regions" in params and params["regions"] is not None:
            if not isinstance(params["regions"], list):
                raise ValueError("scan_dna: params.regions must be a list")
            for r in params["regions"]:
                if not (isinstance(r, (list, tuple)) and len(r) == 2):
                    raise ValueError(f"scan_dna: region must be [start,end], got {r!r}")
        elif "region" in params and params["region"] is not None:
            r = params["region"]
            if not (isinstance(r, (list, tuple)) and len(r) == 2):
                raise ValueError(f"scan_dna: region must be [start,end], got {r!r}")

    def generate(
        self, *, ref_entry: Dict, params: Dict, rng: np.random.Generator
    ) -> Iterable[Dict]:
        raw_seq = ref_entry.get("sequence")
        name = ref_entry.get("ref_name", "<unknown>")

        if not isinstance(raw_seq, str):
            raise ValueError(
                f"[{name}] sequence must be a string, got {type(raw_seq).__name__}"
            )
        if not re.fullmatch(r"[ACGTacgt]+", raw_seq or ""):
            raise ValueError(f"[{name}] invalid characters in sequence: {raw_seq!r}")

        seq = raw_seq.upper()
        regions = _normalize_regions(len(seq), params)

        for start, end in regions:
            for idx, wt_nt in enumerate(seq[start:end], start=start):
                for alt_nt in _ALPHABET:
                    if alt_nt == wt_nt:
                        continue
                    mutated_seq = seq[:idx] + alt_nt + seq[idx + 1 :]
                    yield {
                        "sequence": mutated_seq,
                        "modifications": [f"nt pos={idx} wt={wt_nt} alt={alt_nt}"],
                        "nt_pos": idx,
                        "nt_wt": wt_nt,
                        "nt_alt": alt_nt,
                    }
