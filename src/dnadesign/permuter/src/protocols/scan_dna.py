"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/protocols/scan_dna.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .base import Protocol, assert_dna

_ALPHABET = ("A", "C", "G", "T")


def _normalize_regions(seq_len: int, params: Dict) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    raw = params.get("regions")
    if not raw:
        return [(0, seq_len)]
    if not isinstance(raw, list):
        raise ValueError("scan_dna.params.regions must be a list of [start,end] pairs")
    for r in raw:
        if not (isinstance(r, (list, tuple)) and len(r) == 2):
            raise ValueError(f"region must be [start,end], got {r!r}")
        s, e = int(r[0]), int(r[1])
        if not (0 <= s < e <= seq_len):
            raise ValueError(f"region out of bounds for length {seq_len}: {r}")
        regions.append((s, e))
    return regions


class ScanDNA(Protocol):
    def validate_cfg(self, *, params: Dict) -> None:
        pass

    def generate(
        self,
        *,
        ref_entry: Dict,
        params: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Dict]:
        # Preserve original casing in outputs; operate case-insensitively.
        orig = str(ref_entry["sequence"])
        assert_dna(orig)
        regions = _normalize_regions(len(orig), params)
        for start, end in regions:
            for idx in range(start, end):
                wt_char = orig[idx]
                wt_upper = wt_char.upper()
                for alt_upper in _ALPHABET:
                    if alt_upper == wt_upper:
                        continue
                    # Change just this position, matching original casing
                    alt_char = alt_upper if wt_char.isupper() else alt_upper.lower()
                    mutated = orig[:idx] + alt_char + orig[idx + 1 :]
                    yield {
                        "sequence": mutated,
                        "modifications": [
                            f"nt pos={idx+1} wt={wt_upper} alt={alt_upper}"
                        ],  # 1-based
                        "nt_pos": idx + 1,
                        "nt_wt": wt_upper,
                        "nt_alt": alt_upper,
                    }
