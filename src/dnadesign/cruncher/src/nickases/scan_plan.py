"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/nickases/scan_plan.py

Precomputed motif scan plans for nickase recognition-site scanning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dnadesign.cruncher.nickases.models import (
    NickaseCatalogEntry,
    iupac_bases_for_symbol,
    normalize_iupac,
    reverse_complement_iupac,
)


@dataclass(frozen=True)
class NickaseOrientationScanPlan:
    orientation: Literal["forward", "reverse"]
    motif_text: str
    motif_len: int
    allowed_bases_by_position: tuple[frozenset[str], ...]


def _allowed_bases_by_position(motif_text: str) -> tuple[frozenset[str], ...]:
    return tuple(frozenset(iupac_bases_for_symbol(symbol)) for symbol in motif_text)


@lru_cache(maxsize=256)
def _cached_scan_plans(motif_top_5to3: str) -> tuple[NickaseOrientationScanPlan, ...]:
    motif_text = normalize_iupac(motif_top_5to3)
    reverse_text = reverse_complement_iupac(motif_text)
    plans = [
        NickaseOrientationScanPlan(
            orientation="forward",
            motif_text=motif_text,
            motif_len=len(motif_text),
            allowed_bases_by_position=_allowed_bases_by_position(motif_text),
        )
    ]
    if reverse_text != motif_text:
        plans.append(
            NickaseOrientationScanPlan(
                orientation="reverse",
                motif_text=reverse_text,
                motif_len=len(reverse_text),
                allowed_bases_by_position=_allowed_bases_by_position(reverse_text),
            )
        )
    return tuple(plans)


def build_scan_plans(sequence_top_5to3: str) -> tuple[NickaseOrientationScanPlan, ...]:
    return _cached_scan_plans(sequence_top_5to3)


def build_entry_scan_plans(
    entry: NickaseCatalogEntry,
    *,
    use_vendor_diagram: bool = False,
) -> tuple[NickaseOrientationScanPlan, ...]:
    if use_vendor_diagram:
        return _cached_scan_plans(entry.resolved_vendor_diagram_top_5to3)
    return _cached_scan_plans(entry.motif_top_5to3)


def window_matches_plan(sequence: str, *, start: int, plan: NickaseOrientationScanPlan) -> bool:
    end = start + plan.motif_len
    if start < 0 or end > len(sequence):
        return False
    for offset, allowed in enumerate(plan.allowed_bases_by_position):
        if sequence[start + offset] not in allowed:
            return False
    return True


__all__ = [
    "NickaseOrientationScanPlan",
    "build_scan_plans",
    "build_entry_scan_plans",
    "window_matches_plan",
]
