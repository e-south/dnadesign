"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/qc/motifs.py

Motif marker calls for Eco1 retron-RT conservation source QC.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_MOTIF_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "rt_catalytic_dd_or_yadd_like_region": (r"YADD", r"[A-Z]{2}DD"),
    "retron_x_naxxH_like_motif": (r"NA..H",),
    "retron_y_vtg_like_motif": (r"VTG",),
}


def call_motif_markers(sequence: str, marker_ids: Iterable[str]) -> dict[str, str]:
    """Return present/absent calls for declared motif-QC marker ids."""

    calls: dict[str, str] = {}
    for marker_id in marker_ids:
        patterns = _MOTIF_PATTERNS.get(marker_id)
        if patterns is None:
            raise ValueError(f"unsupported motif_qc_marker {marker_id!r}")
        calls[marker_id] = "present" if any(re.search(pattern, sequence) for pattern in patterns) else "absent"
    return calls
