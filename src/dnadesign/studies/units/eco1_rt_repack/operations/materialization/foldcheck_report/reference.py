"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/reference.py

Resolve the Eco1 residue correspondence used for reference-backed fold checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


def mapped_reference_positions(residue_map_path: Path) -> tuple[int, ...]:
    """Return ordered canonical positions represented by the Eco1 reference backbone."""

    if not residue_map_path.exists():
        raise FileNotFoundError(residue_map_path)
    rows = pq.read_table(residue_map_path, columns=["canonical_position", "mapping_status"]).to_pylist()
    positions = sorted(int(row["canonical_position"]) for row in rows if str(row.get("mapping_status")) == "mapped")
    if not positions:
        raise ValueError(f"residue map has no mapped canonical positions: {residue_map_path}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"residue map contains duplicate mapped canonical positions: {residue_map_path}")
    return tuple(positions)


__all__ = ["mapped_reference_positions"]
