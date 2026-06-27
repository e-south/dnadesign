"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/mask_rows.py

Mask-row loading for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_mask_residues(path: Path) -> list[dict[str, Any]]:
    """Read generated Eco1 mask residue rows."""

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("residues"), list):
        raise ValueError(f"Expected mask_set residues in {path}")
    return [row for row in loaded["residues"] if isinstance(row, dict)]
