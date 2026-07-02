"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/models.py

Typed values for Eco1 SAE window-summary materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WindowSpec:
    """A named canonical-position window for local SAE review."""

    window_id: str
    window_label: str
    residue_positions_1based: tuple[int, ...]
    purpose: str


@dataclass(frozen=True)
class MaterializedSaeWindowSummary:
    """Paths emitted by SAE window-summary materialization."""

    summary_path: Path
    manifest_path: Path
