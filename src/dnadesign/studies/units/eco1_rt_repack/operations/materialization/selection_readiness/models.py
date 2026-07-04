"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/models.py

Data models for Eco1 panel-selection materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedSelectionReadiness:
    """Paths emitted by panel-selection materialization."""

    feasibility_report_path: Path
    candidate_triage_table_path: Path
    candidate_selection_panel_path: Path
    candidate_handoff_sequence_csv_path: Path
    plots_root: Path
    manifest_path: Path
