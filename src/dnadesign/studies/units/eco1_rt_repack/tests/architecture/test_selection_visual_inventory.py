"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_selection_visual_inventory.py

Selection-visual inventory tests for Eco1 RT repack study records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
    RETIRED_SELECTION_PLOT_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_CURRENT_RECORD_SURFACES = (
    Path("docs/studies/eco1_rt_repack/record/status.md"),
    Path("docs/studies/eco1_rt_repack/record/datasets.yaml"),
    Path("docs/studies/eco1_rt_repack/routes/README.md"),
    Path("docs/studies/eco1_rt_repack/operations/runtime/command-groups/pipeline.yaml"),
)


def test_current_study_records_name_current_selection_plots() -> None:
    text = _current_record_text()

    for plot_id in CURRENT_SELECTION_PLOT_IDS:
        assert plot_id in text


def test_current_study_records_do_not_name_retired_selection_plots() -> None:
    text = _current_record_text()

    for plot_id in RETIRED_SELECTION_PLOT_IDS:
        assert plot_id not in text


def _current_record_text() -> str:
    root = repo_root()
    return "\n".join((root / path).read_text(encoding="utf-8") for path in _CURRENT_RECORD_SURFACES)
