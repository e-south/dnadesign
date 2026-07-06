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


def test_runtime_readme_names_current_selection_visual_count() -> None:
    root = repo_root()
    text = (root / "docs/studies/eco1_rt_repack/operations/runtime/command-groups/README.md").read_text(
        encoding="utf-8"
    )

    assert "eight selection-readiness SVGs" in text
    assert "seven selection-readiness SVGs" not in text
    assert "six selection-readiness SVGs" not in text
    assert "orders variants by SAE" not in text


def test_current_study_docs_do_not_reintroduce_removed_selection_surfaces() -> None:
    text = _current_selection_doc_text().lower()

    stale_fragments = (
        "full-population stratification",
        "handoff-boundary",
        "handoff boundary",
        "selection readiness and handoff boundary",
        "semantic stratification",
        "semantic review context",
        "semantic review evidence",
        "assay-panel stratification",
        "6b esmc llr scores rank candidates",
        "orders variants by sae",
    )
    for fragment in stale_fragments:
        assert fragment not in text


def _current_record_text() -> str:
    root = repo_root()
    return "\n".join((root / path).read_text(encoding="utf-8") for path in _CURRENT_RECORD_SURFACES)


def _current_selection_doc_text() -> str:
    root = repo_root()
    paths = (
        *_CURRENT_RECORD_SURFACES,
        Path("docs/studies/eco1_rt_repack/operations/runtime/command-groups/README.md"),
        Path("docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md"),
    )
    return "\n".join((root / path).read_text(encoding="utf-8") for path in paths)
