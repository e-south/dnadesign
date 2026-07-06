"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_materialization_assertions.py

Materialization assertions for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._svg_assertions import (
    assert_heatmap_cells_are_square,
    assert_svg_has_square_panel,
)


def assert_selection_plot_contract(
    *,
    result: Any,
    manifest: dict[str, Any],
    retired_plot: Path,
) -> None:
    assert [plot["plot_id"] for plot in manifest["plots"]] == list(CURRENT_SELECTION_PLOT_IDS)
    plot_text_by_id: dict[str, str] = {}
    for plot in manifest["plots"]:
        plot_path = result.manifest_path.parent / plot["path"]
        assert plot_path.exists()
        plot_text = plot_path.read_text(encoding="utf-8")
        plot_text_by_id[str(plot["plot_id"])] = plot_text
        assert "<title" in plot_text
        assert plot["alt_text"].strip()
        assert plot["interpretation_limit"].strip()

    premise_text = plot_text_by_id["selection_premise_alignment"]
    assert "Core/direct edits" in premise_text
    assert "Local structure" in premise_text
    assert "ESMC/SAE" not in premise_text
    assert_heatmap_cells_are_square(premise_text, row_count=len(ALL_SPECS), column_count=7)

    gate_count_text = plot_text_by_id["selection_design_class_gate_counts"]
    assert "Passes protein gate" in gate_count_text
    assert "Blocked by gate" in gate_count_text
    assert "Missing gate input" in gate_count_text
    assert "Fold-review reserve" not in gate_count_text
    assert "Manual reserve" not in gate_count_text
    assert "Excluded" not in gate_count_text
    assert_svg_has_square_panel(gate_count_text)

    assert_heatmap_cells_are_square(
        plot_text_by_id["selection_class_local_percentiles"],
        row_count=len(ALL_SPECS),
        column_count=6,
    )
    assert_heatmap_cells_are_square(
        plot_text_by_id["selection_regional_mutation_burden"],
        row_count=len(ALL_SPECS),
        column_count=4,
    )
    assert_heatmap_cells_are_square(
        plot_text_by_id["selection_local_structure_by_region"],
        row_count=len(ALL_SPECS),
        column_count=len(LOCAL_STRUCTURE_REGION_IDS),
    )
    assert "selection_local_structure_stratification" in plot_text_by_id
    assert "Exploratory threshold" in plot_text_by_id["selection_local_structure_stratification"]
    assert not retired_plot.exists()
