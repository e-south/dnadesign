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
    SELECTION_PLOT_METADATA,
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
        metadata_text = " ".join(
            str(plot.get(field) or "") for field in ("title", "description", "alt_text", "interpretation_limit")
        )
        assert "mask class" not in metadata_text.lower()
        assert "annulus" not in metadata_text.lower()
        plot_path = result.manifest_path.parent / plot["path"]
        assert plot_path.exists()
        plot_text = plot_path.read_text(encoding="utf-8")
        plot_text_by_id[str(plot["plot_id"])] = plot_text
        assert "<title" in plot_text
        assert plot["alt_text"].strip()
        assert plot["interpretation_limit"].strip()
        expected_metadata = SELECTION_PLOT_METADATA[str(plot["plot_id"])]
        assert plot["selection_role"] == expected_metadata["selection_role"]
        assert plot["notebook_group"] == expected_metadata["notebook_group"]
        if expected_metadata["not_a_selector_reason"]:
            assert plot["not_a_selector_reason"] == expected_metadata["not_a_selector_reason"]

    premise_text = plot_text_by_id["selection_premise_alignment"]
    assert "Core/direct edits" in premise_text
    assert "Near retained DNA/RNA edits" in premise_text
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

    primary_sankey_text = plot_text_by_id["selection_primary_panel_sankey"]
    assert "Broad protein" in primary_sankey_text
    assert "Primary panel" in primary_sankey_text
    assert "Selected primary" in primary_sankey_text
    assert "design-class quota" in primary_sankey_text
    assert_heatmap_cells_are_square(
        plot_text_by_id["selection_regional_mutation_burden"],
        row_count=len(ALL_SPECS),
        column_count=5,
    )
    assert_heatmap_cells_are_square(
        plot_text_by_id["selection_local_structure_by_region"],
        row_count=len(ALL_SPECS),
        column_count=len(LOCAL_STRUCTURE_REGION_IDS),
    )
    assert "selection_local_structure_stratification" in plot_text_by_id
    local_structure_stratification_text = plot_text_by_id["selection_local_structure_stratification"]
    assert "Threshold" in local_structure_stratification_text
    assert "Selected rows" in local_structure_stratification_text
    assert "selection_local_structure_threshold_sensitivity" in plot_text_by_id
    assert "selection_regionwise_msa_support" in plot_text_by_id
    assert_heatmap_cells_are_square(
        plot_text_by_id["selection_regionwise_msa_support"],
        row_count=len(ALL_SPECS),
        column_count=5,
    )
    assert not retired_plot.exists()
