"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_materialization_assertions.py

Materialization assertions for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
    SELECTION_PLOT_METADATA,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._svg_assertions import (
    assert_svg_aspect_ratio_at_least,
    assert_svg_aspect_ratio_at_most,
)


def assert_selection_plot_contract(
    *,
    result: Any,
    manifest: dict[str, Any],
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
        assert "design_classes/selection" not in " ".join(str(source) for source in plot["data_sources"])
        if expected_metadata["not_a_selector_reason"]:
            assert plot["not_a_selector_reason"] == expected_metadata["not_a_selector_reason"]

    hypothesis_flow_text = plot_text_by_id["selection_hypothesis_panel_flow"]
    assert "Sequence generation and structural review produce the selected panel" in hypothesis_flow_text
    assert "ColabFold" in hypothesis_flow_text
    assert "local Cα RMSD ≤2.5 Å" in hypothesis_flow_text
    assert "Generation" in hypothesis_flow_text
    assert "policy" in hypothesis_flow_text
    assert "WT R13" not in hypothesis_flow_text
    assert "First order" not in hypothesis_flow_text
    assert "Distal scaffold" in hypothesis_flow_text
    assert "Peripheral shell" in hypothesis_flow_text
    assert "Combined" in hypothesis_flow_text
    assert "panel" in hypothesis_flow_text
    assert "ProteinMPNN" in hypothesis_flow_text
    assert "ColabFold" in hypothesis_flow_text
    assert "not selected" in hypothesis_flow_text
    for plot_id in ("selection_local_structure_by_region", "selection_regionwise_msa_support"):
        assert_svg_aspect_ratio_at_most(plot_text_by_id[plot_id], maximum=1.05)
    assert_svg_aspect_ratio_at_most(plot_text_by_id["selection_regional_mutation_burden"], maximum=1.15)
    selected_substitutions_text = plot_text_by_id["selection_selected_substitutions_across_rt"]
    assert_svg_aspect_ratio_at_least(selected_substitutions_text, minimum=2.2)
    assert selected_substitutions_text.count("Eco1 RT residue position") == 1
    mutation_distance_text = plot_text_by_id["selection_mutation_set_dissimilarity"]
    assert "All same-group candidate pairs" in mutation_distance_text
    assert "Selected same-group pairs" in mutation_distance_text
    assert "d_J" in mutation_distance_text
    assert "selection_local_structure_stratification" in plot_text_by_id
    local_structure_stratification_text = plot_text_by_id["selection_local_structure_stratification"]
    assert "review cutoff" in local_structure_stratification_text
    assert "Selected rows" in local_structure_stratification_text
    assert "selection_local_structure_threshold_sensitivity" in plot_text_by_id
    assert "selection_regionwise_msa_support" in plot_text_by_id
    assert "selection_near_region_charge_sensitivity" not in plot_text_by_id
    assert "selection_design_class_contrast" not in plot_text_by_id
    assert "selection_design_class_gate_counts" not in plot_text_by_id
    assert "selection_premise_alignment" not in plot_text_by_id
