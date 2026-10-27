"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_registry.py

Plot registry coverage for the canonical DenseGen plot surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.viz.plot_registry import PLOT_SPECS


def test_plot_registry_has_descriptions() -> None:
    for name, meta in PLOT_SPECS.items():
        assert "description" in meta, f"Missing description for plot '{name}'"
        assert str(meta["description"]).strip(), f"Empty description for plot '{name}'"


def test_plot_registry_is_canonical_set() -> None:
    assert set(PLOT_SPECS.keys()) == {
        "attempt_outcome_timeline",
        "background_sequence_logo",
        "compression_ratio_by_plan",
        "dense_array_showcase_video",
        "plan_regulator_deployment_heatmap",
        "placement_occupancy_map",
        "retained_pool_coverage_by_regulator",
        "retained_vs_deployed_length_mix_by_regulator",
        "retained_vs_deployed_tier_mix_by_regulator",
        "score_strata_and_deployed_length_bridge",
        "solve_pressure_and_progress",
        "source_cohort_concentration",
        "source_plan_input_heatmap",
        "stage_a_pool_diversity",
        "stage_a_pool_score_strata",
        "stage_a_sampling_yield",
        "tfbs_concentration_profile",
        "upstream_motif_supply_and_pwm_strength",
    }
