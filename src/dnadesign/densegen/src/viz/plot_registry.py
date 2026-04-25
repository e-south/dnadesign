"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/viz/plot_registry.py

Plot registry metadata (names + descriptions) without importing matplotlib.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

PLOT_SPECS = {
    "attempt_outcome_timeline": {
        "fn": "plot_attempt_outcome_timeline",
        "description": "Run diagnostics timeline showing accepted, duplicate, rejected, and failed attempts over time.",
        "label": "Attempt outcome timeline",
        "requires": ["attempts", "config"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/tables/attempts.parquet or attempts_part-*.parquet",
            "outputs/meta/effective_config.json or config.yaml fallback",
        ],
        "missing_hint": (
            "This plot needs workspace-local attempt traces. "
            "Sync or regenerate `outputs/tables` and `outputs/meta`, then run "
            "`uv run dense plot --only attempt_outcome_timeline`."
        ),
    },
    "background_sequence_logo": {
        "fn": "plot_background_sequence_logo",
        "description": "Stage-A background sequence logo showing the context sequence pool used during sampling.",
        "label": "Background sequence logo",
        "requires": ["pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet with background sequences",
        ],
        "missing_hint": (
            "This plot needs workspace-local Stage-A background pools. "
            "Sync or rebuild `outputs/pools`, then run `uv run dense plot --only background_sequence_logo`."
        ),
    },
    "compression_ratio_by_plan": {
        "fn": "plot_compression_ratio_by_plan",
        "description": "Run diagnostics drilldown showing compression-ratio distributions by plan.",
        "label": "Compression ratio by plan",
        "requires": ["outputs"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with densegen__compression_ratio and densegen__plan",
        ],
        "missing_hint": (
            "This plot reads compression-ratio annotations from DenseGen output records. "
            "Verify the selected records source carries `densegen__compression_ratio`, then run "
            "`uv run dense plot --only compression_ratio_by_plan`."
        ),
    },
    "dense_array_showcase_video": {
        "fn": "plot_dense_array_showcase_video",
        "description": "Opt-in DenseGen showcase video rendered from sampled accepted outputs.",
        "label": "Dense array showcase video",
        "requires": ["outputs"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail",
            "plots.video configuration",
        ],
        "missing_hint": (
            "This plot can be generated from DenseGen output records alone. "
            "Verify `plots.source` resolves to parquet or USR records with `densegen__used_tfbs_detail`, "
            "and that `plots.video` is configured, then run "
            "`uv run dense plot --only dense_array_showcase_video`."
        ),
    },
    "plan_regulator_deployment_heatmap": {
        "fn": "plot_plan_regulator_deployment_heatmap",
        "description": "Stage-B deployment heatmap across plans and regulators.",
        "label": "Plan regulator deployment heatmap",
        "requires": ["outputs"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with densegen__plan and densegen__used_tfbs_detail",
        ],
        "missing_hint": (
            "This summary reads deployed TFBS annotations from DenseGen output records. "
            "Verify the selected records source carries densegen__used_tfbs_detail, then run "
            "`uv run dense plot --only plan_regulator_deployment_heatmap`."
        ),
    },
    "placement_occupancy_map": {
        "fn": "plot_placement_occupancy_map",
        "description": "Stage-B positional occupancy map for accepted arrays.",
        "label": "Placement occupancy map",
        "requires": ["outputs", "composition", "config"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": True,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail",
            "outputs/meta/effective_config.json or config.yaml fallback",
        ],
        "missing_hint": (
            "This plot can recover placement composition from DenseGen output records plus config metadata. "
            "If it is still missing, verify the records source and local effective config, then run "
            "`uv run dense plot --only placement_occupancy_map`."
        ),
    },
    "retained_pool_coverage_by_regulator": {
        "fn": "plot_retained_pool_coverage_by_regulator",
        "description": (
            "Compare how many unique Stage-A TFBS were retained for each regulator with how many of those retained "
            "motifs were actually deployed into accepted DenseGen arrays."
        ),
        "label": "Retained pool coverage by regulator",
        "requires": ["outputs", "pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail",
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet",
        ],
        "missing_hint": (
            "This summary needs deployed TFBS annotations plus retained Stage-A pools. "
            "Sync or rebuild `outputs/pools`, verify the records source, then run "
            "`uv run dense plot --only retained_pool_coverage_by_regulator`."
        ),
    },
    "score_strata_and_deployed_length_bridge": {
        "fn": "plot_score_strata_and_deployed_length_bridge",
        "description": (
            "Bridge Stage A score strata to Stage B deployment by showing each regulator's eligible-vs-retained "
            "score distribution alongside the deployed TFBS length mix."
        ),
        "label": "Score strata and deployed length bridge",
        "requires": ["outputs", "pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail",
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet with best_hit_score and tfbs_core",
        ],
        "missing_hint": (
            "This bridge view needs deployed TFBS annotations, retained Stage-A pools, and Stage-A score histograms. "
            "Sync or rebuild `outputs/pools`, verify the records source, then run "
            "`uv run dense plot --only score_strata_and_deployed_length_bridge`."
        ),
    },
    "retained_vs_deployed_length_mix_by_regulator": {
        "fn": "plot_retained_vs_deployed_length_mix_by_regulator",
        "description": (
            "Compare each regulator's retained Stage-A TFBS length mix with the TFBS lengths that were actually "
            "deployed into accepted arrays."
        ),
        "label": "Retained vs deployed length mix by regulator",
        "requires": ["outputs", "pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail",
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet",
        ],
        "missing_hint": (
            "This summary needs deployed TFBS annotations plus retained Stage-A pools. "
            "Sync or rebuild `outputs/pools`, verify the records source, then run "
            "`uv run dense plot --only retained_vs_deployed_length_mix_by_regulator`."
        ),
    },
    "retained_vs_deployed_tier_mix_by_regulator": {
        "fn": "plot_retained_vs_deployed_tier_mix_by_regulator",
        "description": (
            "Compare each regulator's retained Stage-A score-tier mix with the mapped tier mix that was actually "
            "deployed into accepted arrays."
        ),
        "label": "Retained vs deployed tier mix by regulator",
        "requires": ["outputs", "pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail",
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet with tier assignments",
        ],
        "missing_hint": (
            "This summary needs deployed TFBS annotations and retained Stage-A tier metadata. "
            "Sync or rebuild `outputs/pools`, then run "
            "`uv run dense plot --only retained_vs_deployed_tier_mix_by_regulator`."
        ),
    },
    "solve_pressure_and_progress": {
        "fn": "plot_solve_pressure_and_progress",
        "description": (
            "The left panel counts failed-solve pressure by reason family, and the right panel shows accepted progress "
            "by plan."
        ),
        "label": "Solve pressure and progress",
        "requires": ["attempts", "config"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/tables/attempts.parquet or attempts_part-*.parquet",
            "outputs/meta/effective_config.json or config.yaml fallback",
        ],
        "missing_hint": (
            "This plot needs workspace-local run diagnostics, especially attempts tables. "
            "Sync or regenerate `outputs/tables` and `outputs/meta`, then run "
            "`uv run dense plot --only solve_pressure_and_progress`."
        ),
    },
    "source_cohort_concentration": {
        "fn": "plot_source_cohort_concentration",
        "description": "Break DenseGen arrays down by source-derived part-composition cohorts.",
        "label": "Source cohort concentration",
        "requires": ["outputs"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with source/densegen__plan/densegen__input_name columns",
        ],
        "missing_hint": (
            "This plot reads the selected DenseGen output records source directly. "
            "Verify `plots.source` resolves to parquet or USR records, then run "
            "`uv run dense plot --only source_cohort_concentration`."
        ),
    },
    "source_plan_input_heatmap": {
        "fn": "plot_source_plan_input_heatmap",
        "description": (
            "Dataset provenance drilldown showing source-to-plan and source-to-input relationships on a shared "
            "source-cohort axis."
        ),
        "label": "Source plan input heatmap",
        "requires": ["outputs"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with source/densegen__plan/densegen__input_name columns",
        ],
        "missing_hint": (
            "This plot reads the selected DenseGen output records source directly. "
            "Verify `plots.source` resolves to parquet or USR records, then run "
            "`uv run dense plot --only source_plan_input_heatmap`."
        ),
    },
    "stage_a_pool_diversity": {
        "fn": "plot_stage_a_pool_diversity",
        "description": "Stage-A health panel summarizing unique-core and motif diversity across retained pools.",
        "label": "Stage A pool diversity",
        "requires": ["pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet",
        ],
        "missing_hint": (
            "This plot needs workspace-local Stage-A pool artifacts. "
            "Sync or rebuild `outputs/pools`, then run `uv run dense plot --only stage_a_pool_diversity`."
        ),
    },
    "stage_a_pool_score_strata": {
        "fn": "plot_stage_a_pool_score_strata",
        "description": "Stage-A context panel comparing score strata and retained-pool composition.",
        "label": "Stage A pool score strata",
        "requires": ["pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet",
        ],
        "missing_hint": (
            "This plot needs workspace-local Stage-A pool artifacts. "
            "Sync or rebuild `outputs/pools`, then run `uv run dense plot --only stage_a_pool_score_strata`."
        ),
    },
    "stage_a_sampling_yield": {
        "fn": "plot_stage_a_sampling_yield",
        "description": "Stage-A health panel showing retained yield and regulator-level sampling bias.",
        "label": "Stage A sampling yield",
        "requires": ["pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet",
        ],
        "missing_hint": (
            "This plot needs workspace-local Stage-A pool artifacts. "
            "Sync or rebuild `outputs/pools`, then run `uv run dense plot --only stage_a_sampling_yield`."
        ),
    },
    "tfbs_concentration_profile": {
        "fn": "plot_tfbs_concentration_profile",
        "description": (
            "Stage-B allocation drilldown showing TFBS rank concentration, share decay, and used-vs-available behavior."
        ),
        "label": "TFBS concentration profile",
        "requires": ["composition"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": True,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail or outputs/tables/composition.parquet",
        ],
        "missing_hint": (
            "This plot can recover TFBS allocation from DenseGen output records "
            "when placement annotations are present. "
            "Verify the records source or local composition table, then run "
            "`uv run dense plot --only tfbs_concentration_profile`."
        ),
    },
    "upstream_motif_supply_and_pwm_strength": {
        "fn": "plot_upstream_motif_supply_and_pwm_strength",
        "description": (
            "The left panel compares source hits, eligible unique motifs, and "
            "retained Stage-A motifs per regulator, and the right panel shows "
            "each regulator's PWM consensus score as a fraction of that PWM's "
            "theoretical maximum."
        ),
        "label": "Upstream motif supply and PWM strength",
        "requires": ["pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/pools/pool_manifest.json with stage_a_sampling summaries",
        ],
        "missing_hint": (
            "This summary needs Stage-A sampling metadata from `outputs/pools/pool_manifest.json`. "
            "Sync or rebuild Stage-A pools, then run "
            "`uv run dense plot --only upstream_motif_supply_and_pwm_strength`."
        ),
    },
}
