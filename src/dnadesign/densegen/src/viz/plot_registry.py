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
    "dataset_source_inventory": {
        "fn": "plot_dataset_source_inventory",
        "description": "Dataset-native source inventory and DenseGen metadata coverage summary.",
        "requires": ["outputs"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with source/densegen__plan/densegen__input_name columns",
        ],
        "missing_hint": (
            "This plot reads the selected DenseGen output records source directly. "
            "Verify `plots.source` resolves to parquet or USR records, then run "
            "`uv run dense plot --only dataset_source_inventory`."
        ),
    },
    "dataset_metadata_heatmap": {
        "fn": "plot_dataset_metadata_heatmap",
        "description": "Dataset-native provenance heatmaps for source-to-plan and source-to-input relationships.",
        "requires": ["outputs"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "selected output records source with source/densegen__plan/densegen__input_name columns",
        ],
        "missing_hint": (
            "This plot reads the selected DenseGen output records source directly. "
            "Verify `plots.source` resolves to parquet or USR records, then run "
            "`uv run dense plot --only dataset_metadata_heatmap`."
        ),
    },
    "dense_array_video_showcase": {
        "fn": "plot_dense_array_video_showcase",
        "description": "Stage-B showcase video: sampled accepted outputs rendered as an MP4 timeline.",
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
            "and that `plots.video` is configured, "
            "then run `uv run dense plot --only dense_array_video_showcase`."
        ),
    },
    "placement_map": {
        "fn": "plot_placement_map",
        "description": "Stage-B fingerprint: per-position occupancy across accepted outputs.",
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
            "`uv run dense plot --only placement_map`."
        ),
    },
    "tfbs_usage": {
        "fn": "plot_tfbs_usage",
        "description": "TFBS allocation summary across all placements (rank + distribution).",
        "requires": ["composition"],
        "missing_state": "recoverable_read_only",
        "seed_stage_b_scope_when_missing": True,
        "required_artifacts": [
            "selected output records source with densegen__used_tfbs_detail or outputs/tables/composition.parquet",
        ],
        "missing_hint": (
            "This plot can recover TFBS usage from DenseGen output records when placement annotations are present. "
            "Verify the records source or local composition table, then run `uv run dense plot --only tfbs_usage`."
        ),
    },
    "run_health": {
        "fn": "plot_run_health",
        "description": "Run health summary (outcomes, waste pressure, reason families, plan quota progress).",
        "requires": ["outputs", "composition", "attempts", "config"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/tables/attempts.parquet or attempts_part-*.parquet",
            "outputs/meta/effective_config.json or config.yaml fallback",
            "outputs/tables/composition.parquet or output-record placement annotations",
        ],
        "missing_hint": (
            "This plot needs workspace-local analysis artifacts, especially attempts tables. "
            "Sync or regenerate `outputs/tables` and `outputs/meta`, then run `uv run dense plot --only run_health`."
        ),
    },
    "stage_a_summary": {
        "fn": "plot_stage_a_summary",
        "description": "Stage-A pool quality, yield, bias, and core diversity summary.",
        "requires": ["pools"],
        "missing_state": "requires_local_artifacts",
        "seed_stage_b_scope_when_missing": False,
        "required_artifacts": [
            "outputs/pools/pool_manifest.json",
            "outputs/pools/*__pool.parquet",
        ],
        "missing_hint": (
            "This plot needs workspace-local Stage-A pool artifacts. "
            "Sync or rebuild `outputs/pools`, then run `uv run dense plot --only stage_a_summary`."
        ),
    },
}
