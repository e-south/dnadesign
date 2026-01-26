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
    "compression_ratio": {
        "fn": "plot_compression_ratio",
        "description": "Histogram of compression ratios across sequences.",
    },
    "tf_usage": {
        "fn": "plot_tf_usage",
        "description": "TF usage summary (stacked by length/TFBS or totals).",
    },
    "pad_gc": {
        "fn": "plot_pad_gc",
        "description": "GC content target vs actual for pad bases.",
    },
    "plan_counts": {
        "fn": "plot_plan_counts",
        "description": "Plan counts over time by promoter constraint bucket.",
    },
    "tf_coverage": {
        "fn": "plot_tf_coverage",
        "description": "Per-base TFBS coverage across sequences.",
    },
    "tfbs_positional_frequency": {
        "fn": "plot_tfbs_positional_frequency",
        "description": "Positional frequency of TFBS placements (line plot).",
    },
    "tfbs_positional_histogram": {
        "fn": "plot_tfbs_positional_histogram",
        "description": "Positional TFBS histogram (overlaid, per-nt).",
    },
    "diversity_health": {
        "fn": "plot_diversity_health",
        "description": "Diversity health over time (unique TF/TFBS coverage and entropy).",
    },
    "tfbs_length_density": {
        "fn": "plot_tfbs_length_density",
        "description": "TFBS length distribution (histogram/KDE).",
    },
    "tfbs_usage": {
        "fn": "plot_tfbs_usage",
        "description": "TFBS usage by TF, ranked by occurrences.",
    },
    "stage_a_strata_overview": {
        "fn": "plot_stage_a_strata_overview",
        "description": "Stage-A strata overview (eligible scores + retained TFBS lengths).",
        "requires": ["pools"],
    },
    "run_timeline_funnel": {
        "fn": "plot_run_timeline_funnel",
        "description": "Run timeline funnel (attempts by status with resample/stall markers).",
        "requires": ["attempts", "events"],
    },
    "run_failure_pareto": {
        "fn": "plot_run_failure_pareto",
        "description": "Failure reason Pareto (overall + by plan).",
        "requires": ["attempts"],
    },
    "stage_b_library_health": {
        "fn": "plot_stage_b_library_health",
        "description": "Stage-B library health over builds (coverage, entropy, score, slack).",
        "requires": ["metrics"],
    },
    "stage_b_library_slack": {
        "fn": "plot_stage_b_library_slack",
        "description": "Stage-B library slack distribution (feasibility check).",
        "requires": ["metrics"],
    },
    "stage_a_score_traceability": {
        "fn": "plot_stage_a_score_traceability",
        "description": "Stage-A score traceability in solutions (tier + quantile enrichment).",
        "requires": ["metrics"],
    },
    "stage_b_offered_vs_used": {
        "fn": "plot_stage_b_offered_vs_used",
        "description": "Stage-B offered vs used TF utilization per library.",
        "requires": ["metrics"],
    },
    "stage_b_sampling_pressure": {
        "fn": "plot_stage_b_sampling_pressure",
        "description": "Stage-B sampling pressure heatmap (coverage weights + penalties).",
        "requires": ["metrics"],
    },
    "tfbs_positional_occupancy": {
        "fn": "plot_tfbs_positional_occupancy",
        "description": "Motif-aware positional occupancy with fixed-element overlays.",
    },
}
