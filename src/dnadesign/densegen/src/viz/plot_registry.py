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
    "gap_fill_gc": {
        "fn": "plot_gap_fill_gc",
        "description": "GC content target vs actual for gap-fill pads.",
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
}
