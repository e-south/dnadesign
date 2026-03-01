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
    "dense_array_video_showcase": {
        "fn": "plot_dense_array_video_showcase",
        "description": "Stage-B showcase video: sampled accepted outputs rendered as an MP4 timeline.",
        "requires": ["outputs"],
    },
    "placement_map": {
        "fn": "plot_placement_map",
        "description": "Stage-B fingerprint: per-position occupancy across accepted outputs.",
        "requires": ["outputs", "composition", "config"],
    },
    "tfbs_usage": {
        "fn": "plot_tfbs_usage",
        "description": "TFBS allocation summary across all placements (rank + distribution).",
        "requires": ["composition"],
    },
    "run_health": {
        "fn": "plot_run_health",
        "description": "Run health summary (outcomes, waste pressure, reason families, plan quota progress).",
        "requires": ["outputs", "composition", "attempts", "config"],
    },
    "stage_a_summary": {
        "fn": "plot_stage_a_summary",
        "description": "Stage-A pool quality, yield, bias, and core diversity summary.",
        "requires": ["pools"],
    },
}
