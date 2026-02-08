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
    "placement_map": {
        "fn": "plot_placement_map",
        "description": "Stage-B fingerprint: per-position occupancy + TFBS leaderboard.",
        "requires": ["outputs", "composition", "config", "libraries"],
    },
    "tfbs_usage": {
        "fn": "plot_tfbs_usage",
        "description": "TFBS allocation summary across all placements (rank + distribution).",
        "requires": ["composition"],
    },
    "run_health": {
        "fn": "plot_run_health",
        "description": "Run health summary (outcomes, waste pressure, reason families, plan quota progress).",
        "requires": ["attempts", "config"],
    },
    "stage_a_summary": {
        "fn": "plot_stage_a_summary",
        "description": "Stage-A pool quality, yield, bias, and core diversity summary.",
        "requires": ["pools"],
    },
}
