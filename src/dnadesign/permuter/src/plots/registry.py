"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/registry.py

Supported plot identifiers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

SUPPORTED_PLOT_IDS = (
    "position_scatter_and_heatmap",
    "ranked_variants",
    "synergy_scatter",
    "metric_by_mutation_count",
    "aa_category_effects",
    "hairpin_length_vs_metric",
)


def supported_plot_ids() -> tuple[str, ...]:
    return SUPPORTED_PLOT_IDS


def assert_supported_plot_id(plot_id: str) -> str:
    name = str(plot_id or "").strip()
    if name not in SUPPORTED_PLOT_IDS:
        raise ValueError(f"Unknown plot {plot_id!r}. Supported plots: {', '.join(SUPPORTED_PLOT_IDS)}")
    return name
