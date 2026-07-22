"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_contracts.py

Typed contracts for response metric metastudy plot deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .plot_narrative import PLOT_DATA_TABLES, PLOT_NON_CLAIM_BOUNDARIES, PLOT_RATIONALES

PlotTier = Literal["primary_decision", "metric_diagnostic", "screen_appendix"]
ReviewSection = Literal["assay_and_labels", "historical_model_screens", "rmf_comparator", "sfxi_comparator"]
PLOT_TIER_DIRS: dict[PlotTier, str] = {
    "primary_decision": "primary",
    "metric_diagnostic": "diagnostics",
    "screen_appendix": "appendix",
}


@dataclass(frozen=True)
class PlotSpec:
    plot_id: str
    filename: str
    tier: PlotTier
    review_section: ReviewSection
    section_order: int
    visual_type: str
    display_title: str
    premise: str
    decision_value: str
    alt_text: str
    review_step: int | None = None

    @property
    def title(self) -> str:
        return self.display_title.rstrip(".")

    @property
    def rationale(self) -> str:
        return PLOT_RATIONALES[self.plot_id]

    @property
    def non_claim_boundary(self) -> str:
        return PLOT_NON_CLAIM_BOUNDARIES[self.plot_id]

    @property
    def data_table(self) -> str:
        return PLOT_DATA_TABLES[self.plot_id]
