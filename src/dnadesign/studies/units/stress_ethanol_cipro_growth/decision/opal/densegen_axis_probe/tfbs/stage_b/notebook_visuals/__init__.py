"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/notebook_visuals/__init__.py

TFBS Stage B notebook visual registration surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .learning_loop import TfbsProbeQuestionLearningLoopSource
from .portfolio import (
    TfbsProbeQuestionReviewSource,
    TfbsStageBReviewPortfolioResult,
    write_tfbs_stage_b_review_portfolio,
)
from .registration import (
    maybe_register_tfbs_stage_b_realized_review_visuals,
    maybe_register_tfbs_stage_b_slot_diagnostic_visuals,
    register_tfbs_stage_b_realized_review_visuals,
    register_tfbs_stage_b_slot_diagnostic_visuals,
)
from .specs import (
    StageBNotebookVisualSpec,
    realized_visual_spec,
    slot_visual_spec,
    slug_token,
)

__all__ = [
    "StageBNotebookVisualSpec",
    "TfbsProbeQuestionLearningLoopSource",
    "TfbsProbeQuestionReviewSource",
    "TfbsStageBReviewPortfolioResult",
    "maybe_register_tfbs_stage_b_realized_review_visuals",
    "maybe_register_tfbs_stage_b_slot_diagnostic_visuals",
    "realized_visual_spec",
    "register_tfbs_stage_b_realized_review_visuals",
    "register_tfbs_stage_b_slot_diagnostic_visuals",
    "slot_visual_spec",
    "slug_token",
    "write_tfbs_stage_b_review_portfolio",
]
