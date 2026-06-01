"""Stage B realized-label review artifact generation surface."""

from __future__ import annotations

from .contracts import TfbsStageBRealizedReviewResult
from .materialization import build_tfbs_stage_b_realized_label_review

__all__ = [
    "TfbsStageBRealizedReviewResult",
    "build_tfbs_stage_b_realized_label_review",
]
