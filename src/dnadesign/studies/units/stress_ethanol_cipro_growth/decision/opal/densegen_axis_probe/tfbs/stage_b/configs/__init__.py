"""Stage B sentinel OPAL config generation surface."""

from __future__ import annotations

from .contracts import TfbsStageBConfig, TfbsStageBResult
from .materialization import materialize_tfbs_stage_b_sentinel_configs

__all__ = [
    "TfbsStageBConfig",
    "TfbsStageBResult",
    "materialize_tfbs_stage_b_sentinel_configs",
]
