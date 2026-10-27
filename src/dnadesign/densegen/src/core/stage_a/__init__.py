"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/stage_a/__init__.py

Stage-A sampling helpers and utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .stage_a_encoding import CoreEncodingStore, encode_cores
from .stage_a_progress import (
    StageAProgressManager,
    _BackgroundSamplingProgress,
    _format_stage_a_milestone,
    _PwmSamplingProgress,
)
from .stage_a_sampling_utils import (
    _background_cdf,
    _pwm_theoretical_max_score,
    _sample_background_batch,
    build_log_odds,
    normalize_background,
)
from .stage_a_types import FimoCandidate, PWMMotif, SelectionMeta

__all__ = [
    "FimoCandidate",
    "PWMMotif",
    "CoreEncodingStore",
    "SelectionMeta",
    "StageAProgressManager",
    "_BackgroundSamplingProgress",
    "_PwmSamplingProgress",
    "_background_cdf",
    "_format_stage_a_milestone",
    "_pwm_theoretical_max_score",
    "_sample_background_batch",
    "build_log_odds",
    "encode_cores",
    "normalize_background",
]
