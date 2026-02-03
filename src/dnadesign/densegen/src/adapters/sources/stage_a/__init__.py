"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a/__init__.py

Stage-A sampling helpers and utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .stage_a_candidate_store import write_candidate_records, write_fimo_debug_tsv
from .stage_a_diversity import _diversity_summary
from .stage_a_encoding import CoreEncodingStore, encode_cores
from .stage_a_metadata import TFBSMeta
from .stage_a_metrics import _mmr_objective, _tail_unique_slope
from .stage_a_mining import mine_pwm_candidates
from .stage_a_paths import safe_label
from .stage_a_pipeline import run_stage_a_pipeline
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
from .stage_a_selection import (
    _core_sequence,
    _select_by_mmr,
    _select_max_diversity,
    _select_top,
)
from .stage_a_summary import PWMSamplingSummary
from .stage_a_types import FimoCandidate, PWMMotif, SelectionMeta

__all__ = [
    "FimoCandidate",
    "PWMMotif",
    "PWMSamplingSummary",
    "CoreEncodingStore",
    "SelectionMeta",
    "StageAProgressManager",
    "TFBSMeta",
    "_BackgroundSamplingProgress",
    "_PwmSamplingProgress",
    "_background_cdf",
    "_core_sequence",
    "_diversity_summary",
    "_format_stage_a_milestone",
    "_mmr_objective",
    "_pwm_theoretical_max_score",
    "_sample_background_batch",
    "_select_by_mmr",
    "_select_max_diversity",
    "_select_top",
    "_tail_unique_slope",
    "build_log_odds",
    "encode_cores",
    "mine_pwm_candidates",
    "normalize_background",
    "run_stage_a_pipeline",
    "safe_label",
    "write_candidate_records",
    "write_fimo_debug_tsv",
]
