"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/__init__.py

Response-metric metastudy evaluation routines.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from . import window_evidence
from .grouped_models import CAMPAIGN_MODEL_SCREEN_ID, DEFAULT_MODEL_SCREEN_SPECS, ModelScreenSpec
from .model_representations import (
    build_label_representations,
    decode_to_response_magnitude,
    response_magnitude_to_factorial_contrast7,
)
from .model_screen import screen_label_models
from .multistate_behavior_cohort import (
    VerifiedBehaviorCohortReceipt,
    behavior_cohort_unit_ids_sha256,
    behavior_normalization_source_rows_sha256,
)
from .multistate_behavior_comparison import (
    build_repeated_behavior_agreement,
    compare_hard_and_behavior_scores,
)
from .multistate_behavior_event import build_multistate_behavior_event_sensitivity
from .multistate_behavior_normalization import (
    build_multistate_behavior_normalization_record,
    derive_multistate_behavior_normalization,
    verify_multistate_behavior_normalization_source,
)
from .multistate_behavior_protocol import BehaviorProtocolError, load_multistate_behavior_protocol
from .multistate_behavior_rows import bootstrap_rows_with_identity
from .multistate_behavior_shadow import build_multistate_behavior_shadow_evidence
from .multistate_behavior_stability import build_bootstrap_rank_stability
from .observed_sfxi_replay import (
    ObservedSfxiViewContext,
    build_observed_sfxi_decomposition,
    summarize_observed_sfxi_decomposition,
)
from .sfxi_greedy_replay import build_historical_sfxi_greedy_replay

__all__ = [
    "CAMPAIGN_MODEL_SCREEN_ID",
    "DEFAULT_MODEL_SCREEN_SPECS",
    "ModelScreenSpec",
    "ObservedSfxiViewContext",
    "VerifiedBehaviorCohortReceipt",
    "behavior_cohort_unit_ids_sha256",
    "behavior_normalization_source_rows_sha256",
    "build_label_representations",
    "build_multistate_behavior_normalization_record",
    "build_multistate_behavior_event_sensitivity",
    "build_repeated_behavior_agreement",
    "build_multistate_behavior_shadow_evidence",
    "build_observed_sfxi_decomposition",
    "build_historical_sfxi_greedy_replay",
    "build_bootstrap_rank_stability",
    "compare_hard_and_behavior_scores",
    "bootstrap_rows_with_identity",
    "decode_to_response_magnitude",
    "derive_multistate_behavior_normalization",
    "BehaviorProtocolError",
    "load_multistate_behavior_protocol",
    "response_magnitude_to_factorial_contrast7",
    "screen_label_models",
    "summarize_observed_sfxi_decomposition",
    "verify_multistate_behavior_normalization_source",
    "window_evidence",
]
