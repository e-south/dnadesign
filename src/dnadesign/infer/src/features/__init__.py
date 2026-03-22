"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/__init__.py

Feature-contract helpers for higher-level infer workflows.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from .contracts import (
    FEATURE_SCHEMA_VERSION,
    OpalMatrixExport,
    PromoterContextConfig,
    PromoterDebugConfig,
    PromoterFeatureBundleConfig,
    PromoterPoolingConfig,
    SelectorResolution,
    SequenceContextRecord,
)
from .selectors import canonical_selector_for_block, provider_layer_for_block, resolve_intermediate_selector

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "OpalMatrixExport",
    "PromoterContextConfig",
    "PromoterDebugConfig",
    "PromoterFeatureBundleConfig",
    "PromoterPoolingConfig",
    "SelectorResolution",
    "SequenceContextRecord",
    "canonical_selector_for_block",
    "provider_layer_for_block",
    "resolve_intermediate_selector",
]
