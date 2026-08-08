"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/config.py

Canonical import surface for Construct configuration contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .base import StrictConfigModel
from .datasets import InputConfig, USRDatasetLocatorConfig
from .job import InnerJobConfig, JobConfig
from .loader import load_job_config
from .normalize_anchor import (
    AnnotationFeatureCenterSelectorConfig,
    AnnotationPairMidpointSelectorConfig,
    FallbackPolicyConfig,
    FeatureMatchConfig,
    FeatureRetentionPolicyConfig,
    NormalizeAnchorConfig,
    NormalizeAnchorSelectorConfig,
    NormalizeTemplateConfig,
    OutputSequenceViewConfig,
    OverLengthTrimPolicyConfig,
    SelectorChainConfig,
    SequenceMidpointSelectorConfig,
    SequenceOffsetSelectorConfig,
    UnderLengthExpandFromTemplatePolicyConfig,
)
from .output import OutputConfig, OutputVariantConfig
from .parts import (
    CoordinatePlacementLocatorConfig,
    FlankPlacementLocatorConfig,
    PartConfig,
    PartSequenceConfig,
    PlacementConfig,
    PlacementGuardsConfig,
    PlacementLocatorConfig,
)
from .realization import RealizeConfig, WindowConfig
from .templates import (
    TemplateConfig,
    TemplateLiteralSourceConfig,
    TemplatePathSourceConfig,
    TemplateSourceConfig,
    TemplateUSRSourceConfig,
)

__all__ = [
    "AnnotationFeatureCenterSelectorConfig",
    "AnnotationPairMidpointSelectorConfig",
    "CoordinatePlacementLocatorConfig",
    "FallbackPolicyConfig",
    "FeatureMatchConfig",
    "FeatureRetentionPolicyConfig",
    "FlankPlacementLocatorConfig",
    "InnerJobConfig",
    "InputConfig",
    "JobConfig",
    "NormalizeAnchorConfig",
    "NormalizeAnchorSelectorConfig",
    "NormalizeTemplateConfig",
    "OutputConfig",
    "OutputSequenceViewConfig",
    "OutputVariantConfig",
    "OverLengthTrimPolicyConfig",
    "PartConfig",
    "PartSequenceConfig",
    "PlacementConfig",
    "PlacementGuardsConfig",
    "PlacementLocatorConfig",
    "RealizeConfig",
    "SelectorChainConfig",
    "SequenceMidpointSelectorConfig",
    "SequenceOffsetSelectorConfig",
    "StrictConfigModel",
    "TemplateConfig",
    "TemplateLiteralSourceConfig",
    "TemplatePathSourceConfig",
    "TemplateSourceConfig",
    "TemplateUSRSourceConfig",
    "USRDatasetLocatorConfig",
    "UnderLengthExpandFromTemplatePolicyConfig",
    "WindowConfig",
    "load_job_config",
]
