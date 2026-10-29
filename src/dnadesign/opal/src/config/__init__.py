"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/config/__init__.py

Package exports for OPAL config.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .loader import load_config  # noqa: F401
from .types import (  # noqa: F401
    CandidateScope,
    LabelsBlock,
    LabelSourceCampaignHistory,
    LabelSourceUSRSidecar,
    LocationLocal,
    LocationUSR,
    RootConfig,
    WritebackBlock,
)

# Re-export common config types for convenient imports in the CLI layer.
__all__ = [
    "load_config",
    "RootConfig",
    "CandidateScope",
    "LocationLocal",
    "LocationUSR",
    "LabelsBlock",
    "LabelSourceCampaignHistory",
    "LabelSourceUSRSidecar",
    "WritebackBlock",
]
