"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/__init__.py

Generic MSA QC and visualization sidecar API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.aligner.msa.visualization.contracts import MsaPanelSpec
from dnadesign.aligner.msa.visualization.contracts.models import (
    MsaVisualizationRequest,
    MsaVisualizationResult,
)
from dnadesign.aligner.msa.visualization.materialization import (
    materialize_msa_visualizations,
)

__all__ = [
    "MsaPanelSpec",
    "MsaVisualizationRequest",
    "MsaVisualizationResult",
    "materialize_msa_visualizations",
]
