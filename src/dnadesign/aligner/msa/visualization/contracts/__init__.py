"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/contracts/__init__.py

Contracts for generic MSA visualization sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.aligner.msa.visualization.contracts.annotation_tracks import (
    load_annotation_tracks,
    validate_annotation_track_ranges,
)
from dnadesign.aligner.msa.visualization.contracts.exemplar_rows import (
    load_exemplar_rows,
    validate_exemplar_rows,
)
from dnadesign.aligner.msa.visualization.contracts.models import (
    AnnotationFeature,
    AnnotationTrack,
    ExemplarRow,
    ExemplarRowsSpec,
    FeatureWindow,
    MsaVisualizationRequest,
    MsaVisualizationResult,
    PositionQc,
    ProfileQc,
)
from dnadesign.aligner.msa.visualization.contracts.panel_spec import (
    MsaPanelSpec,
    load_panel_spec,
)

__all__ = [
    "AnnotationFeature",
    "AnnotationTrack",
    "ExemplarRow",
    "ExemplarRowsSpec",
    "FeatureWindow",
    "MsaPanelSpec",
    "MsaVisualizationRequest",
    "MsaVisualizationResult",
    "PositionQc",
    "ProfileQc",
    "load_annotation_tracks",
    "load_exemplar_rows",
    "load_panel_spec",
    "validate_annotation_track_ranges",
    "validate_exemplar_rows",
]
