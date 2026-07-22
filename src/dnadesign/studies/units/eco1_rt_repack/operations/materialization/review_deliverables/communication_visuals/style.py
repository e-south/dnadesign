"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/style.py

Shared visual grammar for Eco1 communication figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.policy_visuals import (
    POLICY_COLORS,
    POLICY_LABELS,
    POLICY_ORDER,
    policy_color,
    policy_label,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import OKABE_ITO
from dnadesign.thread.structure_views import styles as molecular_styles

PROTEIN_SURFACE_COLOR = molecular_styles.PROTEIN_SURFACE_COLOR
PROTEIN_SURFACE_OPACITY = molecular_styles.PROTEIN_SURFACE_OPACITY
DNA_COLOR = molecular_styles.DNA_COLOR
RNA_COLOR = molecular_styles.RNA_COLOR
PROTECTED_COLOR = "#B63A3A"
MOTIF_COLOR = "#8A4A11"
CONTACT_COLOR = "#D99400"
CONSERVATION_COLOR = OKABE_ITO["blue"]
THUMB_COLOR = "#76549A"
RECOGNITION_COLOR = "#666666"
TEXT_COLOR = "#24292F"
GRID_COLOR = "#D8DEE4"

__all__ = [
    "CONSERVATION_COLOR",
    "CONTACT_COLOR",
    "DNA_COLOR",
    "GRID_COLOR",
    "MOTIF_COLOR",
    "POLICY_COLORS",
    "POLICY_LABELS",
    "POLICY_ORDER",
    "PROTECTED_COLOR",
    "PROTEIN_SURFACE_COLOR",
    "PROTEIN_SURFACE_OPACITY",
    "RECOGNITION_COLOR",
    "RNA_COLOR",
    "TEXT_COLOR",
    "THUMB_COLOR",
    "policy_color",
    "policy_label",
]
