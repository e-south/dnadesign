"""Shared visual grammar for Eco1 communication figures."""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import OKABE_ITO
from dnadesign.thread.structure_views import styles as molecular_styles

POLICY_ORDER = (
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
)
POLICY_LABELS = {
    DISTAL_SCAFFOLD_POLICY_ID: "Distal",
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID: "Peripheral",
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID: "Combined",
}
POLICY_COLORS = {
    DISTAL_SCAFFOLD_POLICY_ID: OKABE_ITO["blue"],
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID: OKABE_ITO["green"],
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID: OKABE_ITO["orange"],
}

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


def policy_label(policy_id: str) -> str:
    """Return the plain group label for a declared generation policy."""

    return POLICY_LABELS.get(policy_id, policy_id.replace("_", " "))


def policy_color(policy_id: str) -> str:
    """Return a stable, colorblind-safe group color."""

    return POLICY_COLORS.get(policy_id, "#777777")
