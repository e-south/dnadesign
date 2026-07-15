"""Shared labels and colors for Eco1 generation-policy visuals."""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import OKABE_ITO

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


def policy_label(policy_id: str) -> str:
    """Return the plain display label for one generation policy."""

    return POLICY_LABELS.get(policy_id, policy_id.replace("_", " "))


def policy_color(policy_id: str) -> str:
    """Return the stable colorblind-safe color for one generation policy."""

    return POLICY_COLORS.get(policy_id, "#777777")


__all__ = ["POLICY_COLORS", "POLICY_LABELS", "POLICY_ORDER", "policy_color", "policy_label"]
