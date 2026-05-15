"""Public promoter metadata derivation dispatcher for LatentDNA workspaces."""

from __future__ import annotations

from ..contracts.errors import ContractViolationError
from ..contracts.promoter_metadata import REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS
from .promoter_metadata_sequence import sig35_variant, spacer_length
from .promoter_metadata_stress import (
    campaign_prior,
    design_family,
    design_regulator_composition,
    is_control_row,
    source_class,
)


def derive_promoter_metadata_value(row: dict[str, object], *, derive: str) -> object:
    if derive in REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS:
        if derive not in row:
            raise ContractViolationError(f"native RegulonDB promoter metadata column is missing: {derive}")
        return row[derive]
    if derive == "design_family":
        return design_family(row)
    if derive == "design_regulator_composition":
        return design_regulator_composition(row)
    if derive == "sig35_variant":
        return sig35_variant(row)
    if derive == "spacer_length":
        return spacer_length(row)
    if derive == "campaign_prior":
        return campaign_prior(row)
    if derive == "is_control":
        return is_control_row(row)
    if derive == "source_class":
        return source_class(row)
    raise ContractViolationError(f"unsupported promoter metadata derivation: {derive}")
