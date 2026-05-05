"""Shared promoter metadata contracts for LatentDNA workspaces."""

from __future__ import annotations

REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS = (
    "regulondb__sigma_factor_set",
    "regulondb__sigma_factor_count",
    "regulondb__regulator_composition",
    "regulondb__box_pattern",
    "regulondb__confidence_level_set",
    "regulondb__metadata_completeness_class",
    "regulondb__source_strata_set",
    "regulondb__primary_promoter_name",
)

SYNTHETIC_PROMOTER_METADATA_INPUT_COLUMNS = (
    "densegen__plan",
    "densegen__required_regulators",
    "densegen__used_tfbs_detail",
    "source_family",
    "selection_basis",
    "view_collections",
    "role_tags",
    "usr_label__primary",
    "template_id",
    "construct__template_id",
    "seq_annot__sequence_region_start_0",
    "seq_annot__sequence_region_end_0",
    "seq_annot__features",
    "derived__target_length",
    "derived__features_retained",
    "promoter_standard__collection_id",
    "promoter_standard__promoter_id",
    "promoter_standard__display_name",
    "promoter_standard__strength_metric",
    "promoter_standard__strength_value",
    "promoter_standard__strength_value_numeric",
    "promoter_standard__strength_reference",
)

PROMOTER_METADATA_INPUT_COLUMNS = (
    *SYNTHETIC_PROMOTER_METADATA_INPUT_COLUMNS,
    *REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS,
)

SYNTHETIC_PROMOTER_METADATA_DERIVATIONS = (
    "design_family",
    "design_regulator_composition",
    "sig35_variant",
    "spacer_length",
    "campaign_prior",
    "is_control",
    "source_class",
)

PROMOTER_METADATA_DERIVATIONS = (
    *SYNTHETIC_PROMOTER_METADATA_DERIVATIONS,
    *REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS,
)

PROMOTER_METADATA_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "design_family": {"densegen__plan", "usr_label__primary"},
    "design_regulator_composition": {
        "densegen__plan",
        "densegen__required_regulators",
        "usr_label__primary",
    },
    "sig35_variant": {"usr_label__primary"},
    "spacer_length": {"densegen__used_tfbs_detail", "usr_label__primary"},
    "campaign_prior": {"densegen__plan", "usr_label__primary"},
    "is_control": {"densegen__plan", "usr_label__primary"},
    "source_class": {"densegen__plan", "usr_label__primary"},
    **{column: {column} for column in REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS},
}

PROMOTER_METADATA_ANY_COLUMN_GROUPS: dict[str, tuple[set[str], ...]] = {
    "sig35_variant": (
        {"densegen__plan"},
        {"densegen__used_tfbs_detail"},
        {"seq_annot__features"},
        {"sequence", "derived__features_retained"},
    ),
}

__all__ = [
    "PROMOTER_METADATA_ANY_COLUMN_GROUPS",
    "PROMOTER_METADATA_DERIVATIONS",
    "PROMOTER_METADATA_INPUT_COLUMNS",
    "PROMOTER_METADATA_REQUIRED_COLUMNS",
    "REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS",
    "SYNTHETIC_PROMOTER_METADATA_DERIVATIONS",
    "SYNTHETIC_PROMOTER_METADATA_INPUT_COLUMNS",
]
