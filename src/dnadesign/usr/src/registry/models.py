"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/registry/models.py

USR registry dataclasses and reserved namespace declarations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

REGISTRY_FILENAME = "registry.yaml"
USR_STATE_NAMESPACE = "usr_state"
USR_LABEL_NAMESPACE = "usr_label"
SEQ_ANNOT_NAMESPACE = "seq_annot"
DERIVED_NAMESPACE = "derived"


@dataclass(frozen=True)
class RegistryColumn:
    name: str
    type: str


@dataclass(frozen=True)
class RegistryEntry:
    namespace: str
    owner: str | None
    description: str | None
    columns: list[RegistryColumn]


USR_STATE_COLUMNS: list[RegistryColumn] = [
    RegistryColumn("usr_state__masked", "bool"),
    RegistryColumn("usr_state__qc_status", "string"),
    RegistryColumn("usr_state__split", "string"),
    RegistryColumn("usr_state__supersedes", "string"),
    RegistryColumn("usr_state__lineage", "list<string>"),
]

USR_LABEL_COLUMNS: list[RegistryColumn] = [
    RegistryColumn("usr_label__primary", "string"),
    RegistryColumn("usr_label__aliases", "list<string>"),
]

SEQ_ANNOT_COLUMNS: list[RegistryColumn] = [
    RegistryColumn("seq_annot__format", "string"),
    RegistryColumn("seq_annot__source_file", "string"),
    RegistryColumn("seq_annot__source_sha256", "string"),
    RegistryColumn("seq_annot__source_artifact_uri", "string"),
    RegistryColumn("seq_annot__parser", "string"),
    RegistryColumn("seq_annot__parser_version", "string"),
    RegistryColumn("seq_annot__record_id", "string"),
    RegistryColumn("seq_annot__record_name", "string"),
    RegistryColumn("seq_annot__description", "string"),
    RegistryColumn("seq_annot__topology", "string"),
    RegistryColumn("seq_annot__molecule_type", "string"),
    RegistryColumn("seq_annot__sequence_region_start_0", "int64"),
    RegistryColumn("seq_annot__sequence_region_end_0", "int64"),
    RegistryColumn(
        "seq_annot__features",
        (
            "list<struct<feature_id:string,feature_order:int64,feature_type:string,label:string,"
            "role_hint:string,location_raw:string,location_kind:string,start_0:int64,end_0:int64,"
            "strand:int64,intervals_0:list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>,"
            "is_fuzzy:bool,is_compound:bool,qualifiers:list<struct<key:string,value:string>>,"
            "confidence:string,source:string>>"
        ),
    ),
]

DERIVED_COLUMNS: list[RegistryColumn] = [
    RegistryColumn("derived__parent_id", "string"),
    RegistryColumn("derived__parent_dataset", "string"),
    RegistryColumn("derived__operation", "string"),
    RegistryColumn("derived__product_kind", "string"),
    RegistryColumn("derived__target_length", "int64"),
    RegistryColumn("derived__source_interval_start_0", "int64"),
    RegistryColumn("derived__source_interval_end_0", "int64"),
    RegistryColumn(
        "derived__source_intervals_0",
        "list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>",
    ),
    RegistryColumn("derived__orientation", "string"),
    RegistryColumn("derived__template_id", "string"),
    RegistryColumn("derived__template_dataset", "string"),
    RegistryColumn("derived__focal_rule", "string"),
    RegistryColumn("derived__focal_features", "list<string>"),
    RegistryColumn("derived__focal_confidence", "string"),
    RegistryColumn("derived__analysis_only", "bool"),
    RegistryColumn("derived__added_left_bp", "int64"),
    RegistryColumn("derived__added_right_bp", "int64"),
    RegistryColumn("derived__added_sequence_source", "string"),
    RegistryColumn(
        "derived__features_retained",
        (
            "list<struct<feature_id:string,label:string,role_hint:string,feature_type:string,status:string,"
            "original_intervals_0:list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>,"
            "derived_intervals_0:list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>,"
            "clipped_bp:int64,reason:string>>"
        ),
    ),
    RegistryColumn(
        "derived__features_clipped",
        (
            "list<struct<feature_id:string,label:string,role_hint:string,feature_type:string,status:string,"
            "original_intervals_0:list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>,"
            "derived_intervals_0:list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>,"
            "clipped_bp:int64,reason:string>>"
        ),
    ),
    RegistryColumn(
        "derived__features_lost",
        (
            "list<struct<feature_id:string,label:string,role_hint:string,feature_type:string,status:string,"
            "original_intervals_0:list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>,"
            "derived_intervals_0:list<struct<start_0:int64,end_0:int64,strand:int64,partial:bool>>,"
            "clipped_bp:int64,reason:string>>"
        ),
    ),
    RegistryColumn("derived__created_by", "string"),
    RegistryColumn("derived__spec_id", "string"),
]


def _clone_registry_entries(entries: dict[str, RegistryEntry]) -> dict[str, RegistryEntry]:
    return {
        name: RegistryEntry(
            namespace=entry.namespace,
            owner=entry.owner,
            description=entry.description,
            columns=list(entry.columns),
        )
        for name, entry in entries.items()
    }


def usr_state_entry() -> RegistryEntry:
    return RegistryEntry(
        namespace=USR_STATE_NAMESPACE,
        owner="usr",
        description="Reserved record-state overlay (masked/qc/split/lineage).",
        columns=list(USR_STATE_COLUMNS),
    )


def usr_label_entry() -> RegistryEntry:
    return RegistryEntry(
        namespace=USR_LABEL_NAMESPACE,
        owner="usr",
        description="Human-readable labels and aliases for canonical sequence records.",
        columns=list(USR_LABEL_COLUMNS),
    )


def seq_annot_entry() -> RegistryEntry:
    return RegistryEntry(
        namespace=SEQ_ANNOT_NAMESPACE,
        owner="usr",
        description="Imported source annotation overlays with preserved GenBank location fidelity.",
        columns=list(SEQ_ANNOT_COLUMNS),
    )


def derived_entry() -> RegistryEntry:
    return RegistryEntry(
        namespace=DERIVED_NAMESPACE,
        owner="usr",
        description="Derived-product lineage, focal selection, and feature-retention overlays.",
        columns=list(DERIVED_COLUMNS),
    )
