"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/__init__.py

Source-ingest helpers for Retron hairpin study records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .msd_region_genbank import (
    compare_records_to_existing_sources,
    compiler_spec_payload_from_records,
    parse_msd_region_genbank,
    write_msd_region_record_bundle,
)
from .selected_lineage import (
    MaterializedVariantLineageEntryV1,
    MaterializedVariantLineageError,
    MaterializedVariantLineageV1,
    MsdStructuralPrimitiveRefsV1,
    load_materialized_variant_lineage,
)

__all__ = [
    "MaterializedVariantLineageEntryV1",
    "MaterializedVariantLineageError",
    "MaterializedVariantLineageV1",
    "MsdStructuralPrimitiveRefsV1",
    "compiler_spec_payload_from_records",
    "compare_records_to_existing_sources",
    "load_materialized_variant_lineage",
    "parse_msd_region_genbank",
    "write_msd_region_record_bundle",
]
