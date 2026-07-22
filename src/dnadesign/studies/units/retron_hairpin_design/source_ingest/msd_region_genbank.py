"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/msd_region_genbank.py

Public boundary for MSD-region GenBank source ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .bundle_writer import write_msd_region_record_bundle
from .comparison import compare_records_to_existing_sources
from .compiler_spec_payload import compiler_spec_payload_from_records
from .genbank_bundle import (
    parse_msd_region_genbank,
    parse_msd_region_genbank_dir,
    parse_msd_region_genbank_with_replacements,
)
from .payload_binding import load_payload_binding_catalog
from .variant_sources import write_variant_genbank_sources

__all__ = [
    "compiler_spec_payload_from_records",
    "compare_records_to_existing_sources",
    "load_payload_binding_catalog",
    "parse_msd_region_genbank",
    "parse_msd_region_genbank_dir",
    "parse_msd_region_genbank_with_replacements",
    "write_msd_region_record_bundle",
    "write_variant_genbank_sources",
]
