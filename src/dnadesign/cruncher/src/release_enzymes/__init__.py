"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/release_enzymes/__init__.py

Shared release-enzyme catalog and scanning contracts.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from dnadesign.cruncher.release_enzymes.catalog import (
    dump_release_enzyme_catalog_payload,
    dump_release_enzyme_catalog_yaml,
    load_builtin_release_enzyme_catalog_preset,
    load_merged_release_enzyme_catalog,
    load_release_enzyme_catalog,
    merge_release_enzyme_catalogs,
    read_builtin_release_enzyme_catalog_preset_text,
    resolve_builtin_catalog_resource,
)
from dnadesign.cruncher.release_enzymes.errors import ReleaseEnzymeCatalogError
from dnadesign.cruncher.release_enzymes.models import (
    ReleaseCutEvent,
    ReleaseEnzymeCatalog,
    ReleaseEnzymeCatalogDocument,
    ReleaseEnzymeEntry,
    ReleaseRecognitionSiteInstance,
)
from dnadesign.cruncher.release_enzymes.scanning import (
    ReleaseEvaluatedMatch,
    build_evaluated_match,
    derive_release_cut,
    display_motif_for_orientation,
    enumerate_site_instances,
    enumerate_top_cut_placements,
)
from dnadesign.cruncher.release_enzymes.selection import release_entry_priority_key

__all__ = [
    "ReleaseCutEvent",
    "ReleaseEnzymeCatalog",
    "ReleaseEnzymeCatalogDocument",
    "ReleaseEnzymeCatalogError",
    "ReleaseEnzymeEntry",
    "ReleaseEvaluatedMatch",
    "ReleaseRecognitionSiteInstance",
    "build_evaluated_match",
    "derive_release_cut",
    "display_motif_for_orientation",
    "dump_release_enzyme_catalog_payload",
    "dump_release_enzyme_catalog_yaml",
    "enumerate_site_instances",
    "enumerate_top_cut_placements",
    "load_builtin_release_enzyme_catalog_preset",
    "load_merged_release_enzyme_catalog",
    "load_release_enzyme_catalog",
    "merge_release_enzyme_catalogs",
    "read_builtin_release_enzyme_catalog_preset_text",
    "release_entry_priority_key",
    "resolve_builtin_catalog_resource",
]
