"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/nickases/__init__.py

Shared nickase catalog and scanning contracts for Cruncher workflow families.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.cruncher.nickases.catalog import (
    dump_nickase_catalog_payload,
    dump_nickase_catalog_yaml,
    load_builtin_nickase_catalog_preset,
    load_merged_nickase_catalog,
    load_nickase_catalog,
    merge_nickase_catalogs,
    read_builtin_nickase_catalog_preset_text,
    resolve_builtin_catalog_resource,
    resolve_workspace_relative_path,
)
from dnadesign.cruncher.nickases.errors import NickaseCatalogError
from dnadesign.cruncher.nickases.models import (
    NickaseCatalog,
    NickaseCatalogDocument,
    NickaseCatalogEntry,
    NickaseProductAlias,
    NickEvent,
    RecognitionSiteInstance,
    iupac_bases_for_symbol,
    motif_matches,
    normalize_dna,
    normalize_iupac,
    reverse_complement,
    reverse_complement_iupac,
)
from dnadesign.cruncher.nickases.scanning import (
    EvaluatedMatch,
    derive_nick_event,
    display_motif_for_orientation,
    enumerate_site_instances,
)

__all__ = [
    "EvaluatedMatch",
    "NickEvent",
    "NickaseCatalog",
    "NickaseCatalogDocument",
    "NickaseCatalogEntry",
    "NickaseCatalogError",
    "NickaseProductAlias",
    "RecognitionSiteInstance",
    "derive_nick_event",
    "display_motif_for_orientation",
    "dump_nickase_catalog_payload",
    "dump_nickase_catalog_yaml",
    "enumerate_site_instances",
    "iupac_bases_for_symbol",
    "load_builtin_nickase_catalog_preset",
    "load_merged_nickase_catalog",
    "load_nickase_catalog",
    "merge_nickase_catalogs",
    "motif_matches",
    "normalize_dna",
    "normalize_iupac",
    "read_builtin_nickase_catalog_preset_text",
    "resolve_builtin_catalog_resource",
    "resolve_workspace_relative_path",
    "reverse_complement",
    "reverse_complement_iupac",
]
