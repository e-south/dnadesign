"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/esm_atlas/__init__.py

ESM Atlas API adapter and sparse activation normalizers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.thread.adapters.esm_atlas.client import AtlasClient, AtlasRequestError
from dnadesign.thread.adapters.esm_atlas.hashes import (
    atlas_query_hash,
    atlas_request_hash,
    raw_response_hash,
    sequence_md5,
)
from dnadesign.thread.adapters.esm_atlas.normalize import (
    build_error_profile_row,
    normalize_protein_lookup_response,
)
from dnadesign.thread.adapters.esm_atlas.structure_predictions import build_atlas_structure_prediction_row
from dnadesign.thread.adapters.esm_atlas.tables import (
    AtlasSemanticArtifacts,
    validate_atlas_semantic_artifacts,
    write_atlas_semantic_artifacts,
)

__all__ = [
    "AtlasClient",
    "AtlasRequestError",
    "AtlasSemanticArtifacts",
    "atlas_query_hash",
    "atlas_request_hash",
    "build_atlas_structure_prediction_row",
    "build_error_profile_row",
    "normalize_protein_lookup_response",
    "raw_response_hash",
    "sequence_md5",
    "validate_atlas_semantic_artifacts",
    "write_atlas_semantic_artifacts",
]
