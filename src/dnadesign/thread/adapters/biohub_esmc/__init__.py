"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/__init__.py

Biohub ESMC logits adapter and SAE activation normalizers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.thread.adapters.biohub_esmc.auth import BiohubCredential, load_biohub_credential
from dnadesign.thread.adapters.biohub_esmc.client import (
    DEFAULT_ESMC_MODEL,
    DEFAULT_ESMC_SAE_MODEL,
    BiohubEsmcClient,
    BiohubEsmcRequestError,
)
from dnadesign.thread.adapters.biohub_esmc.hashes import biohub_query_hash, biohub_request_hash, raw_response_hash
from dnadesign.thread.adapters.biohub_esmc.normalize import (
    build_error_profile_row,
    normalize_logits_response,
)
from dnadesign.thread.adapters.biohub_esmc.tables import (
    BiohubEsmcArtifacts,
    validate_biohub_esmc_artifacts,
    write_biohub_esmc_artifacts,
)

__all__ = [
    "BiohubCredential",
    "BiohubEsmcArtifacts",
    "BiohubEsmcClient",
    "BiohubEsmcRequestError",
    "DEFAULT_ESMC_MODEL",
    "DEFAULT_ESMC_SAE_MODEL",
    "biohub_query_hash",
    "biohub_request_hash",
    "build_error_profile_row",
    "load_biohub_credential",
    "normalize_logits_response",
    "raw_response_hash",
    "validate_biohub_esmc_artifacts",
    "write_biohub_esmc_artifacts",
]
