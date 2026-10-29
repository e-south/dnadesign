"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/biohub_esmc/test_feature_descriptions.py

Biohub ESMC SAE feature-description helper tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.thread.adapters.biohub_esmc import (
    DEFAULT_ESMC_SAE_MODEL,
    FEATURE_DESCRIPTION_CODEBOOK_SIZE,
    FEATURE_DESCRIPTION_SAE_MODEL,
    BiohubSaeFeatureDescriptionClient,
    BiohubSaeFeatureDescriptionError,
    parse_feature_description_response,
    supports_feature_description_endpoint,
)


def test_feature_description_endpoint_is_exact_dictionary_gated() -> None:
    assert supports_feature_description_endpoint(FEATURE_DESCRIPTION_SAE_MODEL)
    assert DEFAULT_ESMC_SAE_MODEL == FEATURE_DESCRIPTION_SAE_MODEL
    assert supports_feature_description_endpoint(DEFAULT_ESMC_SAE_MODEL)


def test_parse_feature_description_response_preserves_source_label_and_hash() -> None:
    parsed = parse_feature_description_response(
        {
            "feature_index": 14365,
            "label": "Polymerase thumb/palm nucleic acid binding",
            "summary": "Short summary.",
            "description": "Long source-backed description.",
        },
        sae_model=FEATURE_DESCRIPTION_SAE_MODEL,
        feature_index=14365,
    )

    assert parsed.feature_index == 14365
    assert parsed.label == "Polymerase thumb/palm nucleic acid binding"
    assert parsed.description == "Long source-backed description."
    assert parsed.raw_feature_hash.startswith("sha256:")


def test_feature_description_client_rejects_out_of_dictionary_index_before_request() -> None:
    client = BiohubSaeFeatureDescriptionClient()

    with pytest.raises(BiohubSaeFeatureDescriptionError, match="0-16383"):
        client.fetch(
            sae_model=FEATURE_DESCRIPTION_SAE_MODEL,
            feature_index=FEATURE_DESCRIPTION_CODEBOOK_SIZE,
        )


def test_feature_description_client_uses_public_biohub_base_url() -> None:
    assert BiohubSaeFeatureDescriptionClient(base_url="https://www.biohub.ai/").base_url == "https://www.biohub.ai"
    with pytest.raises(ValueError, match="Biohub API base URL"):
        BiohubSaeFeatureDescriptionClient(base_url="https://example.org")
