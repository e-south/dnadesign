"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/feature_descriptions.py

Biohub ESMC SAE feature-description helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dnadesign.thread.adapters.biohub_esmc.hashes import raw_response_hash

FEATURE_DESCRIPTION_SAE_MODEL = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"
FEATURE_DESCRIPTION_CODEBOOK_SIZE = 16384
FEATURE_DESCRIPTION_API_PATH = "/esm/protein/api/v1alpha1/features/{feature_index}"
DEFAULT_FEATURE_DESCRIPTION_USER_AGENT = "dnadesign-thread-biohub-esmc-feature-descriptions/0.1"


class BiohubSaeFeatureDescriptionError(RuntimeError):
    """Raised when a Biohub SAE feature-description request fails."""


@dataclass(frozen=True)
class BiohubSaeFeatureDescription:
    """One source-backed SAE feature description row."""

    sae_model: str
    feature_index: int
    label: str
    description: str
    raw_feature_hash: str


def supports_feature_description_endpoint(sae_model: str) -> bool:
    """Return whether the public feature-description endpoint matches this SAE dictionary.

    Biohub's public feature-description endpoint is model-dictionary specific.
    The currently documented source-backed descriptions are for the 6B layer-60
    16,384-feature dictionary used by the ESM Atlas workflow, not every SAE
    dictionary exposed by `/api/v1/logits`.
    """

    return str(sae_model) == FEATURE_DESCRIPTION_SAE_MODEL


def parse_feature_description_response(
    response: dict[str, Any],
    *,
    sae_model: str,
    feature_index: int,
) -> BiohubSaeFeatureDescription:
    """Parse the public Biohub feature-description response into the local catalog shape."""

    observed_index = int(response.get("feature_index", feature_index))
    if observed_index != int(feature_index):
        raise BiohubSaeFeatureDescriptionError(
            f"Feature-description response index {observed_index} did not match requested {feature_index}"
        )
    return BiohubSaeFeatureDescription(
        sae_model=str(sae_model),
        feature_index=int(feature_index),
        label=str(response.get("label") or ""),
        description=str(response.get("description") or response.get("summary") or ""),
        raw_feature_hash=raw_response_hash(response),
    )


@dataclass(frozen=True)
class BiohubSaeFeatureDescriptionClient:
    """Small no-auth client for the public Biohub SAE feature-description endpoint."""

    base_url: str = "https://biohub.ai"
    timeout_seconds: float = 30.0
    user_agent: str = DEFAULT_FEATURE_DESCRIPTION_USER_AGENT

    def fetch(self, *, sae_model: str, feature_index: int) -> BiohubSaeFeatureDescription:
        """Fetch one source-backed feature description when the SAE dictionary is compatible."""

        if not supports_feature_description_endpoint(sae_model):
            raise BiohubSaeFeatureDescriptionError(
                "Biohub feature descriptions are currently source-backed only for "
                f"{FEATURE_DESCRIPTION_SAE_MODEL} "
                f"(codebook{FEATURE_DESCRIPTION_CODEBOOK_SIZE}); got {sae_model!r}"
            )
        if int(feature_index) >= FEATURE_DESCRIPTION_CODEBOOK_SIZE:
            raise BiohubSaeFeatureDescriptionError(
                "Biohub feature descriptions are currently exposed for feature indices "
                f"0-{FEATURE_DESCRIPTION_CODEBOOK_SIZE - 1}; got F{feature_index}"
            )
        url = self.base_url.rstrip("/") + FEATURE_DESCRIPTION_API_PATH.format(feature_index=int(feature_index))
        request = Request(url, method="GET", headers={"Accept": "application/json", "User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            raise BiohubSaeFeatureDescriptionError(
                f"Biohub feature-description request failed with HTTP {error.code}: F{feature_index}"
            ) from error
        except (TimeoutError, URLError) as error:
            raise BiohubSaeFeatureDescriptionError(
                f"Biohub feature-description request failed for F{feature_index}: {error}"
            ) from error
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise BiohubSaeFeatureDescriptionError(
                f"Biohub feature-description response was not JSON: F{feature_index}"
            ) from error
        if not isinstance(decoded, dict):
            raise BiohubSaeFeatureDescriptionError(
                f"Biohub feature-description response must be a JSON object: F{feature_index}"
            )
        return parse_feature_description_response(decoded, sae_model=sae_model, feature_index=feature_index)
