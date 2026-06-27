"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/esm_atlas/client.py

Small, explicit ESM Atlas API client.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dnadesign.thread.adapters.esm_atlas.hashes import sequence_md5

DEFAULT_BASE_URL = "https://biohub.ai"
API_PREFIX = "/esm/protein/api/v1alpha1"
DEFAULT_USER_AGENT = "dnadesign-thread-esm-atlas/0.1"


class AtlasRequestError(RuntimeError):
    """Raised when the Atlas API does not return a usable JSON payload."""


@dataclass(frozen=True)
class AtlasClient:
    """No-auth Atlas API client with bounded request parameters."""

    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT

    def protein_lookup_by_sequence(
        self,
        sequence: str,
        *,
        topk_features: int = 100,
        fold_on_miss: bool = False,
        normalize_features: bool = True,
        feature_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Look up one protein by MD5 hash of its amino-acid sequence."""

        protein_hash = sequence_md5(sequence)
        return self.protein_lookup_by_hash(
            protein_hash,
            topk_features=topk_features,
            fold_on_miss=fold_on_miss,
            normalize_features=normalize_features,
            feature_indices=feature_indices,
        )

    def protein_lookup_by_hash(
        self,
        protein_hash: str,
        *,
        topk_features: int = 100,
        fold_on_miss: bool = False,
        normalize_features: bool = True,
        feature_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Look up protein metadata, features, and sparse activations by hash."""

        _require_md5(protein_hash, "protein_hash")
        _require_range(topk_features, "topk_features", minimum=1, maximum=100)
        params: list[tuple[str, str | int]] = [
            ("topk_features", topk_features),
            ("fold_on_miss", str(fold_on_miss).lower()),
            ("normalize_features", str(normalize_features).lower()),
        ]
        if feature_indices is not None:
            if len(feature_indices) > 100:
                raise ValueError("feature_indices is capped at 100 entries by the Atlas API")
            for feature_index in feature_indices:
                _require_range(feature_index, "feature_index", minimum=0, maximum=16383)
                params.append(("feature_indices", feature_index))
        return self._get_json(f"{API_PREFIX}/proteins/{protein_hash}", params)

    def similarity_search(
        self,
        sequence: str,
        *,
        topk_results: int = 5,
        topk_features: int = 10,
        include_cluster_info: bool = True,
    ) -> dict[str, Any]:
        """Search Atlas by sequence-level SAE feature similarity."""

        normalized = _normalize_sequence(sequence)
        if len(normalized) > 800:
            raise ValueError("Atlas similarity-search sequence length is capped at 800 residues")
        _require_range(topk_results, "topk_results", minimum=1, maximum=100)
        _require_range(topk_features, "topk_features", minimum=1, maximum=100)
        return self._get_json(
            f"{API_PREFIX}/similarity-search",
            [
                ("sequence", normalized),
                ("topk_results", topk_results),
                ("topk_features", topk_features),
                ("include_cluster_info", str(include_cluster_info).lower()),
            ],
        )

    def _get_json(self, path: str, params: list[tuple[str, str | int]]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path + "?" + urlencode(params, doseq=True)
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise AtlasRequestError(f"Atlas API request failed with HTTP {error.code}: {path}") from error
        except URLError as error:
            raise AtlasRequestError(f"Atlas API request failed: {error.reason}") from error
        except json.JSONDecodeError as error:
            raise AtlasRequestError(f"Atlas API response was not JSON: {path}") from error
        if not isinstance(payload, dict):
            raise AtlasRequestError(f"Atlas API response must be a JSON object: {path}")
        return payload


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("sequence must be non-empty")
    return normalized


def _require_range(value: int, field: str, *, minimum: int, maximum: int) -> None:
    if not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(f"{field} must be an integer from {minimum} to {maximum}")


def _require_md5(value: str, field: str) -> None:
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{field} must be a 32-character lowercase MD5 hex digest")
