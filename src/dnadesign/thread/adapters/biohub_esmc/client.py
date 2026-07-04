"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/client.py

Small authenticated Biohub ESMC API client.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dnadesign.thread.adapters.biohub_esmc.auth import BiohubCredential

DEFAULT_BASE_URL = "https://biohub.ai"
DEFAULT_USER_AGENT = "dnadesign-thread-biohub-esmc/0.1"
BIOHUB_API_VERSION = "v1"
TRUSTED_BIOHUB_API_HOSTS = frozenset({"biohub.ai", "www.biohub.ai"})
ENCODE_PATH = "/api/v1/encode"
LOGITS_PATH = "/api/v1/logits"
DEFAULT_ESMC_MODEL = "esmc-6b-2024-12"
DEFAULT_ESMC_SAE_MODEL = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"
CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
_READ_CHUNK_BYTES = 65536
_MAX_SOCKET_READ_TIMEOUT_SECONDS = 30.0


class BiohubEsmcRequestError(RuntimeError):
    """Raised when Biohub ESMC API requests fail or return unusable JSON."""


@dataclass(frozen=True)
class BiohubEsmcClient:
    """Authenticated Biohub ESMC client using the documented encode -> logits path."""

    credential: BiohubCredential
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 60.0
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", validate_biohub_api_base_url(self.base_url))

    def encode_sequence(
        self,
        sequence: str,
        *,
        model: str = DEFAULT_ESMC_MODEL,
        potential_sequence_of_concern: bool = False,
    ) -> dict[str, Any]:
        """Encode one amino-acid sequence into Biohub model tokens."""

        payload: dict[str, Any] = {
            "model": model,
            "inputs": {"sequence": _normalize_sequence(sequence)},
            "potential_sequence_of_concern": potential_sequence_of_concern,
        }
        return self._post_json(ENCODE_PATH, payload)

    def logits_for_tokens(
        self,
        sequence_tokens: list[int],
        *,
        model: str = DEFAULT_ESMC_MODEL,
        sae_model: str = DEFAULT_ESMC_SAE_MODEL,
        normalize_features: bool = False,
        potential_sequence_of_concern: bool = False,
    ) -> dict[str, Any]:
        """Request ESMC SAE outputs for encoded sequence tokens."""

        if not sequence_tokens:
            raise ValueError("sequence_tokens must be non-empty")
        payload: dict[str, Any] = {
            "model": model,
            "inputs": {"sequence": [int(token) for token in sequence_tokens]},
            "logits_config": {
                "sequence": False,
                "return_embeddings": False,
                "return_mean_embedding": False,
                "return_hidden_states": False,
                "return_mean_hidden_states": False,
                "ith_hidden_layer": -1,
                "sae_config": {
                    "models": [sae_model],
                    "normalize_features": bool(normalize_features),
                },
            },
            "potential_sequence_of_concern": potential_sequence_of_concern,
        }
        return self._post_json(LOGITS_PATH, payload)

    def logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str = DEFAULT_ESMC_MODEL,
        sae_model: str = DEFAULT_ESMC_SAE_MODEL,
        normalize_features: bool = False,
        potential_sequence_of_concern: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        """Encode one sequence and request SAE outputs from the encoded tokens."""

        encode_response = self.encode_sequence(
            sequence,
            model=model,
            potential_sequence_of_concern=potential_sequence_of_concern,
        )
        tokens = extract_sequence_tokens(encode_response)
        logits_response = self.logits_for_tokens(
            tokens,
            model=model,
            sae_model=sae_model,
            normalize_features=normalize_features,
            potential_sequence_of_concern=potential_sequence_of_concern,
        )
        return encode_response, logits_response, tokens

    def sequence_logits_for_tokens(
        self,
        sequence_tokens: list[int],
        *,
        model: str = DEFAULT_ESMC_MODEL,
        potential_sequence_of_concern: bool = False,
    ) -> dict[str, Any]:
        """Request amino-acid sequence logits for encoded sequence tokens."""

        if not sequence_tokens:
            raise ValueError("sequence_tokens must be non-empty")
        payload: dict[str, Any] = {
            "model": model,
            "inputs": {"sequence": [int(token) for token in sequence_tokens]},
            "logits_config": {
                "sequence": True,
                "return_embeddings": False,
                "return_mean_embedding": False,
                "return_hidden_states": False,
                "return_mean_hidden_states": False,
                "ith_hidden_layer": -1,
            },
            "potential_sequence_of_concern": potential_sequence_of_concern,
        }
        return self._post_json(LOGITS_PATH, payload)

    def sequence_logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str = DEFAULT_ESMC_MODEL,
        potential_sequence_of_concern: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        """Encode one sequence and request sequence logits from the encoded tokens."""

        encode_response = self.encode_sequence(
            sequence,
            model=model,
            potential_sequence_of_concern=potential_sequence_of_concern,
        )
        tokens = extract_sequence_tokens(encode_response)
        logits_response = self.sequence_logits_for_tokens(
            tokens,
            model=model,
            potential_sequence_of_concern=potential_sequence_of_concern,
        )
        return encode_response, logits_response, tokens

    def amino_acid_token_indices(self, *, model: str = DEFAULT_ESMC_MODEL) -> dict[str, int]:
        """Resolve canonical amino-acid token ids through the Biohub encode API."""

        token_indices: dict[str, int] = {}
        for aa in CANONICAL_AMINO_ACIDS:
            tokens = extract_sequence_tokens(self.encode_sequence(aa, model=model))
            token_indices[aa] = _single_residue_token(tokens, residue=aa)
        return token_indices

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.credential.token}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, _MAX_SOCKET_READ_TIMEOUT_SECONDS)) as response:
                response_body = _read_response_body(
                    response,
                    total_timeout_seconds=self.timeout_seconds,
                    path=path,
                )
        except HTTPError as error:
            raise BiohubEsmcRequestError(f"Biohub ESMC request failed with HTTP {error.code}: {path}") from error
        except TimeoutError as error:
            message = f"Biohub ESMC request timed out after {self.timeout_seconds:g}s: {path}"
            raise BiohubEsmcRequestError(message) from error
        except URLError as error:
            raise BiohubEsmcRequestError(f"Biohub ESMC request failed: {error.reason}") from error
        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise BiohubEsmcRequestError(f"Biohub ESMC response was not JSON: {path}") from error
        if not isinstance(decoded, dict):
            raise BiohubEsmcRequestError(f"Biohub ESMC response must be a JSON object: {path}")
        return decoded


def extract_sequence_tokens(response: dict[str, Any]) -> list[int]:
    """Extract sequence tokens from the documented encode response."""

    outputs = response.get("outputs")
    if isinstance(outputs, dict):
        raw_tokens = outputs.get("sequence")
    else:
        raw_tokens = response.get("sequence")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise BiohubEsmcRequestError("Biohub encode response did not include outputs.sequence tokens")
    tokens: list[int] = []
    for token in raw_tokens:
        if not isinstance(token, int):
            raise BiohubEsmcRequestError("Biohub encode response sequence tokens must be integers")
        tokens.append(int(token))
    return tokens


def validate_biohub_api_base_url(base_url: str) -> str:
    """Return a normalized Biohub API base URL after enforcing the public endpoint."""

    parsed = urlsplit(str(base_url))
    host = parsed.hostname.lower() if parsed.hostname else ""
    if (
        parsed.scheme != "https"
        or host not in TRUSTED_BIOHUB_API_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path not in {"", "/"}
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        message = "Biohub API base URL must be https://biohub.ai or https://www.biohub.ai"
        raise ValueError(message)
    return f"https://{host}"


def _single_residue_token(tokens: list[int], *, residue: str) -> int:
    if len(tokens) >= 3:
        return int(tokens[1])
    if len(tokens) == 1:
        return int(tokens[0])
    raise BiohubEsmcRequestError(f"Biohub encode response for residue {residue!r} did not expose one residue token")


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("sequence must be non-empty")
    return normalized


def _read_response_body(response: Any, *, total_timeout_seconds: float, path: str) -> str:
    """Read chunked Biohub responses with a wall-clock deadline."""

    deadline = monotonic() + total_timeout_seconds
    chunks: list[bytes] = []
    while True:
        if monotonic() > deadline:
            message = f"Biohub ESMC response read timed out after {total_timeout_seconds:g}s: {path}"
            raise BiohubEsmcRequestError(message)
        chunk = response.read(_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks).decode("utf-8")
        chunks.append(chunk)
