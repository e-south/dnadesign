"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/biohub_esmc/test_auth_and_client.py

Biohub ESMC auth and client tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dnadesign.thread.adapters.biohub_esmc import (
    DEFAULT_ESMC_MODEL,
    DEFAULT_ESMC_SAE_MODEL,
    BiohubCredential,
    BiohubEsmcClient,
    BiohubEsmcRequestError,
    load_biohub_credential,
    validate_biohub_api_base_url,
)
from dnadesign.thread.adapters.biohub_esmc import client as biohub_client_module
from dnadesign.thread.adapters.biohub_esmc.client import _read_response_body, extract_sequence_tokens


def test_load_biohub_credential_keeps_token_out_of_repr(tmp_path: Path) -> None:
    key_path = tmp_path / "key.md"
    key_path.write_text("bu-dunlop-lab\nsecret-token-value\n", encoding="utf-8")

    credential = load_biohub_credential(key_path)

    assert credential.key_label == "bu-dunlop-lab"
    assert credential.token == "secret-token-value"
    assert "secret-token-value" not in repr(credential)
    assert credential.redacted_token == "<redacted>"


def test_load_biohub_credential_rejects_unexpected_label(tmp_path: Path) -> None:
    key_path = tmp_path / "key.md"
    key_path.write_text("wrong-label\nsecret-token-value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="key label"):
        load_biohub_credential(key_path)


def test_extract_sequence_tokens_accepts_documented_encode_shape() -> None:
    assert extract_sequence_tokens({"outputs": {"sequence": [0, 4, 5, 2]}}) == [0, 4, 5, 2]


def test_validate_biohub_api_base_url_accepts_public_hosts() -> None:
    assert validate_biohub_api_base_url("https://biohub.ai/") == "https://biohub.ai"
    assert validate_biohub_api_base_url("https://www.biohub.ai") == "https://www.biohub.ai"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://biohub.ai",
        "https://biohub.ai.example.org",
        "https://biohub.ai:8443",
        "https://biohub.ai/api",
        "https://token@biohub.ai",
        "https://biohub.ai?redirect=https://example.org",
    ],
)
def test_validate_biohub_api_base_url_rejects_untrusted_or_ambiguous_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="Biohub API base URL"):
        validate_biohub_api_base_url(base_url)


def test_authenticated_client_rejects_untrusted_base_url_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Biohub client should reject the base URL before opening a request")

    monkeypatch.setattr(biohub_client_module, "urlopen", fail_urlopen)

    with pytest.raises(ValueError, match="Biohub API base URL"):
        BiohubEsmcClient(
            credential=BiohubCredential(key_label="bu-dunlop-lab", token="fixture-secret"),
            base_url="https://example.org",
        )


def test_client_uses_encode_then_logits_request_shape() -> None:
    client = _FakeBiohubClient()

    encode_response, logits_response, tokens = client.logits_for_sequence(
        "ACDE",
        model=DEFAULT_ESMC_MODEL,
        sae_model=DEFAULT_ESMC_SAE_MODEL,
        normalize_features=False,
    )

    assert tokens == [0, 4, 5, 6, 7, 2]
    assert encode_response["outputs"]["sequence"] == tokens
    assert logits_response["sae_outputs"] == {}
    assert [request["path"] for request in client.requests] == ["/api/v1/encode", "/api/v1/logits"]
    encode_payload = client.requests[0]["payload"]
    logits_payload = client.requests[1]["payload"]
    assert encode_payload["inputs"]["sequence"] == "ACDE"
    assert logits_payload["inputs"]["sequence"] == tokens
    assert logits_payload["logits_config"]["sae_config"] == {
        "models": [DEFAULT_ESMC_SAE_MODEL],
        "normalize_features": False,
    }


def test_client_requests_sequence_logits_without_sae_config() -> None:
    client = _FakeBiohubClient()

    _encode_response, logits_response, tokens = client.sequence_logits_for_sequence("A_DE", model=DEFAULT_ESMC_MODEL)

    assert tokens == [0, 4, 5, 6, 7, 2]
    assert logits_response["logits"]["sequence"] == [[[0.0, 1.0], [2.0, 3.0]]]
    assert [request["path"] for request in client.requests] == ["/api/v1/encode", "/api/v1/logits"]
    encode_payload = client.requests[0]["payload"]
    logits_payload = client.requests[1]["payload"]
    assert encode_payload["inputs"]["sequence"] == "A_DE"
    assert logits_payload["inputs"]["sequence"] == tokens
    assert logits_payload["logits_config"]["sequence"] is True
    assert "sae_config" not in logits_payload["logits_config"]


def test_client_resolves_amino_acid_token_indices_from_encode() -> None:
    client = _FakeBiohubClient()

    token_indices = client.amino_acid_token_indices(model=DEFAULT_ESMC_MODEL)

    assert token_indices["A"] == 4
    assert token_indices["C"] == 5
    assert len(token_indices) == 20
    assert all(request["path"] == "/api/v1/encode" for request in client.requests)


def test_read_response_body_uses_wall_clock_deadline() -> None:
    assert _read_response_body(_ChunkedResponse([b'{"ok":', b"true}"]), total_timeout_seconds=5, path="/api/test") == (
        '{"ok":true}'
    )
    with pytest.raises(BiohubEsmcRequestError, match="timed out"):
        _read_response_body(_ChunkedResponse([b"{}"]), total_timeout_seconds=-1, path="/api/test")


class _FakeBiohubClient(BiohubEsmcClient):
    _AA_TOKENS = {aa: index + 4 for index, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def __init__(self) -> None:
        super().__init__(credential=BiohubCredential(key_label="bu-dunlop-lab", token="fixture-secret"))
        object.__setattr__(self, "requests", [])

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"path": path, "payload": payload})
        if path.endswith("/encode"):
            sequence = str(payload["inputs"]["sequence"])
            if len(sequence) == 1 and sequence in self._AA_TOKENS:
                return {
                    "outputs": {"sequence": [0, self._AA_TOKENS[sequence], 2]},
                    "potential_sequence_of_concern": False,
                }
            return {"outputs": {"sequence": [0, 4, 5, 6, 7, 2]}, "potential_sequence_of_concern": False}
        if payload["logits_config"].get("sequence") is True:
            return {"logits": {"sequence": [[[0.0, 1.0], [2.0, 3.0]]]}, "embeddings": None, "hidden_states": None}
        return {"sae_outputs": {}, "logits": None, "embeddings": None, "hidden_states": None}


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = [*chunks]

    def read(self, _size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)
