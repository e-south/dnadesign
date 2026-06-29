"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/biohub_esmc_sae_profile/fixtures.py

Fixtures for Eco1 Biohub ESMC SAE-profile materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import torch

from dnadesign.thread.adapters.biohub_esmc import (
    DEFAULT_ESMC_SAE_MODEL,
    FEATURE_DESCRIPTION_SAE_MODEL,
    BiohubCredential,
)


class FakeBiohubEsmcClient:
    def __init__(self) -> None:
        self.credential = BiohubCredential(key_label="bu-dunlop-lab", token="fixture-secret")
        self.requested_sequences: list[str] = []

    def logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str,
        sae_model: str,
        normalize_features: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        del model, normalize_features
        normalized = sequence.strip().upper()
        self.requested_sequences.append(normalized[:4])
        tokens = [0, *range(1, len(normalized) + 1), 2]
        if sae_model == FEATURE_DESCRIPTION_SAE_MODEL:
            return _exact_dictionary_response(sae_model=sae_model, normalized=normalized, tokens=tokens)
        tensor = torch.zeros((len(normalized) + 2, 16), dtype=torch.float32)
        tensor[1, 3] = 1.5
        tensor[2, 7] = 2.0
        tensor[len(normalized), 7] = 4.0
        return (
            {"outputs": {"sequence": tokens}, "potential_sequence_of_concern": False},
            {"sae_outputs": {sae_model or DEFAULT_ESMC_SAE_MODEL: tensor}, "logits": None, "embeddings": None},
            tokens,
        )


class TimeoutOnceBiohubEsmcClient(FakeBiohubEsmcClient):
    def logits_for_sequence(
        self,
        sequence: str,
        *,
        model: str,
        sae_model: str,
        normalize_features: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        if len(self.requested_sequences) == 1:
            self.requested_sequences.append(sequence.strip().upper()[:4])
            raise TimeoutError("The read operation timed out")
        return super().logits_for_sequence(
            sequence,
            model=model,
            sae_model=sae_model,
            normalize_features=normalize_features,
        )


class FakeFeatureDescriptionClient:
    def __init__(self) -> None:
        self.requested: list[int] = []

    def fetch(self, *, sae_model: str, feature_index: int) -> object:
        self.requested.append(int(feature_index))
        return _FeatureDescription(
            sae_model=sae_model,
            feature_index=int(feature_index),
            label=f"fixture_feature_{feature_index}",
            description=f"Fixture exact-dictionary description for F{feature_index}.",
            raw_feature_hash="sha256:" + str(feature_index % 10) * 64,
        )


class _FeatureDescription:
    def __init__(
        self,
        *,
        sae_model: str,
        feature_index: int,
        label: str,
        description: str,
        raw_feature_hash: str,
    ) -> None:
        self.sae_model = sae_model
        self.feature_index = feature_index
        self.label = label
        self.description = description
        self.raw_feature_hash = raw_feature_hash


def _exact_dictionary_response(
    *,
    sae_model: str,
    normalized: str,
    tokens: list[int],
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    tensor = torch.zeros((len(normalized) + 2, 16384), dtype=torch.float32)
    for token_index in range(1, len(normalized) + 1):
        for feature_index in range(64):
            tensor[token_index, feature_index] = float(feature_index + 1)
    return (
        {"outputs": {"sequence": tokens}, "potential_sequence_of_concern": False},
        {"sae_outputs": {sae_model: tensor}, "logits": None, "embeddings": None},
        tokens,
    )
