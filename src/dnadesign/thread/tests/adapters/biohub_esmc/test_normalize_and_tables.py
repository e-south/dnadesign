"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/biohub_esmc/test_normalize_and_tables.py

Biohub ESMC normalization and artifact contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch

from dnadesign.thread.adapters.biohub_esmc import (
    DEFAULT_ESMC_MODEL,
    DEFAULT_ESMC_SAE_MODEL,
    biohub_query_hash,
    biohub_request_hash,
    build_error_profile_row,
    normalize_logits_response,
    validate_biohub_esmc_artifacts,
    write_biohub_esmc_artifacts,
)
from dnadesign.thread.foldcheck import sequence_hash

_SOURCE_REQUEST_HASH = "sha256:" + "1" * 64
_RETRIEVED_AT = "2026-06-25T00:00:00Z"


def test_normalize_logits_response_writes_sparse_residue_and_protein_rows() -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    normalized = normalize_logits_response(
        candidate_id="candidate_a",
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        encode_response={"outputs": {"sequence": [0, 1, 2, 3, 4, 5]}},
        logits_response=_logits_response(sequence),
        source_request_hash=_SOURCE_REQUEST_HASH,
        biohub_request_hash=request_hash,
        biohub_query_hash=_query_hash(sequence),
        biohub_api_base_url="https://biohub.ai",
        biohub_api_version="v1",
        model=DEFAULT_ESMC_MODEL,
        sae_model=DEFAULT_ESMC_SAE_MODEL,
        normalize_features=False,
        key_label="bu-dunlop-lab",
        retrieved_at=_RETRIEVED_AT,
    )

    assert normalized.profile_row["status"] == "accepted"
    assert normalized.profile_row["token_count"] == 6
    assert normalized.profile_row["residue_feature_count"] == 3
    assert normalized.residue_feature_rows == [
        {
            "candidate_id": "candidate_a",
            "sequence_hash": sequence_hash(sequence),
            "sae_model": DEFAULT_ESMC_SAE_MODEL,
            "residue_index_zero_based": 0,
            "sequence_position_one_based": 1,
            "feature_index": 3,
            "value": 1.5,
        },
        {
            "candidate_id": "candidate_a",
            "sequence_hash": sequence_hash(sequence),
            "sae_model": DEFAULT_ESMC_SAE_MODEL,
            "residue_index_zero_based": 1,
            "sequence_position_one_based": 2,
            "feature_index": 7,
            "value": 2.0,
        },
        {
            "candidate_id": "candidate_a",
            "sequence_hash": sequence_hash(sequence),
            "sae_model": DEFAULT_ESMC_SAE_MODEL,
            "residue_index_zero_based": 3,
            "sequence_position_one_based": 4,
            "feature_index": 7,
            "value": 4.0,
        },
    ]
    feature_7 = next(row for row in normalized.protein_feature_rows if row["feature_index"] == 7)
    assert feature_7["nonzero_residue_count"] == 2
    assert feature_7["activation_sum"] == 6.0
    assert feature_7["activation_mean"] == 1.5
    assert feature_7["activation_max"] == 4.0


def test_normalize_logits_response_handles_sparse_tensor_with_special_tokens() -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    indices = torch.tensor([[0, 1, 2, 5], [9, 3, 7, 11]])
    values = torch.tensor([99.0, 1.5, 2.0, 88.0])
    sparse = torch.sparse_coo_tensor(indices, values, size=(len(sequence) + 2, 16)).coalesce()

    normalized = normalize_logits_response(
        candidate_id="candidate_a",
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        encode_response={"outputs": {"sequence": [0, 1, 2, 3, 4, 5]}},
        logits_response={"sae_outputs": {DEFAULT_ESMC_SAE_MODEL: sparse}},
        source_request_hash=_SOURCE_REQUEST_HASH,
        biohub_request_hash=request_hash,
        biohub_query_hash=_query_hash(sequence),
        biohub_api_base_url="https://biohub.ai",
        biohub_api_version="v1",
        model=DEFAULT_ESMC_MODEL,
        sae_model=DEFAULT_ESMC_SAE_MODEL,
        normalize_features=False,
        key_label="bu-dunlop-lab",
        retrieved_at=_RETRIEVED_AT,
    )

    assert [
        (row["residue_index_zero_based"], row["feature_index"], row["value"]) for row in normalized.residue_feature_rows
    ] == [
        (0, 3, 1.5),
        (1, 7, 2.0),
    ]


def test_biohub_esmc_artifact_writer_validates_expected_candidates(tmp_path: Path) -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    normalized = normalize_logits_response(
        candidate_id="candidate_a",
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        encode_response={"outputs": {"sequence": [0, 1, 2, 3, 4, 5]}},
        logits_response=_logits_response(sequence),
        source_request_hash=_SOURCE_REQUEST_HASH,
        biohub_request_hash=request_hash,
        biohub_query_hash=_query_hash(sequence),
        biohub_api_base_url="https://biohub.ai",
        biohub_api_version="v1",
        model=DEFAULT_ESMC_MODEL,
        sae_model=DEFAULT_ESMC_SAE_MODEL,
        normalize_features=False,
        key_label="bu-dunlop-lab",
        retrieved_at=_RETRIEVED_AT,
    )
    artifacts = write_biohub_esmc_artifacts(
        output_root=tmp_path,
        profile_rows=[normalized.profile_row],
        protein_feature_rows=normalized.protein_feature_rows,
        residue_feature_rows=normalized.residue_feature_rows,
        feature_catalog_rows=normalized.feature_catalog_rows,
        request_manifest={
            "schema_id": "thread.biohub_esmc.request",
            "biohub_request_hash": request_hash,
            "key_label": "bu-dunlop-lab",
            "token": "<redacted>",
        },
        request_hash=request_hash,
    )

    assert (
        validate_biohub_esmc_artifacts(
            output_root=tmp_path,
            expected_candidate_ids={"candidate_a"},
            request_hash=request_hash,
        )
        == []
    )
    assert pq.read_table(artifacts.residue_features_path).num_rows == 3
    manifest_text = artifacts.request_manifest_path.read_text(encoding="utf-8")
    assert "fixture-secret" not in manifest_text
    assert "<redacted>" in manifest_text


def test_biohub_esmc_validator_accepts_explicit_error_rows(tmp_path: Path) -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    error_row = build_error_profile_row(
        candidate_id="candidate_a",
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        source_request_hash=_SOURCE_REQUEST_HASH,
        biohub_request_hash=request_hash,
        biohub_query_hash=_query_hash(sequence),
        biohub_api_base_url="https://biohub.ai",
        biohub_api_version="v1",
        model=DEFAULT_ESMC_MODEL,
        sae_model=DEFAULT_ESMC_SAE_MODEL,
        normalize_features=False,
        key_label="bu-dunlop-lab",
        retrieved_at=_RETRIEVED_AT,
        failure_reason="fixture failure",
    )

    write_biohub_esmc_artifacts(
        output_root=tmp_path,
        profile_rows=[error_row],
        protein_feature_rows=[],
        residue_feature_rows=[],
        feature_catalog_rows=[],
        request_manifest={
            "schema_id": "thread.biohub_esmc.request",
            "biohub_request_hash": request_hash,
            "key_label": "bu-dunlop-lab",
            "token": "<redacted>",
        },
        request_hash=request_hash,
    )

    assert (
        validate_biohub_esmc_artifacts(
            output_root=tmp_path,
            expected_candidate_ids={"candidate_a"},
            request_hash=request_hash,
        )
        == []
    )


def _request_hash(candidate_ids: tuple[str, ...]) -> str:
    return biohub_request_hash(
        {
            "source_request_hash": _SOURCE_REQUEST_HASH,
            "candidate_ids": list(candidate_ids),
            "biohub_api_base_url": "https://biohub.ai",
            "biohub_api_version": "v1",
            "model": DEFAULT_ESMC_MODEL,
            "sae_model": DEFAULT_ESMC_SAE_MODEL,
            "normalize_features": False,
        }
    )


def _query_hash(sequence: str) -> str:
    return biohub_query_hash(
        {
            "source_request_hash": _SOURCE_REQUEST_HASH,
            "sequence_hash": sequence_hash(sequence),
            "biohub_api_base_url": "https://biohub.ai",
            "biohub_api_version": "v1",
            "model": DEFAULT_ESMC_MODEL,
            "sae_model": DEFAULT_ESMC_SAE_MODEL,
            "normalize_features": False,
        }
    )


def _logits_response(sequence: str) -> dict[str, object]:
    tensor = torch.zeros((len(sequence) + 2, 16), dtype=torch.float32)
    tensor[1, 3] = 1.5
    tensor[2, 7] = 2.0
    tensor[4, 7] = 4.0
    return {"sae_outputs": {DEFAULT_ESMC_SAE_MODEL: tensor}, "logits": None, "embeddings": None}
