"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/biohub_esmc/test_table_validation_hardening.py

Biohub ESMC artifact validation hardening tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dnadesign.thread.adapters.biohub_esmc import (
    DEFAULT_ESMC_MODEL,
    biohub_query_hash,
    biohub_request_hash,
    normalize_logits_response,
    validate_biohub_esmc_artifacts,
    write_biohub_esmc_artifacts,
)
from dnadesign.thread.foldcheck import sequence_hash

_SOURCE_REQUEST_HASH = "sha256:" + "1" * 64
_RETRIEVED_AT = "2026-07-02T00:00:00Z"
_SAE_MODEL = "fixture-sae-k2-codebook8"


def test_biohub_esmc_validator_rejects_duplicate_profile_candidate_ids(tmp_path: Path) -> None:
    normalized = _normalized_rows("candidate_a", "AC")
    request_hash = _request_hash(("candidate_a",))
    write_biohub_esmc_artifacts(
        output_root=tmp_path,
        profile_rows=[normalized.profile_row, dict(normalized.profile_row)],
        protein_feature_rows=normalized.protein_feature_rows,
        residue_feature_rows=normalized.residue_feature_rows,
        feature_catalog_rows=normalized.feature_catalog_rows,
        request_manifest={"schema_id": "thread.biohub_esmc.request", "biohub_request_hash": request_hash},
        request_hash=request_hash,
    )

    issues = validate_biohub_esmc_artifacts(
        output_root=tmp_path,
        expected_candidate_ids={"candidate_a"},
        request_hash=request_hash,
    )

    assert {issue.check_id for issue in issues} == {"thread.biohub_esmc.profile_duplicate_candidate_id"}


def test_biohub_esmc_validator_rejects_stale_feature_row_sequence_hashes(tmp_path: Path) -> None:
    normalized = _normalized_rows("candidate_a", "AC")
    request_hash = _request_hash(("candidate_a",))
    stale_hash = sequence_hash("CA")
    protein_rows = [dict(row) for row in normalized.protein_feature_rows]
    residue_rows = [dict(row) for row in normalized.residue_feature_rows]
    protein_rows[0]["sequence_hash"] = stale_hash
    residue_rows[0]["sequence_hash"] = stale_hash
    write_biohub_esmc_artifacts(
        output_root=tmp_path,
        profile_rows=[normalized.profile_row],
        protein_feature_rows=protein_rows,
        residue_feature_rows=residue_rows,
        feature_catalog_rows=normalized.feature_catalog_rows,
        request_manifest={"schema_id": "thread.biohub_esmc.request", "biohub_request_hash": request_hash},
        request_hash=request_hash,
    )

    issues = validate_biohub_esmc_artifacts(
        output_root=tmp_path,
        expected_candidate_ids={"candidate_a"},
        request_hash=request_hash,
    )

    assert {
        "thread.biohub_esmc.protein_features_sequence_hash_mismatch",
        "thread.biohub_esmc.residue_features_sequence_hash_mismatch",
    } <= {issue.check_id for issue in issues}


@pytest.mark.parametrize(
    ("table_label", "mutated_rows_field"),
    [
        ("protein_features", "protein_feature_rows"),
        ("residue_features", "residue_feature_rows"),
    ],
)
def test_biohub_esmc_validator_rejects_cross_model_feature_rows(
    tmp_path: Path,
    table_label: str,
    mutated_rows_field: str,
) -> None:
    normalized = _normalized_rows("candidate_a", "AC")
    request_hash = _request_hash(("candidate_a",))
    protein_rows = [dict(row) for row in normalized.protein_feature_rows]
    residue_rows = [dict(row) for row in normalized.residue_feature_rows]
    mutated_rows = protein_rows if mutated_rows_field == "protein_feature_rows" else residue_rows
    for row in mutated_rows:
        row["sae_model"] = "different-sae-k2-codebook8"
    write_biohub_esmc_artifacts(
        output_root=tmp_path,
        profile_rows=[normalized.profile_row],
        protein_feature_rows=protein_rows,
        residue_feature_rows=residue_rows,
        feature_catalog_rows=normalized.feature_catalog_rows,
        request_manifest={"schema_id": "thread.biohub_esmc.request", "biohub_request_hash": request_hash},
        request_hash=request_hash,
    )

    issues = validate_biohub_esmc_artifacts(
        output_root=tmp_path,
        expected_candidate_ids={"candidate_a"},
        request_hash=request_hash,
    )

    assert {issue.check_id for issue in issues} == {f"thread.biohub_esmc.{table_label}_sae_model_mismatch"}


def _normalized_rows(candidate_id: str, sequence: str):
    tensor = torch.zeros((len(sequence) + 2, 8), dtype=torch.float32)
    tensor[1, 2] = 1.0
    tensor[1, 3] = 2.0
    tensor[2, 4] = 3.0
    tensor[2, 5] = 4.0
    return normalize_logits_response(
        candidate_id=candidate_id,
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        encode_response={"outputs": {"sequence": [0, 1, 2, 3]}},
        logits_response={"sae_outputs": {_SAE_MODEL: tensor}},
        source_request_hash=_SOURCE_REQUEST_HASH,
        biohub_request_hash=_request_hash((candidate_id,)),
        biohub_query_hash=_query_hash(sequence),
        biohub_api_base_url="https://biohub.ai",
        biohub_api_version="v1",
        model=DEFAULT_ESMC_MODEL,
        sae_model=_SAE_MODEL,
        normalize_features=False,
        key_label="bu-dunlop-lab",
        retrieved_at=_RETRIEVED_AT,
    )


def _request_hash(candidate_ids: tuple[str, ...]) -> str:
    return biohub_request_hash(
        {
            "source_request_hash": _SOURCE_REQUEST_HASH,
            "candidate_ids": list(candidate_ids),
            "biohub_api_base_url": "https://biohub.ai",
            "biohub_api_version": "v1",
            "model": DEFAULT_ESMC_MODEL,
            "sae_model": _SAE_MODEL,
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
            "sae_model": _SAE_MODEL,
            "normalize_features": False,
        }
    )
