"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/biohub_sae_fixtures.py

Biohub ESMC SAE fixtures for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.thread.adapters.biohub_esmc import DEFAULT_ESMC_MODEL, DEFAULT_ESMC_SAE_MODEL


def write_biohub_esmc_sae_outputs(output_root: Path) -> None:
    """Write compact query-time Biohub ESMC SAE fixture outputs."""

    output_root.joinpath("biohub_esmc_request_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "thread.biohub_esmc.request",
                "status": "materialized",
                "endpoint_flow": ["POST /api/v1/encode", "POST /api/v1/logits"],
                "model": DEFAULT_ESMC_MODEL,
                "sae_model": DEFAULT_ESMC_SAE_MODEL,
                "authorization": "<redacted>",
                "method_references": [
                    {
                        "title": "Biohub ESMC SAE feature interpretation notebook",
                        "url": (
                            "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/"
                            "tutorials/esmc_sae_feature_interpretation.ipynb"
                        ),
                        "role": "SAE feature inspection and residue-localization reference",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(_profile_rows()), output_root / "biohub_esmc_sae_profile.parquet")
    pq.write_table(pa.Table.from_pylist(_protein_feature_rows()), output_root / "biohub_esmc_protein_features.parquet")
    pq.write_table(pa.Table.from_pylist(_residue_feature_rows()), output_root / "biohub_esmc_residue_features.parquet")
    pq.write_table(pa.Table.from_pylist(_feature_catalog_rows()), output_root / "biohub_esmc_feature_catalog.parquet")


def _profile_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate_id,
            "sequence_hash": "sha256:" + str(index + 5) * 64,
            "source_request_hash": "sha256:" + "4" * 64,
            "biohub_request_hash": "sha256:" + "5" * 64,
            "biohub_query_hash": "sha256:" + str(index + 6) * 64,
            "biohub_api_base_url": "https://biohub.ai",
            "biohub_api_version": "v1",
            "model": DEFAULT_ESMC_MODEL,
            "sae_model": DEFAULT_ESMC_SAE_MODEL,
            "normalize_features": True,
            "key_label": "fixture",
            "sequence_length": 6,
            "token_count": 8,
            "feature_dictionary_size": 16384,
            "status": "accepted",
            "protein_feature_count": 3,
            "residue_feature_count": 18,
            "encoded_sae_bytes": 2048,
            "raw_encode_response_hash": "sha256:" + "7" * 64,
            "raw_logits_response_hash": "sha256:" + "8" * 64,
            "retrieved_at": "2026-06-25T00:00:00Z",
            "failure_reason": "",
        }
        for index, candidate_id in enumerate(("wild_type", "thread_candidate_alpha", "thread_candidate_beta"))
    ]


def _protein_feature_rows() -> list[dict[str, object]]:
    values = {
        "wild_type": {101: 12.0, 202: 8.0, 303: 4.0},
        "thread_candidate_alpha": {101: 11.0, 202: 7.4, 303: 4.5},
        "thread_candidate_beta": {101: 4.0, 202: 10.5, 303: 1.0},
    }
    rows: list[dict[str, object]] = []
    for candidate_id, feature_values in values.items():
        for feature_index, activation_sum in feature_values.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "sequence_hash": "sha256:" + "9" * 64,
                    "sae_model": DEFAULT_ESMC_SAE_MODEL,
                    "feature_index": feature_index,
                    "sequence_residue_count": 6,
                    "nonzero_residue_count": 3,
                    "activation_sum": activation_sum,
                    "activation_mean": activation_sum / 6.0,
                    "activation_max": activation_sum / 2.0,
                }
            )
    return rows


def _residue_feature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_index, values in {
        101: [3.0, 4.0, 0.0, 0.0, 5.0, 0.0],
        202: [0.0, 2.0, 3.0, 0.0, 0.0, 3.0],
        303: [0.0, 0.0, 1.0, 1.5, 1.5, 0.0],
    }.items():
        for position, value in enumerate(values, start=1):
            if value <= 0:
                continue
            rows.append(
                {
                    "candidate_id": "wild_type",
                    "sequence_hash": "sha256:" + "9" * 64,
                    "sae_model": DEFAULT_ESMC_SAE_MODEL,
                    "residue_index_zero_based": position - 1,
                    "sequence_position_one_based": position,
                    "feature_index": feature_index,
                    "value": value,
                }
            )
    return rows


def _feature_catalog_rows() -> list[dict[str, object]]:
    rows = []
    for feature_index in (101, 202, 303):
        rows.append(
            {
                "sae_model": DEFAULT_ESMC_SAE_MODEL,
                "feature_index": feature_index,
                "label": "fixture_peak_feature" if feature_index == 101 else "",
                "description": "Fixture exact-dictionary feature description." if feature_index == 101 else "",
                "source_retrieved_at": "2026-06-25T00:00:00Z",
                "raw_feature_hash": "sha256:" + "a" * 64 if feature_index == 101 else "",
            }
        )
    return rows
