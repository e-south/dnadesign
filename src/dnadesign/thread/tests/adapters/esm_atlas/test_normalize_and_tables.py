"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/esm_atlas/test_normalize_and_tables.py

ESM Atlas normalization and artifact contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.thread.adapters.esm_atlas import (
    atlas_query_hash,
    atlas_request_hash,
    build_atlas_structure_prediction_row,
    normalize_protein_lookup_response,
    sequence_md5,
    validate_atlas_api_base_url,
    validate_atlas_semantic_artifacts,
    write_atlas_semantic_artifacts,
)
from dnadesign.thread.adapters.esm_atlas import client as atlas_client_module
from dnadesign.thread.adapters.esm_atlas.client import AtlasClient
from dnadesign.thread.foldcheck import sequence_hash
from dnadesign.thread.structure_predictions import (
    validate_structure_prediction_registry,
    write_structure_prediction_registry,
)

_SOURCE_REQUEST_HASH = "sha256:" + "1" * 64
_RETRIEVED_AT = "2026-06-25T00:00:00Z"


def test_normalize_protein_lookup_response_keeps_sparse_rows_compact() -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    normalized = normalize_protein_lookup_response(
        candidate_id="candidate_a",
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        response=_protein_response(sequence),
        source_request_hash=_SOURCE_REQUEST_HASH,
        atlas_request_hash=request_hash,
        atlas_query_hash=_query_hash(sequence),
        atlas_api_base_url="https://biohub.ai",
        atlas_api_version="v1alpha1",
        retrieved_at=_RETRIEVED_AT,
    )

    assert normalized.profile_row["atlas_hash"] == sequence_md5(sequence)
    assert normalized.profile_row["top_feature_indices"] == [14365, 10777]
    assert normalized.protein_activation_rows == [
        {
            "candidate_id": "candidate_a",
            "sequence_hash": sequence_hash(sequence),
            "feature_index": 14365,
            "value": 1.2,
        },
        {
            "candidate_id": "candidate_a",
            "sequence_hash": sequence_hash(sequence),
            "feature_index": 10777,
            "value": 0.9,
        },
    ]
    assert normalized.residue_activation_rows[0] == {
        "candidate_id": "candidate_a",
        "sequence_hash": sequence_hash(sequence),
        "residue_index_zero_based": 0,
        "sequence_position_one_based": 1,
        "feature_index": 14365,
        "value": 4.0,
    }
    assert all("description" not in row for row in normalized.residue_activation_rows)


def test_atlas_semantic_artifact_writer_validates_expected_candidates(tmp_path: Path) -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    normalized = normalize_protein_lookup_response(
        candidate_id="candidate_a",
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        response=_protein_response(sequence),
        source_request_hash=_SOURCE_REQUEST_HASH,
        atlas_request_hash=request_hash,
        atlas_query_hash=_query_hash(sequence),
        atlas_api_base_url="https://biohub.ai",
        atlas_api_version="v1alpha1",
        retrieved_at=_RETRIEVED_AT,
    )

    artifacts = write_atlas_semantic_artifacts(
        output_root=tmp_path,
        profile_rows=[normalized.profile_row],
        protein_activation_rows=normalized.protein_activation_rows,
        residue_activation_rows=normalized.residue_activation_rows,
        feature_catalog_rows=normalized.feature_catalog_rows,
        request_hash=request_hash,
    )

    assert (
        validate_atlas_semantic_artifacts(
            output_root=tmp_path,
            expected_candidate_ids={"candidate_a"},
            request_hash=request_hash,
        )
        == []
    )
    assert pq.read_table(artifacts.feature_catalog_path).num_rows == 2
    assert pq.read_table(artifacts.residue_activations_path).num_rows == 4


def test_normalize_rejects_unpermitted_on_demand_folding() -> None:
    sequence = "ACDE"
    response = _protein_response(sequence)
    response["folded_on_demand"] = True

    with pytest.raises(ValueError, match="fold_on_miss"):
        normalize_protein_lookup_response(
            candidate_id="candidate_a",
            sequence=sequence,
            sequence_hash=sequence_hash(sequence),
            response=response,
            source_request_hash=_SOURCE_REQUEST_HASH,
            atlas_request_hash=_request_hash(("candidate_a",)),
            atlas_query_hash=_query_hash(sequence),
            atlas_api_base_url="https://biohub.ai",
            atlas_api_version="v1alpha1",
            retrieved_at=_RETRIEVED_AT,
        )


def test_atlas_validator_allows_on_demand_folding_only_when_explicit(tmp_path: Path) -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    response = _protein_response(sequence)
    response["folded_on_demand"] = True
    normalized = normalize_protein_lookup_response(
        candidate_id="candidate_a",
        sequence=sequence,
        sequence_hash=sequence_hash(sequence),
        response=response,
        source_request_hash=_SOURCE_REQUEST_HASH,
        atlas_request_hash=request_hash,
        atlas_query_hash=_query_hash(sequence),
        atlas_api_base_url="https://biohub.ai",
        atlas_api_version="v1alpha1",
        retrieved_at=_RETRIEVED_AT,
        allow_fold_on_miss=True,
    )
    write_atlas_semantic_artifacts(
        output_root=tmp_path,
        profile_rows=[normalized.profile_row],
        protein_activation_rows=normalized.protein_activation_rows,
        residue_activation_rows=normalized.residue_activation_rows,
        feature_catalog_rows=normalized.feature_catalog_rows,
        request_hash=request_hash,
    )

    default_issues = validate_atlas_semantic_artifacts(
        output_root=tmp_path,
        expected_candidate_ids={"candidate_a"},
        request_hash=request_hash,
    )
    permitted_issues = validate_atlas_semantic_artifacts(
        output_root=tmp_path,
        expected_candidate_ids={"candidate_a"},
        request_hash=request_hash,
        allow_fold_on_miss=True,
    )

    assert [issue.check_id for issue in default_issues] == ["thread.esm_atlas.profile_folded_on_demand"]
    assert permitted_issues == []


def test_atlas_structure_prediction_registry_saves_pdb_payload(tmp_path: Path) -> None:
    sequence = "ACDE"
    request_hash = _request_hash(("candidate_a",))
    response = _protein_response(sequence)
    response.update(
        {
            "folded_on_demand": True,
            "pdb": _PDB_TEXT,
            "mean_plddt": 86.0,
            "ptm": 0.91,
            "pae_summary": {"mean": 4.0},
        }
    )

    row = build_atlas_structure_prediction_row(
        candidate_id="candidate_a",
        sequence_hash=sequence_hash(sequence),
        response=response,
        atlas_request_hash=request_hash,
        source_request_hash=_SOURCE_REQUEST_HASH,
        prediction_set_id="atlas_fixture_fold_on_miss",
        output_root=tmp_path / "structures",
        atlas_api_base_url="https://biohub.ai",
        atlas_api_version="v1alpha1",
    )
    if row is None:
        raise AssertionError("folded-on-demand response must emit a structure-prediction row")
    artifacts = write_structure_prediction_registry(
        output_root=tmp_path / "registry",
        rows=[row],
        request_hash=request_hash,
    )

    assert Path(str(row["local_structure_path"])).exists()
    assert row["backend_kind"] == "esm_atlas"
    assert (
        validate_structure_prediction_registry(
            registry_path=artifacts.registry_path,
            request_hash=request_hash,
        )
        == []
    )


def test_normalize_rejects_sparse_shape_drift() -> None:
    sequence = "ACDE"
    response = _protein_response(sequence)
    response["per_residue_activations"]["shape"] = [4, 4096]

    with pytest.raises(ValueError, match="16,384-feature"):
        normalize_protein_lookup_response(
            candidate_id="candidate_a",
            sequence=sequence,
            sequence_hash=sequence_hash(sequence),
            response=response,
            source_request_hash=_SOURCE_REQUEST_HASH,
            atlas_request_hash=_request_hash(("candidate_a",)),
            atlas_query_hash=_query_hash(sequence),
            atlas_api_base_url="https://biohub.ai",
            atlas_api_version="v1alpha1",
            retrieved_at=_RETRIEVED_AT,
        )


def test_normalize_rejects_residue_activation_length_drift() -> None:
    sequence = "ACDE"
    response = _protein_response(sequence)
    response["per_residue_activations"]["shape"] = [5, 16384]

    with pytest.raises(ValueError, match="sequence length"):
        normalize_protein_lookup_response(
            candidate_id="candidate_a",
            sequence=sequence,
            sequence_hash=sequence_hash(sequence),
            response=response,
            source_request_hash=_SOURCE_REQUEST_HASH,
            atlas_request_hash=_request_hash(("candidate_a",)),
            atlas_query_hash=_query_hash(sequence),
            atlas_api_base_url="https://biohub.ai",
            atlas_api_version="v1alpha1",
            retrieved_at=_RETRIEVED_AT,
        )


def test_client_rejects_unbounded_feature_request() -> None:
    client = AtlasClient()

    with pytest.raises(ValueError, match="topk_features"):
        client.protein_lookup_by_hash(sequence_md5("ACDE"), topk_features=101)


def test_validate_atlas_api_base_url_accepts_public_hosts() -> None:
    assert validate_atlas_api_base_url("https://biohub.ai/") == "https://biohub.ai"
    assert validate_atlas_api_base_url("https://www.biohub.ai") == "https://www.biohub.ai"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://biohub.ai",
        "https://biohub.ai.example.org",
        "https://biohub.ai:8443",
        "https://biohub.ai/esm/protein/api/v1alpha1",
        "https://token@biohub.ai",
        "https://biohub.ai?redirect=https://example.org",
    ],
)
def test_validate_atlas_api_base_url_rejects_untrusted_or_ambiguous_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="Atlas API base URL"):
        validate_atlas_api_base_url(base_url)


def test_atlas_client_rejects_untrusted_base_url_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Atlas client should reject the base URL before opening a request")

    monkeypatch.setattr(atlas_client_module, "urlopen", fail_urlopen)

    with pytest.raises(ValueError, match="Atlas API base URL"):
        AtlasClient(base_url="https://example.org").protein_lookup_by_hash(sequence_md5("ACDE"))


def _request_hash(candidate_ids: tuple[str, ...]) -> str:
    return atlas_request_hash(
        {
            "source_request_hash": _SOURCE_REQUEST_HASH,
            "candidate_ids": list(candidate_ids),
            "atlas_api_base_url": "https://biohub.ai",
            "atlas_api_version": "v1alpha1",
            "topk_features": 100,
            "fold_on_miss": False,
        }
    )


def _query_hash(sequence: str) -> str:
    return atlas_query_hash(
        {
            "source_request_hash": _SOURCE_REQUEST_HASH,
            "sequence_md5": sequence_md5(sequence),
            "atlas_api_base_url": "https://biohub.ai",
            "atlas_api_version": "v1alpha1",
            "topk_features": 100,
            "fold_on_miss": False,
            "normalize_features": True,
        }
    )


def _protein_response(sequence: str) -> dict[str, object]:
    return {
        "protein_hash": sequence_md5(sequence),
        "accession": "fixture_accession",
        "source": "fixture_source",
        "sequence": sequence,
        "sequence_length": len(sequence),
        "folded_on_demand": False,
        "sae_features": [
            {
                "feature_index": 14365,
                "value": 1.2,
                "label": "Polymerase thumb/palm nucleic acid binding",
                "description": "Fixture feature",
                "residue_regions": [{"start": 0, "end": 3, "peak_residue": 1, "mean_activation": 3.0}],
            },
            {
                "feature_index": 10777,
                "value": 0.9,
                "label": "RT/RdRp pre-catalytic region",
                "description": "Fixture feature",
                "residue_regions": [{"start": 1, "end": 2, "peak_residue": 1, "mean_activation": 2.0}],
            },
        ],
        "protein_activations": {
            "shape": [16384],
            "indices": [[14365, 10777]],
            "values": [1.2, 0.9],
        },
        "per_residue_activations": {
            "shape": [4, 16384],
            "indices": [[0, 1, 2, 3], [14365, 14365, 10777, 9008]],
            "values": [4.0, 3.0, 2.0, 1.0],
        },
    }


_PDB_TEXT = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 86.00           C\nEND\n"
