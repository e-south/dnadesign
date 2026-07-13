"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies/test_candidate_outputs.py

Candidate-pool and foldcheck materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    GENERATION_POLICY_VERSION,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    PRIMARY_POLICY_IDS,
    materialize_generation_policies,
    materialize_generation_policy_candidate_pool,
    materialize_generation_policy_foldcheck_request,
    materialize_generation_policy_requests,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import require_ec86kit_source_artifacts

from ._candidate_tables import (
    candidate_row,
    request_hash_for_policy,
    sequence_for_index,
    write_candidate_table,
    write_policy_sample_table,
)


def test_generation_policy_candidate_pool_aggregates_complete_policy_outputs(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path)
    materialize_generation_policy_requests(repo_root=Path.cwd(), generation_policy_root=tmp_path)
    distal_hash = request_hash_for_policy(tmp_path, "distal_scaffold_repack_v1")
    near_hash = request_hash_for_policy(tmp_path, NEAR_DNA_RNA_ACID_FREE_POLICY_ID)
    combined_hash = request_hash_for_policy(tmp_path, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID)
    distal_rows = [
        candidate_row("thread_candidate_a", sequence_for_index(1), ["M20I"], 1, request_hash=distal_hash),
        candidate_row(
            "thread_candidate_shared",
            sequence_for_index(4),
            ["M20I", "D25S"],
            2,
            request_hash=distal_hash,
        ),
    ]
    write_candidate_table(
        tmp_path / "distal_scaffold_repack_v1" / "candidate_table.parquet",
        distal_rows,
        request_hash=distal_hash,
    )
    write_policy_sample_table(tmp_path, "distal_scaffold_repack_v1", distal_rows)
    near_rows = [
        candidate_row("thread_candidate_b", sequence_for_index(2), ["N21R"], 1, request_hash=near_hash),
        candidate_row(
            "thread_candidate_shared",
            sequence_for_index(4),
            ["M20I", "D25S"],
            2,
            request_hash=near_hash,
        ),
    ]
    write_candidate_table(
        tmp_path / NEAR_DNA_RNA_ACID_FREE_POLICY_ID / "candidate_table.parquet",
        near_rows,
        request_hash=near_hash,
    )
    write_policy_sample_table(tmp_path, NEAR_DNA_RNA_ACID_FREE_POLICY_ID, near_rows)
    combined_rows = [
        candidate_row(
            "thread_candidate_c",
            sequence_for_index(3),
            ["M20I", "N21R"],
            1,
            request_hash=combined_hash,
        )
    ]
    write_candidate_table(
        tmp_path / COMBINED_NEAR_PLUS_DISTAL_POLICY_ID / "candidate_table.parquet",
        combined_rows,
        request_hash=combined_hash,
    )
    write_policy_sample_table(tmp_path, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID, combined_rows)

    result = materialize_generation_policy_candidate_pool(repo_root=Path.cwd(), generation_policy_root=tmp_path)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = pq.read_table(result.candidate_pool_path).to_pylist()

    assert result.policy_manifest_path == tmp_path / "generation_policy_manifest.yaml"
    assert manifest["schema_id"] == "eco1_rt.generation_policy_candidate_pool_manifest"
    assert manifest["candidate_pool_row_count"] == 4
    assert manifest["duplicate_sequence_count"] == 1
    assert len(rows) == 4
    assert {row["policy_id"] for row in rows} <= set(PRIMARY_POLICY_IDS)
    shared = next(row for row in rows if "thread_candidate_shared" in row["source_candidate_ids"])
    assert shared["source_policy_ids"] == ["distal_scaffold_repack_v1", NEAR_DNA_RNA_ACID_FREE_POLICY_ID]
    assert shared["primary_policy_id"] == "distal_scaffold_repack_v1"
    assert all(row["policy_manifest_hash"] == manifest["policy_manifest_hash"] for row in rows)


def test_generation_policy_candidate_pool_rejects_stale_request_provenance(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path)
    materialize_generation_policy_requests(repo_root=Path.cwd(), generation_policy_root=tmp_path)
    for policy_id in PRIMARY_POLICY_IDS:
        expected_hash = request_hash_for_policy(tmp_path, policy_id)
        row_hash = "sha256:stale" if policy_id == NEAR_DNA_RNA_ACID_FREE_POLICY_ID else expected_hash
        candidate_rows = [
            candidate_row(
                f"candidate_{policy_id}",
                sequence_for_index(PRIMARY_POLICY_IDS.index(policy_id) + 1),
                ["M20I"],
                1,
                request_hash=row_hash,
            )
        ]
        write_candidate_table(
            tmp_path / policy_id / "candidate_table.parquet",
            candidate_rows,
            request_hash=row_hash,
        )
        write_policy_sample_table(tmp_path, policy_id, candidate_rows)

    with pytest.raises(ValueError, match="candidate table validation failed"):
        materialize_generation_policy_candidate_pool(repo_root=Path.cwd(), generation_policy_root=tmp_path)


def test_generation_policy_foldcheck_request_writes_active_policy_fasta(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path)
    materialize_generation_policy_requests(repo_root=Path.cwd(), generation_policy_root=tmp_path)
    distal_hash = request_hash_for_policy(tmp_path, "distal_scaffold_repack_v1")
    near_hash = request_hash_for_policy(tmp_path, NEAR_DNA_RNA_ACID_FREE_POLICY_ID)
    combined_hash = request_hash_for_policy(tmp_path, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID)
    distal_rows = [candidate_row("thread_candidate_a", sequence_for_index(1), ["M20I"], 1, request_hash=distal_hash)]
    write_candidate_table(
        tmp_path / "distal_scaffold_repack_v1" / "candidate_table.parquet",
        distal_rows,
        request_hash=distal_hash,
    )
    write_policy_sample_table(tmp_path, "distal_scaffold_repack_v1", distal_rows)
    near_rows = [candidate_row("thread_candidate_b", sequence_for_index(2), ["N21R"], 1, request_hash=near_hash)]
    write_candidate_table(
        tmp_path / NEAR_DNA_RNA_ACID_FREE_POLICY_ID / "candidate_table.parquet",
        near_rows,
        request_hash=near_hash,
    )
    write_policy_sample_table(tmp_path, NEAR_DNA_RNA_ACID_FREE_POLICY_ID, near_rows)
    combined_rows = [
        candidate_row(
            "thread_candidate_c",
            sequence_for_index(3),
            ["M20I", "N21R"],
            1,
            request_hash=combined_hash,
        )
    ]
    write_candidate_table(
        tmp_path / COMBINED_NEAR_PLUS_DISTAL_POLICY_ID / "candidate_table.parquet",
        combined_rows,
        request_hash=combined_hash,
    )
    write_policy_sample_table(tmp_path, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID, combined_rows)

    result = materialize_generation_policy_foldcheck_request(
        repo_root=Path.cwd(),
        generation_policy_root=tmp_path,
    )
    manifest = yaml.safe_load(result.request_manifest_path.read_text(encoding="utf-8"))
    fasta = result.input_fasta_path.read_text(encoding="utf-8")

    assert result.candidate_pool_path == tmp_path / "candidate_pool.parquet"
    assert manifest["schema_id"] == "thread.foldcheck_request"
    assert manifest["artifact_id"] == f"eco1_rt_generation_policies_v{GENERATION_POLICY_VERSION}.foldcheck_request"
    assert manifest["sequence_count"] == 4
    assert manifest["storage_policy"]["preferred_runtime_locus"] == "local_julius_colabfold"
    assert ">wild_type" in fasta
    assert ">thread_candidate_a" in fasta
    assert ">thread_candidate_b" in fasta
    assert ">thread_candidate_c" in fasta
