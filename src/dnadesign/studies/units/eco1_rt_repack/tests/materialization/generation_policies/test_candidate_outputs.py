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
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies import (
    PRIMARY_POLICY_IDS,
    materialize_generation_policies,
    materialize_generation_policy_candidate_pool,
    materialize_generation_policy_foldcheck_request,
)

from ._candidate_tables import candidate_row, write_candidate_table


def test_generation_policy_candidate_pool_aggregates_complete_policy_outputs(tmp_path: Path) -> None:
    materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path)
    write_candidate_table(
        tmp_path / "distal_scaffold_repack_v1" / "candidate_table.parquet",
        [
            candidate_row("thread_candidate_a", "sha256:a", ["M20I"], 1),
            candidate_row("thread_candidate_shared", "sha256:shared", ["M20I", "D25S"], 2),
        ],
    )
    write_candidate_table(
        tmp_path / "near_dna_rna_acid_free_v1" / "candidate_table.parquet",
        [
            candidate_row("thread_candidate_b", "sha256:b", ["N21R"], 1),
            candidate_row("thread_candidate_shared", "sha256:shared", ["M20I", "D25S"], 2),
        ],
    )
    write_candidate_table(
        tmp_path / "combined_near_acid_free_plus_distal_v1" / "candidate_table.parquet",
        [candidate_row("thread_candidate_c", "sha256:c", ["M20I", "N21R"], 1)],
    )

    result = materialize_generation_policy_candidate_pool(repo_root=Path.cwd(), generation_policy_root=tmp_path)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = pq.read_table(result.candidate_pool_path).to_pylist()

    assert result.policy_manifest_path == tmp_path / "generation_policy_manifest.yaml"
    assert manifest["schema_id"] == "eco1_rt.generation_policy_candidate_pool_manifest"
    assert manifest["candidate_pool_row_count"] == 4
    assert manifest["duplicate_sequence_count"] == 1
    assert len(rows) == 4
    assert {row["policy_id"] for row in rows} <= set(PRIMARY_POLICY_IDS)
    shared = next(row for row in rows if row["sequence_hash"] == "sha256:shared")
    assert shared["source_policy_ids"] == ["distal_scaffold_repack_v1", "near_dna_rna_acid_free_v1"]
    assert shared["primary_policy_id"] == "distal_scaffold_repack_v1"


def test_generation_policy_foldcheck_request_writes_v2_fasta(tmp_path: Path) -> None:
    materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path)
    write_candidate_table(
        tmp_path / "distal_scaffold_repack_v1" / "candidate_table.parquet",
        [candidate_row("thread_candidate_a", "sha256:a", ["M20I"], 1)],
    )
    write_candidate_table(
        tmp_path / "near_dna_rna_acid_free_v1" / "candidate_table.parquet",
        [candidate_row("thread_candidate_b", "sha256:b", ["N21R"], 1)],
    )
    write_candidate_table(
        tmp_path / "combined_near_acid_free_plus_distal_v1" / "candidate_table.parquet",
        [candidate_row("thread_candidate_c", "sha256:c", ["M20I", "N21R"], 1)],
    )

    result = materialize_generation_policy_foldcheck_request(repo_root=Path.cwd(), generation_policy_root=tmp_path)
    manifest = yaml.safe_load(result.request_manifest_path.read_text(encoding="utf-8"))
    fasta = result.input_fasta_path.read_text(encoding="utf-8")

    assert result.candidate_pool_path == tmp_path / "candidate_pool.parquet"
    assert manifest["schema_id"] == "thread.foldcheck_request"
    assert manifest["artifact_id"] == "eco1_rt_generation_policies_v2.foldcheck_request"
    assert manifest["sequence_count"] == 4
    assert manifest["storage_policy"]["preferred_runtime_locus"] == "local_julius_colabfold"
    assert ">wild_type" in fasta
    assert ">thread_candidate_a" in fasta
    assert ">thread_candidate_b" in fasta
    assert ">thread_candidate_c" in fasta
