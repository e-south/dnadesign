"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/foldcheck/test_request.py

Fold-check request tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.thread.foldcheck import (
    FOLDCHECK_REQUEST_SCHEMA_ID,
    FoldCheckSequenceRecord,
    build_foldcheck_request_manifest,
    sequence_hash,
    write_foldcheck_fasta,
)


def test_foldcheck_request_manifest_requires_wt_sequence(tmp_path: Path) -> None:
    records = [
        FoldCheckSequenceRecord(
            sequence_id="candidate_a",
            sequence="ACDE",
            sequence_hash=sequence_hash("ACDE"),
            source_kind="candidate",
        )
    ]

    try:
        build_foldcheck_request_manifest(
            artifact_id="artifact",
            created_by="test",
            created_at="2026-06-25T00:00:00Z",
            backend_kind="colabfold",
            runtime_kind="alphafold_family_colabfold",
            execution_status="planned_not_run",
            input_fasta_path=tmp_path / "input.fasta",
            output_root=tmp_path / "outputs",
            sequence_records=records,
            wt_sequence_id="wild_type",
            reference_structure_id="reference",
            threshold_policy_id="foldcheck_thresholds_v1",
            threshold_values={"min_plddt": 70},
            upstream_artifact_hashes={"candidate_table": "sha256:" + "a" * 64},
            storage_policy={"raw_outputs": "external_runtime"},
        )
    except ValueError as error:
        assert "WT baseline" in str(error)
    else:
        raise AssertionError("Expected missing WT baseline to fail")


def test_foldcheck_fasta_and_manifest_are_model_agnostic(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    records = [
        FoldCheckSequenceRecord("wild_type", "ACDEFG", sequence_hash("ACDEFG"), "wild_type"),
        FoldCheckSequenceRecord("candidate_a", "ACDFFG", sequence_hash("ACDFFG"), "candidate"),
    ]

    write_foldcheck_fasta(fasta, records)
    manifest = build_foldcheck_request_manifest(
        artifact_id="artifact",
        created_by="test",
        created_at="2026-06-25T00:00:00Z",
        backend_kind="colabfold",
        runtime_kind="alphafold_family_colabfold",
        execution_status="planned_not_run",
        input_fasta_path=fasta,
        output_root=tmp_path / "outputs",
        sequence_records=records,
        wt_sequence_id="wild_type",
        reference_structure_id="reference",
        threshold_policy_id="foldcheck_thresholds_v1",
        threshold_values={"min_plddt": 70},
        upstream_artifact_hashes={"candidate_table": "sha256:" + "a" * 64},
        storage_policy={"raw_outputs": "external_runtime"},
    )

    assert fasta.read_text(encoding="utf-8") == ">wild_type\nACDEFG\n>candidate_a\nACDFFG\n"
    assert manifest["schema_id"] == FOLDCHECK_REQUEST_SCHEMA_ID
    assert manifest["request_hash"].startswith("sha256:")
    assert manifest["sequence_count"] == 2
    assert manifest["backend_kind"] == "colabfold"


def test_foldcheck_request_hash_is_portable_across_local_paths(tmp_path: Path) -> None:
    records = [
        FoldCheckSequenceRecord("wild_type", "ACDEFG", sequence_hash("ACDEFG"), "wild_type"),
        FoldCheckSequenceRecord("candidate_a", "ACDFFG", sequence_hash("ACDFFG"), "candidate"),
    ]
    common_kwargs = {
        "artifact_id": "artifact",
        "created_by": "test",
        "created_at": "2026-06-25T00:00:00Z",
        "backend_kind": "colabfold",
        "runtime_kind": "alphafold_family_colabfold",
        "execution_status": "planned_not_run",
        "sequence_records": records,
        "wt_sequence_id": "wild_type",
        "reference_structure_id": "reference",
        "threshold_policy_id": "foldcheck_thresholds_v1",
        "threshold_values": {"min_plddt": 70},
        "upstream_artifact_hashes": {"candidate_table": "sha256:" + "a" * 64},
        "storage_policy": {"raw_outputs": "external_runtime"},
    }

    laptop_manifest = build_foldcheck_request_manifest(
        input_fasta_path=tmp_path / "laptop/input.fasta",
        output_root=tmp_path / "laptop/outputs",
        **common_kwargs,
    )
    scc_manifest = build_foldcheck_request_manifest(
        input_fasta_path=Path("/project/dunlop/esouth/dnadesign/input.fasta"),
        output_root=Path("/project/dunlop/esouth/foldcheck/outputs"),
        **common_kwargs,
    )

    assert laptop_manifest["input_fasta_path"] != scc_manifest["input_fasta_path"]
    assert laptop_manifest["output_root"] != scc_manifest["output_root"]
    assert laptop_manifest["request_hash"] == scc_manifest["request_hash"]
