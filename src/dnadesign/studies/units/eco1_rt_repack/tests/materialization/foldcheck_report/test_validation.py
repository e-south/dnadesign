"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_report/test_validation.py

Eco1 fold-check report authority-validation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.foldcheck import (
    validate_foldcheck_report_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report import (
    materialize_foldcheck_report,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request import (
    materialize_foldcheck_request,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_report.test_materialization import (
    _write_ca_pdb,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_request._fixtures import (
    write_minimal_foldcheck_inputs,
)
from dnadesign.thread.foldcheck import write_foldcheck_report


def test_foldcheck_report_validator_rejects_sequence_hash_drift(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    colabfold_root = request.request_manifest_path.parent / "colabfold_outputs"
    colabfold_root.mkdir()
    _write_ca_pdb(tmp_path / "proteinmpnn_request/chain_a_backbone.pdb", bfactor=0.0, residue_count=309)
    _write_ca_pdb(
        colabfold_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=90.0,
        residue_count=320,
    )
    _write_ca_pdb(
        colabfold_root / "thread_candidate_test_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=82.0,
        residue_count=320,
    )
    result = materialize_foldcheck_report(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        colabfold_output_root=colabfold_root,
        runtime_version="colabfold-test",
    )
    rows = pq.read_table(result.foldcheck_report_path).to_pylist()
    rows[1]["input_sequence_hash"] = "sha256:" + "9" * 64
    manifest = yaml.safe_load(request.request_manifest_path.read_text(encoding="utf-8"))
    write_foldcheck_report(result.foldcheck_report_path, rows, request_hash=str(manifest["request_hash"]))

    issues = validate_foldcheck_report_content(
        result.foldcheck_report_path,
        repo_root=Path.cwd(),
        output_root=tmp_path,
    )

    assert [issue.check_id for issue in issues] == ["eco1_rt.foldcheck_report.sequence_hash_mismatch"]


def test_foldcheck_report_validator_uses_generation_policy_candidate_pool(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    (tmp_path / "candidate_table.parquet").rename(tmp_path / "candidate_pool.parquet")
    manifest = yaml.safe_load(request.request_manifest_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "foldcheck_report.parquet"
    write_foldcheck_report(
        report_path,
        [_report_row(row) for row in manifest["sequences"]],
        request_hash=str(manifest["request_hash"]),
    )

    assert validate_foldcheck_report_content(report_path, repo_root=Path.cwd(), output_root=tmp_path) == []


def test_foldcheck_report_validator_requires_candidate_authority(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    (tmp_path / "candidate_table.parquet").unlink()
    manifest = yaml.safe_load(request.request_manifest_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "foldcheck_report.parquet"
    write_foldcheck_report(
        report_path,
        [_report_row(row) for row in manifest["sequences"]],
        request_hash=str(manifest["request_hash"]),
    )

    issues = validate_foldcheck_report_content(report_path, repo_root=Path.cwd(), output_root=tmp_path)

    assert [issue.check_id for issue in issues] == ["eco1_rt.foldcheck_report.candidate_authority_missing"]


def test_foldcheck_report_validator_does_not_depend_on_process_cwd(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path.cwd()
    output_root = tmp_path / "report"
    write_minimal_foldcheck_inputs(output_root)
    request = materialize_foldcheck_request(repo_root=repo_root, output_root=output_root)
    colabfold_root = output_root / "foldcheck_request/colabfold_outputs"
    colabfold_root.mkdir()
    _write_ca_pdb(output_root / "proteinmpnn_request/chain_a_backbone.pdb", bfactor=0.0, residue_count=309)
    _write_ca_pdb(
        colabfold_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=90.0,
        residue_count=320,
    )
    _write_ca_pdb(
        colabfold_root / "thread_candidate_test_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=82.0,
        residue_count=320,
    )
    result = materialize_foldcheck_report(
        repo_root=repo_root,
        output_root=output_root,
        colabfold_output_root=request.request_manifest_path.parent / "colabfold_outputs",
        runtime_version="colabfold-test",
    )
    monkeypatch.chdir(tmp_path)

    assert (
        validate_foldcheck_report_content(
            result.foldcheck_report_path,
            repo_root=repo_root,
            output_root=output_root,
        )
        == []
    )


def _report_row(sequence: dict[str, object]) -> dict[str, object]:
    candidate_id = str(sequence["sequence_id"])
    return {
        "candidate_id": candidate_id,
        "runtime_kind": "alphafold_family_colabfold",
        "runtime_version": "test",
        "input_sequence_hash": sequence["sequence_hash"],
        "reference_structure_id": "ec86kit_7v9u_protomer1",
        "wt_baseline_artifact_id": "self" if candidate_id == "wild_type" else "wild_type",
        "runtime_parameters_hash": "sha256:" + "1" * 64,
        "threshold_id": "test",
        "threshold_values": {"test": True},
        "plddt": 90.0,
        "pae_summary": {"status": "not_found"},
        "backbone_rmsd_to_reference": 0.0,
        "protected_contact_retention": None,
        "status": "accepted",
        "rejection_reason": "",
        "missing_metric_reason": "",
    }
