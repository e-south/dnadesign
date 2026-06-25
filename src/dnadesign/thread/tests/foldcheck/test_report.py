"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/foldcheck/test_report.py

Fold-check report tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.thread.foldcheck import validate_foldcheck_report, write_foldcheck_report

_REQUEST_HASH = "sha256:" + "1" * 64


def test_foldcheck_report_validator_requires_wt_baseline(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    write_foldcheck_report(report, [_accepted_row("candidate_a")], request_hash=_REQUEST_HASH)

    issues = validate_foldcheck_report(
        report,
        request_hash=_REQUEST_HASH,
        expected_candidate_ids={"wild_type", "candidate_a"},
    )

    assert "thread.foldcheck_report.missing_wt_baseline" in {issue.check_id for issue in issues}
    assert "thread.foldcheck_report.missing_candidates" in {issue.check_id for issue in issues}


def test_foldcheck_report_validator_accepts_complete_rows(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    rows = [_accepted_row("wild_type"), _accepted_row("candidate_a")]
    write_foldcheck_report(report, rows, request_hash=_REQUEST_HASH)

    issues = validate_foldcheck_report(
        report,
        request_hash=_REQUEST_HASH,
        expected_candidate_ids={"wild_type", "candidate_a"},
    )

    assert issues == []


def test_foldcheck_report_validator_rejects_accepted_rows_without_runtime_hash(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    row = _accepted_row("wild_type")
    row["runtime_parameters_hash"] = ""
    write_foldcheck_report(report, [row], request_hash=_REQUEST_HASH)

    issues = validate_foldcheck_report(report, request_hash=_REQUEST_HASH)

    assert [issue.check_id for issue in issues] == ["thread.foldcheck_report.accepted_missing_runtime_hash"]


def _accepted_row(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "runtime_kind": "alphafold_family_colabfold",
        "runtime_version": "planned-test",
        "input_sequence_hash": "sha256:" + "2" * 64,
        "reference_structure_id": "ec86kit_7v9u_protomer1",
        "wt_baseline_artifact_id": "wild_type" if candidate_id != "wild_type" else "self",
        "runtime_parameters_hash": "sha256:" + "3" * 64,
        "threshold_id": "eco1_rt_foldcheck_thresholds_v1",
        "threshold_values": {"min_plddt": 70.0},
        "plddt": 80.0,
        "pae_summary": {"mean": 4.0},
        "backbone_rmsd_to_reference": 1.2,
        "protected_contact_retention": True,
        "status": "accepted",
        "rejection_reason": "",
        "missing_metric_reason": "",
    }
