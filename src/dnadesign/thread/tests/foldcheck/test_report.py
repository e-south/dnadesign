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

import pyarrow.parquet as pq

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


def test_foldcheck_report_validator_requires_accepted_wt_baseline(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    row = {
        **_accepted_row("wild_type"),
        "status": "errored",
        "plddt": None,
        "backbone_rmsd_to_reference": None,
        "rejection_reason": "colabfold_output_missing",
        "missing_metric_reason": "colabfold_output_missing",
    }
    write_foldcheck_report(report, [row], request_hash=_REQUEST_HASH)

    issues = validate_foldcheck_report(report, request_hash=_REQUEST_HASH)

    assert [issue.check_id for issue in issues] == ["thread.foldcheck_report.wt_baseline_not_accepted"]


def test_foldcheck_report_validator_rejects_unexpected_candidates(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    rows = [_accepted_row("wild_type"), _accepted_row("candidate_a"), _accepted_row("foreign_candidate")]
    write_foldcheck_report(report, rows, request_hash=_REQUEST_HASH)

    issues = validate_foldcheck_report(
        report,
        request_hash=_REQUEST_HASH,
        expected_candidate_ids={"wild_type", "candidate_a"},
    )

    assert [issue.check_id for issue in issues] == ["thread.foldcheck_report.unexpected_candidates"]


def test_foldcheck_report_validator_requires_selected_candidate_acceptance(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    rows = [
        _accepted_row("wild_type"),
        {
            **_accepted_row("candidate_a"),
            "status": "errored",
            "plddt": None,
            "backbone_rmsd_to_reference": None,
            "rejection_reason": "colabfold_output_missing",
            "missing_metric_reason": "colabfold_output_missing",
        },
    ]
    write_foldcheck_report(report, rows, request_hash=_REQUEST_HASH)

    issues = validate_foldcheck_report(
        report,
        request_hash=_REQUEST_HASH,
        expected_candidate_ids={"wild_type", "candidate_a"},
        required_accepted_candidate_ids={"candidate_a"},
    )

    assert [issue.check_id for issue in issues] == ["thread.foldcheck_report.selected_candidates_not_accepted"]


def test_foldcheck_report_validator_treats_missing_schema_version_as_v1(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    write_foldcheck_report(report, [_accepted_row("wild_type")], request_hash=_REQUEST_HASH)
    table = pq.read_table(report)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"schema_version")
    pq.write_table(table.replace_schema_metadata(metadata), report)

    assert validate_foldcheck_report(report, request_hash=_REQUEST_HASH) == []


def test_foldcheck_report_validator_rejects_unknown_schema_version(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    write_foldcheck_report(report, [_accepted_row("wild_type")], request_hash=_REQUEST_HASH)
    table = pq.read_table(report)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_version"] = b"3"
    pq.write_table(table.replace_schema_metadata(metadata), report)

    issues = validate_foldcheck_report(report, request_hash=_REQUEST_HASH)

    assert [issue.check_id for issue in issues] == ["thread.foldcheck_report.unsupported_schema_version"]


def test_foldcheck_report_validator_rejects_v2_columns_without_v2_metadata(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    write_foldcheck_report(report, [_v2_accepted_row("wild_type")], request_hash=_REQUEST_HASH)
    table = pq.read_table(report)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"schema_version")
    pq.write_table(table.replace_schema_metadata(metadata), report)

    issues = validate_foldcheck_report(report, request_hash=_REQUEST_HASH)

    assert [issue.check_id for issue in issues] == ["thread.foldcheck_report.schema_version_column_mismatch"]


def test_foldcheck_report_validator_rejects_mixed_v2_reference_lineage(tmp_path: Path) -> None:
    report = tmp_path / "foldcheck_report.parquet"
    wt = _v2_accepted_row("wild_type")
    candidate = _v2_accepted_row("candidate_a")
    candidate["reference_structure_hash"] = "sha256:" + "9" * 64
    write_foldcheck_report(report, [wt, candidate], request_hash=_REQUEST_HASH)

    issues = validate_foldcheck_report(report, request_hash=_REQUEST_HASH)

    assert "thread.foldcheck_report.inconsistent_reference_lineage" in {issue.check_id for issue in issues}


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


def _v2_accepted_row(candidate_id: str) -> dict[str, object]:
    return {
        **_accepted_row(candidate_id),
        "backbone_rmsd_to_wt_baseline": 0.0 if candidate_id == "wild_type" else 0.8,
        "reference_structure_hash": "sha256:" + "4" * 64,
        "reference_mobile_positions_hash": "sha256:" + "5" * 64,
        "reference_coordinate_basis": "reference_ca_order_to_one_based_mobile_sequence_position",
    }
