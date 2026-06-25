"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/foldcheck/report.py

Eco1 fold-check report validator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import _load_yaml
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.constants import (
    WT_SEQUENCE_ID,
)
from dnadesign.thread.foldcheck import validate_foldcheck_report


def validate_foldcheck_report_content(path: Path, *, output_root: Path) -> list[ContractIssue]:
    """Validate the Eco1 fold-check report against the current request and candidate table."""

    request_manifest_path = output_root / "foldcheck_request/foldcheck_request_manifest.yaml"
    if not request_manifest_path.exists():
        return [
            ContractIssue(
                check_id="eco1_rt.foldcheck_report.request_manifest_missing",
                message="fold-check report validation requires the current fold-check request manifest",
                path=str(request_manifest_path),
            )
        ]
    manifest = _load_yaml(request_manifest_path)
    expected_candidate_ids = _expected_candidate_ids(output_root / "candidate_table.parquet")
    issues = [
        ContractIssue(check_id=issue.check_id, message=issue.message, path=issue.path)
        for issue in validate_foldcheck_report(
            path,
            request_hash=str(manifest.get("request_hash", "")),
            expected_candidate_ids=expected_candidate_ids,
            wt_candidate_id=WT_SEQUENCE_ID,
        )
    ]
    issues.extend(_validate_report_sequence_hashes(path, manifest))
    return issues


def _expected_candidate_ids(candidate_table_path: Path) -> set[str]:
    expected = {WT_SEQUENCE_ID}
    if not candidate_table_path.exists():
        return expected
    for row in pq.read_table(candidate_table_path).to_pylist():
        if str(row.get("status")) == "accepted":
            expected.add(str(row["candidate_id"]))
    return expected


def _validate_report_sequence_hashes(path: Path, manifest: Mapping[str, object]) -> list[ContractIssue]:
    expected_hashes = _manifest_sequence_hashes(manifest)
    issues: list[ContractIssue] = []
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        candidate_id = str(row.get("candidate_id", ""))
        expected_hash = expected_hashes.get(candidate_id)
        if expected_hash is None:
            continue
        if str(row.get("input_sequence_hash", "")) != expected_hash:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.foldcheck_report.sequence_hash_mismatch",
                    message=f"fold-check row {candidate_id!r} does not match the current request sequence hash",
                    path=f"{path}:row[{index}]",
                )
            )
    return issues


def _manifest_sequence_hashes(manifest: Mapping[str, object]) -> dict[str, str]:
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list):
        return {}
    return {
        str(row["sequence_id"]): str(row["sequence_hash"])
        for row in sequences
        if isinstance(row, dict) and "sequence_id" in row and "sequence_hash" in row
    }
