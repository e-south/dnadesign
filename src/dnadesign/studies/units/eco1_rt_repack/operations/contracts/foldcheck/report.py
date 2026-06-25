"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/foldcheck/report.py

Eco1 fold-check report validator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
    return issues


def _expected_candidate_ids(candidate_table_path: Path) -> set[str]:
    expected = {WT_SEQUENCE_ID}
    if not candidate_table_path.exists():
        return expected
    for row in pq.read_table(candidate_table_path).to_pylist():
        if str(row.get("status")) == "accepted":
            expected.add(str(row["candidate_id"]))
    return expected
