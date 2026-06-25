"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/candidate_table.py

Candidate-table validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.thread.candidates.proteinmpnn import validate_candidate_table

_THREAD_TO_ECO1_ISSUES = {
    "thread.candidate_table.missing_columns": "eco1_rt.sampling.candidate_table_missing_columns",
    "thread.candidate_table.metadata_mismatch": "eco1_rt.sampling.candidate_table_metadata_mismatch",
    "thread.candidate_table.count_mismatch": "eco1_rt.sampling.candidate_table_count_mismatch",
    "thread.candidate_table.request_hash_mismatch": "eco1_rt.sampling.candidate_table_request_hash_mismatch",
    "thread.candidate_table.protected_mutation": "eco1_rt.sampling.candidate_table_protected_mutation",
    "thread.candidate_table.outside_mutable_position": "eco1_rt.sampling.candidate_table_outside_mutable_position",
}


def validate_candidate_table_content(path: Path, *, output_root: Path) -> list[ContractIssue]:
    """Validate the Eco1 candidate table against the request and sample table."""

    request_manifest_path = output_root / "proteinmpnn_request/request_manifest.yaml"
    sample_table_path = output_root / "sample_table.parquet"
    request_manifest = _load_yaml(request_manifest_path)
    return [
        _adapt_thread_issue(issue)
        for issue in validate_candidate_table(
            path,
            request_hash=str(request_manifest["request_hash"]),
            sample_table_path=sample_table_path,
        )
    ]


def _adapt_thread_issue(issue: Any) -> ContractIssue:
    return ContractIssue(
        check_id=_THREAD_TO_ECO1_ISSUES.get(issue.check_id, "eco1_rt.sampling.candidate_table_invalid"),
        message=issue.message,
        path=issue.path,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded
