"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/issues.py

Issue builders for Eco1 source-sequence bundle validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue


def append_field_mismatch(
    issues: list[ContractIssue],
    *,
    payload: Mapping[str, Any],
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.manifest_field_mismatch",
                    message=f"source-sequence manifest field {field!r} must equal {expected_value!r}",
                    path=f"{path}:{field}",
                )
            )


def invalid_manifest_record_issue(profile_id: str, manifest_path: Path) -> ContractIssue:
    return ContractIssue(
        check_id="eco1_rt.source_sequences.invalid_manifest_record",
        message=f"{profile_id} manifest records must be mappings",
        path=str(manifest_path),
    )
