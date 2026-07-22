"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency/fasta.py

Source FASTA content checks for Eco1 source-sequence sufficiency.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import (
    load_fasta_records_ordered,
    resolve_path,
    sha256_file,
)


def validate_source_fasta(
    issues: list[ContractIssue],
    *,
    profile_manifest: Mapping[str, Any],
    manifest_path: Path,
    profile_id: str,
    target_row_id: str,
    target_sequence_hash: str,
    included_count: int,
    root: Path,
) -> None:
    """Verify source FASTA hash, target-row identity, and row count."""

    fasta_path = resolve_path(root, Path(str(profile_manifest.get("fasta_path", ""))))
    if not fasta_path.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_fasta_missing",
                message=f"{profile_id} source FASTA is missing",
                path=str(fasta_path),
            )
        )
        return
    if profile_manifest.get("fasta_sha256") != "sha256:" + sha256_file(fasta_path):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_fasta_hash_mismatch",
                message=f"{profile_id} source FASTA hash must match its profile manifest",
                path=str(manifest_path),
            )
        )
    records = load_fasta_records_ordered(fasta_path)
    if not records or records[0][0] != target_row_id:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.target_row_not_first",
                message=f"{profile_id} source FASTA must start with the ec86kit target row",
                path=str(fasta_path),
            )
        )
    elif "sha256:" + hashlib.sha256(records[0][1].encode("utf-8")).hexdigest() != target_sequence_hash:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.target_sequence_hash_mismatch",
                message=f"{profile_id} target FASTA row must match the ec86kit sequence hash",
                path=str(fasta_path),
            )
        )
    if len(records) != included_count + 1:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_fasta_record_count_mismatch",
                message=f"{profile_id} source FASTA must contain target plus included records only",
                path=str(fasta_path),
            )
        )
