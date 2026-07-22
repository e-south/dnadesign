"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency/cache_hashes.py

Source-cache presence and hash checks for Eco1 source FASTA bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import sha256_file


def validate_source_cache_presence(issues: list[ContractIssue], *, source_cache_root: Path) -> None:
    """Record missing cache-root and source-record ledger failures."""

    if not source_cache_root.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_cache_root_missing",
                message="source-sequence sufficiency requires the real provider source cache root",
                path=str(source_cache_root),
            )
        )
    source_records_path = source_cache_root / "source_records.yaml"
    if not source_records_path.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_records_missing",
                message="source-sequence sufficiency requires source_records.yaml",
                path=str(source_records_path),
            )
        )


def validate_cache_hashes(
    issues: list[ContractIssue],
    *,
    upstream_hashes: Mapping[str, Any],
    source_records_path: Path,
    provider_cache_root: Path,
    provider_ids: Sequence[str],
    path: Path,
) -> None:
    """Verify manifest hashes against source_records.yaml and provider caches."""

    if source_records_path.exists() and upstream_hashes.get("source_records_yaml") != "sha256:" + sha256_file(
        source_records_path
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_records_hash_mismatch",
                message="source_records.yaml hash must match source-sequence manifests",
                path=str(path),
            )
        )
    for provider_id in provider_ids:
        provider_path = provider_cache_root / f"{provider_id}.fasta"
        provider_key = f"provider_cache_{provider_id}"
        if not provider_path.exists():
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.provider_cache_missing",
                    message=f"provider cache {provider_id!r} is missing",
                    path=str(provider_path),
                )
            )
            continue
        if upstream_hashes.get(provider_key) != "sha256:" + sha256_file(provider_path):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.provider_cache_hash_mismatch",
                    message=f"provider cache hash for {provider_id!r} must match source-sequence manifests",
                    path=str(path),
                )
            )
