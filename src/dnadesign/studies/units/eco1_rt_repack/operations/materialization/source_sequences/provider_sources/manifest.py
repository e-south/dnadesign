"""Manifest writer for Eco1 provider-source FASTA acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


def write_provider_source_manifest(
    path: Path,
    *,
    roster_table: Path,
    roster_table_sha256: str,
    conservation_sources_sha256: str,
    provider_source_hashes: Mapping[str, str],
    provider_record_counts: Mapping[str, int],
    provider_requested_counts: Mapping[str, int],
    provider_missing_counts: Mapping[str, int],
    failure_ledger_path: Path | None,
    created_at: str,
) -> None:
    """Write a manifest for raw provider-source FASTA files."""

    payload = {
        "schema_id": "eco1_rt_repack.conservation_provider_sources.index",
        "schema_version": 1,
        "version": 1,
        "study_id": "eco1_rt_repack",
        "status": "materialized",
        "created_at": created_at,
        "roster_table": str(roster_table),
        "upstream_hashes": {
            "conservation_sources_yaml": conservation_sources_sha256,
            "roster_table": roster_table_sha256,
            **dict(provider_source_hashes),
        },
        "provider_record_counts": dict(provider_record_counts),
        "provider_requested_counts": dict(provider_requested_counts),
        "provider_missing_counts": dict(provider_missing_counts),
        "failure_ledger_path": str(failure_ledger_path) if failure_ledger_path else None,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_provider_failure_ledger(
    path: Path,
    *,
    failures: Mapping[str, list[str]],
    created_at: str,
) -> None:
    """Write explicit unresolved provider accession failures."""

    rows: list[dict[str, str]] = []
    for provider_id, accessions in failures.items():
        rows.extend(
            {
                "provider_id": provider_id,
                "accession": accession,
                "status": "excluded",
                "exclusion_reason": "provider_unresolved_in_declared_source",
            }
            for accession in accessions
        )
    payload = {
        "schema_id": "eco1_rt_repack.conservation_provider_sources.failure_ledger",
        "schema_version": 1,
        "version": 1,
        "study_id": "eco1_rt_repack",
        "status": "materialized",
        "created_at": created_at,
        "failures": rows,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
