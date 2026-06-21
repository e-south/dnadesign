"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/providers.py

Provider-source helpers for Eco1 conservation roster-cache materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.providers import (
    ProviderCache,
    load_provider_caches,
)


def load_provider_source_records(provider_source_root: Path, provider_ids: Sequence[str]) -> dict[str, ProviderCache]:
    """Load explicit provider FASTA sources before writing filtered caches."""

    return load_provider_caches(provider_source_root, provider_ids)


def write_filtered_provider_caches(
    *,
    provider_caches: Mapping[str, ProviderCache],
    provider_accessions: Mapping[str, Sequence[str]],
    cache_root: Path,
) -> dict[str, Path]:
    """Write declared provider caches containing only included source records."""

    provider_cache_root = cache_root / "provider_caches"
    provider_cache_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for provider_id, cache in provider_caches.items():
        records = {accession: cache.records[accession] for accession in provider_accessions.get(provider_id, [])}
        path = provider_cache_root / f"{provider_id}.fasta"
        _write_fasta(path, records)
        paths[provider_id] = path
    return paths


def _write_fasta(path: Path, records: Mapping[str, str]) -> None:
    if not records:
        raise ValueError(f"provider cache would be empty: {path}")
    lines: list[str] = []
    for record_id, sequence in records.items():
        lines.extend([f">{record_id}", sequence.upper()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
