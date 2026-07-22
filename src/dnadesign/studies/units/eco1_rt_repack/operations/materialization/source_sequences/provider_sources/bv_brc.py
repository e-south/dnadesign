"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/bv_brc.py

BV-BRC protein FASTA provider for Eco1 conservation source sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from urllib.parse import quote

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources.fasta import (
    parse_provider_fasta,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources.http import (
    fetch_text,
)

_GENOME_FEATURE_URL = "https://www.bv-brc.org/api/genome_feature/"


def fetch_bv_brc_feature_protein_fastas(
    accessions: Sequence[str],
    *,
    batch_size: int = 75,
    sleep_seconds: float = 0.1,
    base_url: str = _GENOME_FEATURE_URL,
) -> dict[str, str]:
    """Fetch BV-BRC protein FASTA records for explicit ``fig|`` feature ids."""

    records: dict[str, str] = {}
    for batch in _batches(tuple(accessions), batch_size=batch_size):
        query = ",".join(quote(accession, safe="") for accession in batch)
        text = fetch_text(
            f"{base_url}?in(patric_id,({query}))&limit({batch_size})",
            headers={"Accept": "application/protein+fasta"},
        )
        records.update(parse_provider_fasta(text, requested_accessions=batch))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return records


def _batches(accessions: Sequence[str], *, batch_size: int) -> tuple[tuple[str, ...], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(tuple(accessions[index : index + batch_size]) for index in range(0, len(accessions), batch_size))
