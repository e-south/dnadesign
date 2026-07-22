"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/ncbi.py

NCBI Protein EFetch provider for Eco1 conservation source sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from urllib.parse import urlencode

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources.fasta import (
    parse_provider_fasta,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources.http import (
    fetch_text,
)

_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_ncbi_protein_fastas(
    accessions: Sequence[str],
    *,
    batch_size: int = 100,
    sleep_seconds: float = 0.34,
    base_url: str = _EFETCH_URL,
) -> dict[str, str]:
    """Fetch NCBI Protein FASTA records for explicit accessions."""

    records: dict[str, str] = {}
    for batch in _batches(tuple(accessions), batch_size=batch_size):
        query = urlencode({"db": "protein", "id": ",".join(batch), "rettype": "fasta", "retmode": "text"})
        text = fetch_text(f"{base_url}?{query}")
        records.update(parse_provider_fasta(text, requested_accessions=batch))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return records


def _batches(accessions: Sequence[str], *, batch_size: int) -> tuple[tuple[str, ...], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(tuple(accessions[index : index + batch_size]) for index in range(0, len(accessions), batch_size))
