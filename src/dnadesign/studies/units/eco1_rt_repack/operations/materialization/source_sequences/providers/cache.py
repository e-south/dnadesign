"""Local provider FASTA cache loading for Eco1 conservation source sequences."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")


@dataclass(frozen=True)
class ProviderCache:
    """One declared provider FASTA cache."""

    provider_id: str
    path: Path
    sha256: str
    records: dict[str, str]


def load_provider_caches(provider_root: Path, provider_ids: Sequence[str]) -> dict[str, ProviderCache]:
    """Load provider FASTA caches named by declared provider id."""

    caches: dict[str, ProviderCache] = {}
    for provider_id in provider_ids:
        path = provider_root / f"{provider_id}.fasta"
        if not path.exists():
            raise FileNotFoundError(path)
        caches[provider_id] = ProviderCache(
            provider_id=provider_id,
            path=path,
            sha256="sha256:" + _sha256(path),
            records=_load_fasta_records(path),
        )
    return caches


def _load_fasta_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    current_chunks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records[current_id] = _validated_sequence("".join(current_chunks), current_id)
            current_id = line[1:].split()[0]
            if not current_id:
                raise ValueError(f"FASTA record id is empty in {path}")
            if current_id in records:
                raise ValueError(f"duplicate FASTA record id {current_id!r} in {path}")
            current_chunks = []
        elif current_id is None:
            raise ValueError(f"FASTA sequence data appears before a record id in {path}")
        else:
            current_chunks.append(line)
    if current_id is not None:
        records[current_id] = _validated_sequence("".join(current_chunks), current_id)
    if not records:
        raise ValueError(f"FASTA is empty: {path}")
    return records


def _validated_sequence(sequence: str, record_id: str) -> str:
    normalized = sequence.upper()
    if not normalized:
        raise ValueError(f"FASTA record {record_id!r} has an empty sequence")
    invalid = sorted({character for character in normalized if character not in _PROTEIN_ALPHABET})
    if invalid:
        raise ValueError(f"Invalid protein character {invalid[0]!r} in FASTA record {record_id!r}")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
