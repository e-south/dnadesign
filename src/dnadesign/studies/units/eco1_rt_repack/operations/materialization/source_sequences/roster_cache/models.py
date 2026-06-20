"""Data models for Eco1 conservation roster-cache materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RosterRow:
    """One normalized row from the Mestre roster table."""

    row_index: int
    node_id: str
    accession: str
    retron_subtype: str
    cluster_domain: str
    rt_clade: str
    status: str
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class SourceRecord:
    """One source_records.yaml row."""

    profile_id: str
    record_id: str
    provider_id: str
    accession: str
    status: str
    exclusion_reason: str | None = None

    def to_yaml_row(self) -> dict[str, str]:
        """Serialize to the source_records.yaml row contract."""

        row = {
            "profile_id": self.profile_id,
            "record_id": self.record_id,
            "provider_id": self.provider_id,
            "accession": self.accession,
            "status": self.status,
        }
        if self.exclusion_reason:
            row["exclusion_reason"] = self.exclusion_reason
        return row


@dataclass(frozen=True)
class MaterializedConservationRosterCache:
    """Paths emitted by one roster-cache materialization pass."""

    cache_root: Path
    source_records_path: Path
    provider_cache_paths: dict[str, Path]
    manifest_path: Path
