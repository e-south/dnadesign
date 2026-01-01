"""Adapter protocol for motif and site sources."""

from __future__ import annotations

from typing import Iterable, Protocol, Set

from dnadesign.cruncher.ingest.models import (
    DatasetDescriptor,
    DatasetQuery,
    MotifDescriptor,
    MotifQuery,
    MotifRecord,
    SiteInstance,
    SiteQuery,
)


class SourceAdapter(Protocol):
    source_id: str

    def capabilities(self) -> Set[str]: ...

    def list_motifs(self, query: MotifQuery) -> list[MotifDescriptor]: ...

    def get_motif(self, motif_id: str) -> MotifRecord: ...

    def list_sites(self, query: SiteQuery) -> Iterable[SiteInstance]: ...

    def get_sites_for_motif(self, motif_id: str, query: SiteQuery) -> Iterable[SiteInstance]: ...

    def list_datasets(self, query: DatasetQuery) -> list[DatasetDescriptor]: ...
