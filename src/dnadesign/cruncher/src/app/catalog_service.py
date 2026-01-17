"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/catalog_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def list_catalog(
    root: Path,
    *,
    tf_name: Optional[str] = None,
    source: Optional[str] = None,
    organism: Optional[str] = None,
    include_synonyms: bool = False,
) -> List[CatalogEntry]:
    catalog = CatalogIndex.load(root)
    org = {"name": organism} if organism else None
    return catalog.list(tf_name=tf_name, source=source, organism=org, include_synonyms=include_synonyms)


def search_catalog(
    root: Path,
    *,
    query: str,
    source: Optional[str] = None,
    organism: Optional[str] = None,
    regex: bool = False,
    case_sensitive: bool = False,
    fuzzy: bool = False,
    min_score: float = 0.6,
    limit: Optional[int] = None,
) -> List[CatalogEntry]:
    catalog = CatalogIndex.load(root)
    org = {"name": organism} if organism else None
    return catalog.search(
        query=query,
        source=source,
        organism=org,
        regex=regex,
        case_sensitive=case_sensitive,
        fuzzy=fuzzy,
        min_score=min_score,
        limit=limit,
    )


def verify_cache(root: Path) -> list[str]:
    catalog = CatalogIndex.load(root)
    issues: list[str] = []
    for entry in catalog.entries.values():
        if entry.has_matrix:
            motif_path = root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
            if not motif_path.exists():
                issues.append(f"missing motif file: {motif_path}")
        if entry.has_sites:
            sites_path = root / "normalized" / "sites" / entry.source / f"{entry.motif_id}.jsonl"
            if not sites_path.exists():
                issues.append(f"missing sites file: {sites_path}")
    return issues


def get_entry(root: Path, *, source: str, motif_id: str) -> Optional[CatalogEntry]:
    catalog = CatalogIndex.load(root)
    return catalog.entries.get(f"{source}:{motif_id}")


def catalog_stats(root: Path) -> dict[str, int]:
    catalog = CatalogIndex.load(root)
    motif_count = sum(1 for entry in catalog.entries.values() if entry.has_matrix)
    site_sets = sum(1 for entry in catalog.entries.values() if entry.has_sites)
    return {
        "entries": len(catalog.entries),
        "motifs": motif_count,
        "site_sets": site_sets,
    }
