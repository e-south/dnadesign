"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_catalogs.py

Shared nickase catalog resolution helpers for preserved-site Snapback workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.nickases.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.nickases.models import NickaseCatalog
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.models import CatalogSources


@dataclass(frozen=True)
class ResolvedSnapbackCatalog:
    catalog: NickaseCatalog
    resolved_paths: tuple[Path, ...]
    catalog_source: str
    catalog_yaml: str


def unresolved_snapback_catalog_source(*, sources: CatalogSources) -> str:
    return catalog_source_label(
        preset_ids=sources.resolved_preset_ids(),
        resolved_paths=sources.additional_paths,
    )


def resolve_snapback_catalog(*, sources: CatalogSources, workspace_root: Path) -> ResolvedSnapbackCatalog:
    catalog, resolved_paths = load_merged_nickase_catalog(
        preset_id=sources.preset,
        additional_preset_ids=sources.additional_presets,
        additional_paths=sources.additional_paths,
        workspace_root=workspace_root,
    )
    return ResolvedSnapbackCatalog(
        catalog=catalog,
        resolved_paths=tuple(resolved_paths),
        catalog_source=catalog_source_label(
            preset_ids=sources.resolved_preset_ids(),
            resolved_paths=resolved_paths,
        ),
        catalog_yaml=dump_nickase_catalog_yaml(catalog),
    )


__all__ = [
    "ResolvedSnapbackCatalog",
    "resolve_snapback_catalog",
    "unresolved_snapback_catalog_source",
]
