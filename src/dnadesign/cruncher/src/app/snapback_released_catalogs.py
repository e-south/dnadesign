"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_released_catalogs.py

Shared catalog resolution helpers for released-product Snapback workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.nickases.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.nickases.models import NickaseCatalog
from dnadesign.cruncher.release_enzymes.catalog import (
    dump_release_enzyme_catalog_yaml,
    load_merged_release_enzyme_catalog,
)
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.released_models import ReleaseCatalogSources


@dataclass(frozen=True)
class ReleasedCatalogSourcesSummary:
    nick_catalog_source: str
    release_catalog_source: str


@dataclass(frozen=True)
class ReleasedResolvedCatalogs:
    nick_catalog: NickaseCatalog
    release_catalog: ReleaseEnzymeCatalog
    nick_catalog_source: str
    release_catalog_source: str
    nick_catalog_yaml: str
    release_catalog_yaml: str


def released_catalog_sources_summary(
    *,
    nick_sources: CatalogSources,
    release_sources: ReleaseCatalogSources,
) -> ReleasedCatalogSourcesSummary:
    return ReleasedCatalogSourcesSummary(
        nick_catalog_source=catalog_source_label(
            preset_ids=nick_sources.resolved_preset_ids(),
            resolved_paths=nick_sources.additional_paths,
        ),
        release_catalog_source=catalog_source_label(
            preset_ids=release_sources.resolved_preset_ids(),
            resolved_paths=release_sources.additional_paths,
        ),
    )


def resolve_released_catalogs(
    *,
    nick_sources: CatalogSources,
    release_sources: ReleaseCatalogSources,
    workspace_root: Path,
) -> ReleasedResolvedCatalogs:
    nick_catalog, nick_resolved_paths = load_merged_nickase_catalog(
        preset_id=nick_sources.preset,
        additional_preset_ids=nick_sources.additional_presets,
        additional_paths=nick_sources.additional_paths,
        workspace_root=workspace_root,
    )
    release_catalog, release_resolved_paths = load_merged_release_enzyme_catalog(
        preset_id=release_sources.preset,
        additional_preset_ids=release_sources.additional_presets,
        additional_paths=release_sources.additional_paths,
        workspace_root=workspace_root,
    )
    return ReleasedResolvedCatalogs(
        nick_catalog=nick_catalog,
        release_catalog=release_catalog,
        nick_catalog_source=catalog_source_label(
            preset_ids=nick_sources.resolved_preset_ids(),
            resolved_paths=nick_resolved_paths,
        ),
        release_catalog_source=catalog_source_label(
            preset_ids=release_sources.resolved_preset_ids(),
            resolved_paths=release_resolved_paths,
        ),
        nick_catalog_yaml=dump_nickase_catalog_yaml(nick_catalog),
        release_catalog_yaml=dump_release_enzyme_catalog_yaml(release_catalog),
    )


__all__ = [
    "ReleasedCatalogSourcesSummary",
    "ReleasedResolvedCatalogs",
    "released_catalog_sources_summary",
    "resolve_released_catalogs",
]
