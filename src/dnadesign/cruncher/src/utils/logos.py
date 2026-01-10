"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/logos.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def site_kind_label(kind: str | None) -> str:
    if not kind:
        return "unknown"
    lower = kind.lower()
    if lower == "curated":
        return "curated"
    if lower.startswith("ht_"):
        detail = lower[len("ht_") :].replace("_", " ")
        return f"high-throughput ({detail})"
    if lower == "mixed":
        return "combined"
    return kind


def site_entries_for_logo(
    *,
    catalog: CatalogIndex,
    entry: CatalogEntry,
    combine_sites: bool,
    site_kinds: list[str] | None,
) -> list[CatalogEntry]:
    entries = [entry]
    if combine_sites:
        entries = [
            candidate
            for candidate in catalog.entries.values()
            if candidate.tf_name.lower() == entry.tf_name.lower() and candidate.has_sites
        ]
    if site_kinds is not None:
        entries = [candidate for candidate in entries if candidate.site_kind in site_kinds]
    return entries


def logo_subtitle(
    *,
    pwm_source: str,
    entry: CatalogEntry,
    catalog: CatalogIndex,
    combine_sites: bool,
    site_kinds: list[str] | None,
) -> str:
    if pwm_source == "matrix":
        if entry.matrix_source:
            return f"matrix: {entry.matrix_source}"
        return "matrix"
    if pwm_source == "sites":
        entries = site_entries_for_logo(
            catalog=catalog,
            entry=entry,
            combine_sites=combine_sites,
            site_kinds=site_kinds,
        )
        kinds = [candidate.site_kind for candidate in entries if candidate.site_kind]
        if len(kinds) > 1 and "mixed" in kinds:
            kinds = [kind for kind in kinds if kind != "mixed"]
        labels = [site_kind_label(kind) for kind in sorted(set(kinds))]
        if not labels:
            labels = ["unknown"]
        base = " + ".join(labels)
        if combine_sites and (len(entries) > 1 or len(labels) > 1):
            return "combined" if base == "combined" else f"combined ({base})"
        return base
    return "unknown"
