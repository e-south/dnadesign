"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/logos.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _format_site_counts(site_count: int | None, site_total: int | None) -> str | None:
    if site_count is None or site_count <= 0:
        return None
    if site_total and site_total >= site_count:
        return f"n={site_count}/{site_total}"
    return f"n={site_count}"


def _matrix_site_count(entry: CatalogEntry) -> int | None:
    tags = entry.tags or {}
    for key in ("discovery_nsites", "meme_nsites", "site_count", "nsites"):
        parsed = _parse_int(tags.get(key))
        if parsed is not None:
            return parsed
    return None


def _summarize_items(values: list[str], *, limit: int = 2) -> str:
    cleaned = [value for value in values if value]
    unique = sorted(set(cleaned))
    if not unique:
        return "unknown"
    if len(unique) <= limit:
        return "+".join(unique)
    return "+".join(unique[:limit]) + f"+{len(unique) - limit}"


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


def _site_entry_summary(
    *,
    catalog: CatalogIndex,
    entry: CatalogEntry,
    combine_sites: bool,
    site_kinds: list[str] | None,
) -> dict[str, object]:
    entries = site_entries_for_logo(
        catalog=catalog,
        entry=entry,
        combine_sites=combine_sites,
        site_kinds=site_kinds,
    )
    site_count = sum(candidate.site_count for candidate in entries if candidate.site_count)
    site_total = sum(candidate.site_total for candidate in entries if candidate.site_total)
    if site_count == 0:
        site_count = None
    if site_total == 0:
        site_total = None
    kinds = [candidate.site_kind for candidate in entries if candidate.site_kind]
    if len(kinds) > 1 and "mixed" in kinds:
        kinds = [kind for kind in kinds if kind != "mixed"]
    labels = [site_kind_label(kind) for kind in sorted(set(kinds))]
    if not labels:
        labels = ["unknown"]
    return {
        "entries": entries,
        "set_count": len(entries),
        "site_count": site_count,
        "site_total": site_total,
        "sources": _summarize_items([candidate.source for candidate in entries]),
        "kinds": _summarize_items(labels, limit=3),
    }


def pwm_provenance_summary(
    *,
    pwm_source: str,
    entry: CatalogEntry,
    catalog: CatalogIndex,
    combine_sites: bool,
    site_kinds: list[str] | None,
) -> str:
    if pwm_source == "matrix":
        matrix_source = entry.matrix_source or "matrix"
        count_label = _format_site_counts(_matrix_site_count(entry), None)
        if count_label:
            return f"matrix ({matrix_source}, {count_label})"
        return f"matrix ({matrix_source})"
    if pwm_source == "sites":
        summary = _site_entry_summary(
            catalog=catalog,
            entry=entry,
            combine_sites=combine_sites,
            site_kinds=site_kinds,
        )
        count_label = _format_site_counts(summary["site_count"], summary["site_total"])
        mode = "combined" if combine_sites else "single"
        parts = [f"sites {mode} n_sets={summary['set_count']}"]
        if count_label:
            parts.append(count_label)
        parts.append(f"sources={summary['sources']}")
        parts.append(f"kinds={summary['kinds']}")
        return " ".join(parts)
    return "unknown"


def logo_subtitle(
    *,
    pwm_source: str,
    entry: CatalogEntry,
    catalog: CatalogIndex,
    combine_sites: bool,
    site_kinds: list[str] | None,
) -> str:
    if pwm_source == "matrix":
        matrix_source = entry.matrix_source or "matrix"
        count_label = _format_site_counts(_matrix_site_count(entry), None)
        if count_label:
            return f"{entry.source} ({matrix_source}, {count_label})"
        return f"{entry.source} ({matrix_source})"
    if pwm_source == "sites":
        summary = _site_entry_summary(
            catalog=catalog,
            entry=entry,
            combine_sites=combine_sites,
            site_kinds=site_kinds,
        )
        sources = summary["sources"]
        kinds = summary["kinds"]
        count_label = _format_site_counts(summary["site_count"], summary["site_total"])
        if combine_sites:
            details = [count_label, f"sets={summary['set_count']}", sources, kinds]
            detail_text = ", ".join([item for item in details if item])
            return f"combined ({detail_text})"
        if count_label:
            return f"{sources} ({kinds}, {count_label})"
        return f"{sources} ({kinds})"
    return "unknown"
