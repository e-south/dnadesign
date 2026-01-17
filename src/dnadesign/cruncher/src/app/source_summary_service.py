"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/source_summary_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from dnadesign.cruncher.ingest.adapters.base import SourceAdapter
from dnadesign.cruncher.ingest.models import DatasetQuery, MotifDescriptor, MotifQuery
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def summarize_cache(root: Path, *, source: Optional[str] = None) -> dict:
    catalog = CatalogIndex.load(root)
    entries = list(catalog.entries.values())
    if source:
        entries = [entry for entry in entries if entry.source == source]
    totals_acc = _init_accumulator()
    sources_acc: dict[str, dict] = {}
    tf_acc: dict[str, dict] = {}
    for entry in entries:
        _accumulate_entry(entry, totals_acc)
        src_acc = sources_acc.setdefault(entry.source, _init_accumulator())
        _accumulate_entry(entry, src_acc)
        tf_key = entry.tf_name.lower()
        per_tf = tf_acc.setdefault(tf_key, _init_tf_acc(entry.tf_name))
        _accumulate_tf(entry, per_tf)
    totals = _finalize_accumulator(totals_acc)
    sources = {key: _finalize_accumulator(acc) for key, acc in sources_acc.items()}
    regulators = [_finalize_tf_acc(acc) for acc in tf_acc.values()]
    regulators.sort(key=lambda item: (-(item["sites_total"] or 0), item["tf_name"].lower()))
    return {"totals": totals, "sources": sources, "regulators": regulators}


def summarize_remote(
    adapter: SourceAdapter,
    *,
    limit: Optional[int] = None,
    page_size: int = 200,
    include_datasets: bool = True,
) -> dict:
    totals_acc = _init_remote_accumulator()
    tf_acc: dict[str, dict] = {}
    for desc in _iter_motifs(adapter, limit=limit, page_size=page_size):
        _accumulate_remote_motif(desc, totals_acc)
        tf_key = desc.tf_name.lower()
        per_tf = tf_acc.setdefault(tf_key, _init_tf_acc(desc.tf_name))
        per_tf["sources"].add(desc.source)
        per_tf["motifs"] += 1
    dataset_tf_map: dict[str, set[str]] = defaultdict(set)
    if include_datasets and "datasets:list" in adapter.capabilities():
        datasets = adapter.list_datasets(DatasetQuery())
        for ds in datasets:
            totals_acc["dataset_ids"].add(ds.dataset_id)
            if ds.dataset_source:
                totals_acc["dataset_sources"].add(ds.dataset_source)
            if ds.method:
                totals_acc["dataset_methods"].add(ds.method)
            for tf_name in ds.tf_names:
                dataset_tf_map[tf_name.lower()].add(ds.dataset_id)
        for tf_key, acc in tf_acc.items():
            acc["dataset_ids"] = dataset_tf_map.get(tf_key, set())
    totals = _finalize_remote_accumulator(totals_acc)
    regulators = [_finalize_tf_acc(acc, include_sites=False) for acc in tf_acc.values()]
    regulators.sort(key=lambda item: (-(item["motifs"] or 0), item["tf_name"].lower()))
    return {"totals": totals, "regulators": regulators}


def summarize_combined(
    *,
    cache_summary: Optional[dict] = None,
    remote_summaries: Optional[dict[str, dict]] = None,
) -> dict:
    combined: dict[str, dict] = {}

    def ensure_tf(tf_name: str) -> dict:
        key = tf_name.lower()
        if key not in combined:
            combined[key] = {
                "tf_name": tf_name,
                "cache": _init_combined_cache(),
                "remote": _init_combined_remote(),
            }
        return combined[key]

    if cache_summary:
        for reg in cache_summary.get("regulators", []):
            tf_name = reg.get("tf_name") or ""
            if not tf_name:
                continue
            entry = ensure_tf(tf_name)
            cache = entry["cache"]
            cache["sources"].update(reg.get("sources") or [])
            cache["motifs"] += int(reg.get("motifs") or 0)
            cache["site_sets"] += int(reg.get("site_sets") or 0)
            cache["sites_seq"] += int(reg.get("sites_seq") or 0)
            cache["sites_total"] += int(reg.get("sites_total") or 0)
            cache["datasets"] += int(reg.get("datasets") or 0)

    if remote_summaries:
        for source_id, summary in remote_summaries.items():
            for reg in summary.get("regulators", []):
                tf_name = reg.get("tf_name") or ""
                if not tf_name:
                    continue
                entry = ensure_tf(tf_name)
                remote = entry["remote"]
                sources = reg.get("sources") or [source_id]
                remote["sources"].update(sources)
                remote["motifs"] += int(reg.get("motifs") or 0)
                remote["datasets"] += int(reg.get("datasets") or 0)

    regulators = [_finalize_combined_tf(acc) for acc in combined.values()]
    regulators.sort(key=lambda item: item["tf_name"].lower())

    totals = {
        "tfs": len(regulators),
        "cache": cache_summary.get("totals") if cache_summary else None,
        "remote": _aggregate_remote_totals(remote_summaries) if remote_summaries else None,
    }
    return {"totals": totals, "regulators": regulators}


def _iter_motifs(
    adapter: SourceAdapter,
    *,
    limit: Optional[int],
    page_size: int,
) -> Iterable[MotifDescriptor]:
    caps = adapter.capabilities()
    if "motifs:iter" in caps and hasattr(adapter, "iter_motifs"):
        return adapter.iter_motifs(MotifQuery(limit=limit), page_size=page_size)
    if limit is None:
        raise ValueError(
            f"Source '{adapter.source_id}' does not support full inventory; provide a limit or add iter_motifs."
        )
    return adapter.list_motifs(MotifQuery(limit=limit))


def _init_accumulator() -> dict:
    return {
        "entries": 0,
        "tf_set": set(),
        "motifs": 0,
        "site_sets": 0,
        "sites_seq": 0,
        "sites_total": 0,
        "dataset_ids": set(),
        "dataset_sources": set(),
        "dataset_methods": set(),
    }


def _finalize_accumulator(acc: dict) -> dict:
    return {
        "entries": acc["entries"],
        "tfs": len(acc["tf_set"]),
        "motifs": acc["motifs"],
        "site_sets": acc["site_sets"],
        "sites_seq": acc["sites_seq"],
        "sites_total": acc["sites_total"],
        "datasets": len(acc["dataset_ids"]),
        "dataset_sources": sorted(acc["dataset_sources"]),
        "dataset_methods": sorted(acc["dataset_methods"]),
    }


def _accumulate_entry(entry: CatalogEntry, acc: dict) -> None:
    acc["entries"] += 1
    acc["tf_set"].add(entry.tf_name.lower())
    if entry.has_matrix:
        acc["motifs"] += 1
    if entry.has_sites:
        acc["site_sets"] += 1
        acc["sites_seq"] += int(entry.site_count or 0)
        acc["sites_total"] += int(entry.site_total or 0)
    if entry.dataset_id:
        acc["dataset_ids"].add(entry.dataset_id)
    if entry.dataset_source:
        acc["dataset_sources"].add(entry.dataset_source)
    if entry.dataset_method:
        acc["dataset_methods"].add(entry.dataset_method)


def _init_tf_acc(tf_name: str) -> dict:
    return {
        "tf_name": tf_name,
        "sources": set(),
        "motifs": 0,
        "site_sets": 0,
        "sites_seq": 0,
        "sites_total": 0,
        "dataset_ids": set(),
    }


def _accumulate_tf(entry: CatalogEntry, acc: dict) -> None:
    acc["sources"].add(entry.source)
    if entry.has_matrix:
        acc["motifs"] += 1
    if entry.has_sites:
        acc["site_sets"] += 1
        acc["sites_seq"] += int(entry.site_count or 0)
        acc["sites_total"] += int(entry.site_total or 0)
    if entry.dataset_id:
        acc["dataset_ids"].add(entry.dataset_id)


def _finalize_tf_acc(acc: dict, *, include_sites: bool = True) -> dict:
    payload = {
        "tf_name": acc["tf_name"],
        "sources": sorted(acc["sources"]),
        "motifs": acc["motifs"],
        "site_sets": acc["site_sets"] if include_sites else None,
        "sites_seq": acc["sites_seq"] if include_sites else None,
        "sites_total": acc["sites_total"] if include_sites else None,
        "datasets": len(acc["dataset_ids"]),
    }
    return payload


def _init_remote_accumulator() -> dict:
    return {
        "tf_set": set(),
        "motifs": 0,
        "dataset_ids": set(),
        "dataset_sources": set(),
        "dataset_methods": set(),
    }


def _accumulate_remote_motif(desc: MotifDescriptor, acc: dict) -> None:
    acc["motifs"] += 1
    acc["tf_set"].add(desc.tf_name.lower())


def _finalize_remote_accumulator(acc: dict) -> dict:
    return {
        "tfs": len(acc["tf_set"]),
        "motifs": acc["motifs"],
        "datasets": len(acc["dataset_ids"]) if acc["dataset_ids"] else 0,
        "dataset_sources": sorted(acc["dataset_sources"]),
        "dataset_methods": sorted(acc["dataset_methods"]),
    }


def _init_combined_cache() -> dict:
    return {
        "sources": set(),
        "motifs": 0,
        "site_sets": 0,
        "sites_seq": 0,
        "sites_total": 0,
        "datasets": 0,
    }


def _init_combined_remote() -> dict:
    return {
        "sources": set(),
        "motifs": 0,
        "datasets": 0,
    }


def _finalize_combined_tf(acc: dict) -> dict:
    cache = acc["cache"]
    remote = acc["remote"]
    return {
        "tf_name": acc["tf_name"],
        "cache": {
            "sources": sorted(cache["sources"]),
            "motifs": cache["motifs"],
            "site_sets": cache["site_sets"],
            "sites_seq": cache["sites_seq"],
            "sites_total": cache["sites_total"],
            "datasets": cache["datasets"],
        },
        "remote": {
            "sources": sorted(remote["sources"]),
            "motifs": remote["motifs"],
            "datasets": remote["datasets"],
        },
    }


def _aggregate_remote_totals(remote_summaries: Optional[dict[str, dict]]) -> dict:
    if not remote_summaries:
        return {
            "tfs": 0,
            "motifs": 0,
            "datasets": 0,
            "dataset_sources": [],
            "dataset_methods": [],
            "sources": [],
        }
    tf_set: set[str] = set()
    motifs = 0
    datasets = 0
    dataset_sources: set[str] = set()
    dataset_methods: set[str] = set()
    for source_id, summary in remote_summaries.items():
        totals = summary.get("totals") or {}
        motifs += int(totals.get("motifs") or 0)
        datasets += int(totals.get("datasets") or 0)
        dataset_sources.update(totals.get("dataset_sources") or [])
        dataset_methods.update(totals.get("dataset_methods") or [])
        for reg in summary.get("regulators", []):
            tf_name = reg.get("tf_name")
            if tf_name:
                tf_set.add(tf_name.lower())
    return {
        "tfs": len(tf_set),
        "motifs": motifs,
        "datasets": datasets,
        "dataset_sources": sorted(dataset_sources),
        "dataset_methods": sorted(dataset_methods),
        "sources": sorted(remote_summaries.keys()),
    }
