"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/catalog_targets.py

Resolve catalog target selections for matrix and site export flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import typer

from dnadesign.cruncher.app.target_service import select_catalog_entry
from dnadesign.cruncher.cli.catalog_utils import (
    ResolvedTarget,
    _dedupe,
    _ensure_entry_matches_pwm_source,
    _parse_ref,
    _resolve_set_tfs,
)
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.logos import site_entries_for_logo


def _resolve_targets(
    *,
    cfg,
    config_path: Path,
    tfs: Sequence[str],
    refs: Sequence[str],
    set_index: int | None,
    source_filter: str | None,
) -> tuple[list[ResolvedTarget], CatalogIndex]:
    if set_index is not None and (tfs or refs):
        raise typer.BadParameter("--set cannot be combined with --tf or --ref.")
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    tf_names = list(tfs)
    if set_index is not None:
        tf_names = _resolve_set_tfs(cfg, set_index)
        if not tf_names and not refs:
            raise typer.BadParameter(f"regulator_sets[{set_index}] is empty.")
    if not tf_names and not refs:
        tf_names = [tf for group in cfg.regulator_sets for tf in group]
    tf_names = _dedupe(tf_names)
    refs = _dedupe(refs)
    if not tf_names and not refs:
        raise typer.BadParameter("No targets resolved. Provide --tf, --ref, or --set.")

    targets: list[ResolvedTarget] = []
    seen_keys: set[str] = set()

    for ref in refs:
        source, motif_id = _parse_ref(ref)
        if source_filter and source_filter != source:
            raise typer.BadParameter(f"--source {source_filter} does not match explicit ref {ref}.")
        entry = catalog.entries.get(f"{source}:{motif_id}")
        if entry is None:
            raise ValueError(f"No catalog entry found for {ref}.")
        _ensure_entry_matches_pwm_source(
            entry,
            cfg.catalog.pwm_source,
            cfg.catalog.site_kinds,
            tf_name=entry.tf_name,
            ref=ref,
        )
        site_entries = []
        if cfg.catalog.pwm_source == "sites":
            site_entries = site_entries_for_logo(
                catalog=catalog,
                entry=entry,
                combine_sites=cfg.catalog.combine_sites,
                site_kinds=cfg.catalog.site_kinds,
            )
        key = entry.key
        if key in seen_keys:
            continue
        targets.append(
            ResolvedTarget(
                tf_name=entry.tf_name,
                ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                entry=entry,
                site_entries=site_entries,
            )
        )
        seen_keys.add(key)

    if tf_names:
        catalog_for_select = catalog
        if source_filter:
            filtered = {k: v for k, v in catalog.entries.items() if v.source == source_filter}
            catalog_for_select = CatalogIndex(entries=filtered)
        for tf_name in tf_names:
            if source_filter:
                all_candidates = catalog.list(tf_name=tf_name, include_synonyms=True)
                if not any(candidate.source == source_filter for candidate in all_candidates):
                    raise ValueError(f"No cached entries for '{tf_name}' in source '{source_filter}'.")
            entry = select_catalog_entry(
                catalog=catalog_for_select,
                tf_name=tf_name,
                pwm_source=cfg.catalog.pwm_source,
                site_kinds=cfg.catalog.site_kinds,
                combine_sites=cfg.catalog.combine_sites,
                source_preference=cfg.catalog.source_preference,
                dataset_preference=cfg.catalog.dataset_preference,
                dataset_map=cfg.catalog.dataset_map,
                allow_ambiguous=cfg.catalog.allow_ambiguous,
            )
            site_entries = []
            if cfg.catalog.pwm_source == "sites":
                site_entries = site_entries_for_logo(
                    catalog=catalog,
                    entry=entry,
                    combine_sites=cfg.catalog.combine_sites,
                    site_kinds=cfg.catalog.site_kinds,
                )
                if not site_entries:
                    raise ValueError(
                        f"No cached site entries available for '{tf_name}' "
                        f"with cruncher.catalog.site_kinds={cfg.catalog.site_kinds}."
                    )
            key = entry.key
            if key in seen_keys:
                continue
            targets.append(
                ResolvedTarget(
                    tf_name=tf_name,
                    ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                    entry=entry,
                    site_entries=site_entries,
                )
            )
            seen_keys.add(key)

    if not targets:
        raise typer.BadParameter("No catalog entries matched the requested targets.")
    return targets, catalog


def _resolve_site_targets(
    *,
    cfg,
    config_path: Path,
    tfs: Sequence[str],
    refs: Sequence[str],
    set_index: int | None,
    source_filter: str | None,
) -> tuple[list[ResolvedTarget], CatalogIndex]:
    if set_index is not None and (tfs or refs):
        raise typer.BadParameter("--set cannot be combined with --tf or --ref.")
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    tf_names = list(tfs)
    if set_index is not None:
        tf_names = _resolve_set_tfs(cfg, set_index)
        if not tf_names and not refs:
            raise typer.BadParameter(f"regulator_sets[{set_index}] is empty.")
    if not tf_names and not refs:
        tf_names = [tf for group in cfg.regulator_sets for tf in group]
    tf_names = _dedupe(tf_names)
    refs = _dedupe(refs)
    if not tf_names and not refs:
        raise typer.BadParameter("No targets resolved. Provide --tf, --ref, or --set.")

    targets: list[ResolvedTarget] = []
    seen_keys: set[str] = set()

    for ref in refs:
        source, motif_id = _parse_ref(ref)
        if source_filter and source_filter != source:
            raise typer.BadParameter(f"--source {source_filter} does not match explicit ref {ref}.")
        entry = catalog.entries.get(f"{source}:{motif_id}")
        if entry is None:
            raise ValueError(f"No catalog entry found for {ref}.")
        _ensure_entry_matches_pwm_source(
            entry,
            "sites",
            cfg.catalog.site_kinds,
            tf_name=entry.tf_name,
            ref=ref,
        )
        site_entries = site_entries_for_logo(
            catalog=catalog,
            entry=entry,
            combine_sites=cfg.catalog.combine_sites,
            site_kinds=cfg.catalog.site_kinds,
        )
        if not site_entries:
            raise ValueError(
                "No cached site entries available for "
                f"'{entry.tf_name}' with cruncher.catalog.site_kinds={cfg.catalog.site_kinds}."
            )
        key = entry.key
        if key in seen_keys:
            continue
        targets.append(
            ResolvedTarget(
                tf_name=entry.tf_name,
                ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                entry=entry,
                site_entries=site_entries,
            )
        )
        seen_keys.add(key)

    if tf_names:
        catalog_for_select = catalog
        if source_filter:
            filtered = {k: v for k, v in catalog.entries.items() if v.source == source_filter}
            catalog_for_select = CatalogIndex(entries=filtered)
        for tf_name in tf_names:
            if source_filter:
                all_candidates = catalog.list(tf_name=tf_name, include_synonyms=True)
                if not any(candidate.source == source_filter for candidate in all_candidates):
                    raise ValueError(f"No cached entries for '{tf_name}' in source '{source_filter}'.")
            entry = select_catalog_entry(
                catalog=catalog_for_select,
                tf_name=tf_name,
                pwm_source="sites",
                site_kinds=cfg.catalog.site_kinds,
                combine_sites=cfg.catalog.combine_sites,
                source_preference=cfg.catalog.source_preference,
                dataset_preference=cfg.catalog.dataset_preference,
                dataset_map=cfg.catalog.dataset_map,
                allow_ambiguous=cfg.catalog.allow_ambiguous,
            )
            site_entries = site_entries_for_logo(
                catalog=catalog,
                entry=entry,
                combine_sites=cfg.catalog.combine_sites,
                site_kinds=cfg.catalog.site_kinds,
            )
            if not site_entries:
                raise ValueError(
                    "No cached site entries available for "
                    f"'{tf_name}' with cruncher.catalog.site_kinds={cfg.catalog.site_kinds}."
                )
            key = entry.key
            if key in seen_keys:
                continue
            targets.append(
                ResolvedTarget(
                    tf_name=tf_name,
                    ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                    entry=entry,
                    site_entries=site_entries,
                )
            )
            seen_keys.add(key)

    if not targets:
        raise typer.BadParameter("No catalog entries matched the requested targets.")
    return targets, catalog
