"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/campaign_service.py

Build, validate, and expand campaign configurations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Optional

from dnadesign.cruncher.config.schema_v3 import (
    CampaignConfig,
    CampaignSelectorsConfig,
    CruncherConfig,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root


@dataclass(frozen=True)
class RegulatorMetrics:
    tf_name: str
    source: str
    motif_id: str
    matrix_length: Optional[int]
    site_count: int
    site_total: int
    site_kind: Optional[str]
    dataset_id: Optional[str]
    info_bits: Optional[float]


@dataclass(frozen=True)
class CampaignExpansion:
    name: str
    campaign_id: str
    categories: Dict[str, list[str]]
    regulator_sets: list[list[str]]
    selectors: Dict[str, object]
    rules: Dict[str, object]
    metrics: Dict[str, RegulatorMetrics]


@dataclass(frozen=True)
class CampaignValidationResult:
    name: str
    campaign_id: Optional[str]
    categories: Dict[str, list[str]]
    selected: Dict[str, list[str]]
    filtered: Dict[str, list[str]]
    errors: list[str]
    warnings: list[str]
    metrics: Dict[str, RegulatorMetrics]


def resolve_category_targets(*, cfg: CruncherConfig, category_name: str) -> list[str]:
    tfs = cfg.regulator_categories.get(category_name)
    if tfs is None:
        available = ", ".join(sorted(cfg.regulator_categories.keys()))
        raise ValueError(f"category '{category_name}' not found. Available categories: {available or 'none'}")
    if not tfs:
        raise ValueError(f"category '{category_name}' is empty.")
    return list(tfs)


def resolve_campaign_tf_names(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    campaign_name: str,
    apply_selectors: bool = True,
    include_metrics: bool = False,
) -> list[str]:
    campaign = _find_campaign(cfg, campaign_name)
    categories = _resolve_categories(cfg, campaign)

    if apply_selectors and campaign.selectors.requires_catalog():
        categories, _ = _apply_selectors(
            cfg=cfg,
            config_path=config_path,
            campaign=campaign,
            categories=categories,
            include_metrics=include_metrics,
        )

    _validate_category_rules(campaign, categories)

    tf_names = sorted({tf for group in categories.values() for tf in group})
    if not tf_names:
        raise ValueError(f"campaign '{campaign.name}' produced no TFs")
    return tf_names


def validate_campaign(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    campaign_name: str,
    apply_selectors: bool = True,
    include_metrics: bool = True,
) -> CampaignValidationResult:
    campaign = _find_campaign(cfg, campaign_name)
    categories = _resolve_categories(cfg, campaign)

    errors: list[str] = []
    warnings: list[str] = []
    selected: Dict[str, list[str]] = {}
    filtered: Dict[str, list[str]] = {}
    metrics: Dict[str, RegulatorMetrics] = {}

    needs_catalog = apply_selectors or include_metrics
    catalog = None
    store: Optional[CatalogMotifStore] = None
    if needs_catalog:
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
        if not catalog_root.exists():
            errors.append(
                f"Catalog root not found at {catalog_root}. Run `cruncher fetch motifs/sites` before validation."
            )
        else:
            catalog = CatalogIndex.load(catalog_root)
            if campaign.selectors.min_info_bits is not None:
                store = CatalogMotifStore(
                    catalog_root,
                    pwm_source=cfg.catalog.pwm_source,
                    site_kinds=cfg.catalog.site_kinds,
                    combine_sites=cfg.catalog.combine_sites,
                    site_window_lengths=cfg.catalog.site_window_lengths,
                    site_window_center=cfg.catalog.site_window_center,
                    apply_pwm_window=False,
                    min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
                    allow_low_sites=cfg.catalog.allow_low_sites,
                    pseudocounts=cfg.catalog.pseudocounts,
                )

    if errors:
        return CampaignValidationResult(
            name=campaign.name,
            campaign_id=None,
            categories=categories,
            selected=selected,
            filtered=filtered,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    for category_name in campaign.categories:
        tfs = categories.get(category_name, [])
        keep: list[str] = []
        dropped: list[str] = []
        for tf_name in tfs:
            try:
                if catalog is None:
                    if apply_selectors or include_metrics:
                        raise ValueError("Catalog not loaded.")
                    keep.append(tf_name)
                    continue
                entry = select_catalog_entry(
                    catalog=catalog,
                    tf_name=tf_name,
                    pwm_source=cfg.catalog.pwm_source,
                    site_kinds=cfg.catalog.site_kinds,
                    combine_sites=cfg.catalog.combine_sites,
                    source_preference=campaign.selectors.source_preference or cfg.catalog.source_preference,
                    dataset_preference=campaign.selectors.dataset_preference or cfg.catalog.dataset_preference,
                    dataset_map=cfg.catalog.dataset_map,
                    allow_ambiguous=cfg.catalog.allow_ambiguous,
                )
                metric = _build_metrics(
                    entry=entry,
                    catalog=catalog,
                    store=store,
                    selectors=campaign.selectors,
                    include_metrics=include_metrics,
                    combine_sites=cfg.catalog.combine_sites,
                    site_kinds=cfg.catalog.site_kinds,
                )
                if include_metrics:
                    metrics[tf_name] = metric
                if apply_selectors and not _passes_selectors(metric, campaign.selectors):
                    dropped.append(tf_name)
                    continue
                keep.append(tf_name)
            except Exception as exc:
                errors.append(f"{category_name}/{tf_name} - {exc}")
        if apply_selectors and not keep:
            errors.append(f"campaign '{campaign.name}' selectors removed all TFs in category '{category_name}'.")
        selected[category_name] = keep if apply_selectors else list(tfs)
        if apply_selectors:
            filtered[category_name] = dropped

    if not errors:
        try:
            _validate_category_rules(campaign, selected if apply_selectors else categories)
        except ValueError as exc:
            errors.append(str(exc))

    campaign_id = None
    if not errors:
        campaign_id = _campaign_id(campaign, selected if apply_selectors else categories)

    return CampaignValidationResult(
        name=campaign.name,
        campaign_id=campaign_id,
        categories=categories,
        selected=selected,
        filtered=filtered,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def expand_campaign(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    campaign_name: str,
    include_metrics: bool = True,
) -> CampaignExpansion:
    campaign = _find_campaign(cfg, campaign_name)
    categories = _resolve_categories(cfg, campaign)

    metrics: Dict[str, RegulatorMetrics] = {}
    if campaign.selectors.requires_catalog():
        categories, metrics = _apply_selectors(
            cfg=cfg,
            config_path=config_path,
            campaign=campaign,
            categories=categories,
            include_metrics=include_metrics,
        )

    _validate_category_rules(campaign, categories)

    regulator_sets: list[list[str]] = []
    if campaign.within_category is not None:
        regulator_sets.extend(_expand_within(categories, campaign))
    if campaign.across_categories is not None:
        regulator_sets.extend(_expand_across(categories, campaign))

    if campaign.dedupe_sets:
        regulator_sets = _dedupe_sets(regulator_sets)

    if not regulator_sets:
        raise ValueError(f"campaign '{campaign.name}' produced no regulator_sets")

    campaign_id = _campaign_id(campaign, categories)
    rules = {
        "within_category": campaign.within_category.model_dump(mode="json")
        if campaign.within_category is not None
        else None,
        "across_categories": campaign.across_categories.model_dump(mode="json")
        if campaign.across_categories is not None
        else None,
        "allow_overlap": campaign.allow_overlap,
        "distinct_across_categories": campaign.distinct_across_categories,
        "dedupe_sets": campaign.dedupe_sets,
    }
    return CampaignExpansion(
        name=campaign.name,
        campaign_id=campaign_id,
        categories=categories,
        regulator_sets=regulator_sets,
        selectors=campaign.selectors.model_dump(mode="json"),
        rules=rules,
        metrics=metrics,
    )


def build_campaign_manifest(
    *,
    expansion: CampaignExpansion,
    config_path: Path,
) -> Dict[str, object]:
    manifest: Dict[str, object] = {
        "campaign_id": expansion.campaign_id,
        "campaign_name": expansion.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": {
            "path": str(config_path),
            "sha256": sha256_path(config_path),
        },
        "categories": expansion.categories,
        "selectors": expansion.selectors,
        "rules": expansion.rules,
        "expanded_sets": expansion.regulator_sets,
        "expanded_count": len(expansion.regulator_sets),
    }
    if expansion.metrics:
        manifest["targets"] = {
            name: {
                "source": metric.source,
                "motif_id": metric.motif_id,
                "matrix_length": metric.matrix_length,
                "site_count": metric.site_count,
                "site_total": metric.site_total,
                "site_kind": metric.site_kind,
                "dataset_id": metric.dataset_id,
                "info_bits": metric.info_bits,
            }
            for name, metric in expansion.metrics.items()
        }
    return manifest


def collect_campaign_metrics(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    tf_names: Iterable[str],
    source_preference: Optional[list[str]] = None,
    dataset_preference: Optional[list[str]] = None,
) -> Dict[str, RegulatorMetrics]:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    if not catalog_root.exists():
        raise ValueError(
            f"Catalog root not found at {catalog_root}. Run `cruncher fetch motifs/sites` before campaign metrics."
        )
    catalog = CatalogIndex.load(catalog_root)
    store = CatalogMotifStore(
        catalog_root,
        pwm_source=cfg.catalog.pwm_source,
        site_kinds=cfg.catalog.site_kinds,
        combine_sites=cfg.catalog.combine_sites,
        site_window_lengths=cfg.catalog.site_window_lengths,
        site_window_center=cfg.catalog.site_window_center,
        apply_pwm_window=False,
        min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
        allow_low_sites=cfg.catalog.allow_low_sites,
        pseudocounts=cfg.catalog.pseudocounts,
    )
    metrics: Dict[str, RegulatorMetrics] = {}
    for tf_name in sorted({name for name in tf_names if name}):
        entry = select_catalog_entry(
            catalog=catalog,
            tf_name=tf_name,
            pwm_source=cfg.catalog.pwm_source,
            site_kinds=cfg.catalog.site_kinds,
            combine_sites=cfg.catalog.combine_sites,
            source_preference=source_preference or cfg.catalog.source_preference,
            dataset_preference=dataset_preference or cfg.catalog.dataset_preference,
            dataset_map=cfg.catalog.dataset_map,
            allow_ambiguous=cfg.catalog.allow_ambiguous,
        )
        pwm = store.get_pwm(MotifRef(source=entry.source, motif_id=entry.motif_id))
        info_bits = pwm.information_bits()
        site_count, site_total, site_kind = _resolve_site_counts(
            catalog=catalog,
            entry=entry,
            combine_sites=cfg.catalog.combine_sites,
            site_kinds=cfg.catalog.site_kinds,
        )
        dataset_id = _resolve_dataset_id(
            catalog=catalog,
            entry=entry,
            combine_sites=cfg.catalog.combine_sites,
            site_kinds=cfg.catalog.site_kinds,
        )
        metrics[tf_name] = RegulatorMetrics(
            tf_name=entry.tf_name,
            source=entry.source,
            motif_id=entry.motif_id,
            matrix_length=entry.matrix_length,
            site_count=site_count,
            site_total=site_total,
            site_kind=site_kind,
            dataset_id=dataset_id,
            info_bits=info_bits,
        )
    return metrics


def _find_campaign(cfg: CruncherConfig, name: str) -> CampaignConfig:
    for campaign in cfg.campaigns:
        if campaign.name == name:
            return campaign
    available = ", ".join(sorted(c.name for c in cfg.campaigns))
    raise ValueError(f"campaign '{name}' not found. Available campaigns: {available or 'none'}")


def _resolve_categories(cfg: CruncherConfig, campaign: CampaignConfig) -> Dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for name in campaign.categories:
        tfs = cfg.regulator_categories.get(name)
        if tfs is None:
            raise ValueError(f"campaign '{campaign.name}' references unknown category '{name}'")
        categories[name] = list(tfs)
    return categories


def _validate_category_rules(campaign: CampaignConfig, categories: Dict[str, list[str]]) -> None:
    if not campaign.allow_overlap:
        overlaps: set[str] = set()
        seen: set[str] = set()
        for name in campaign.categories:
            for tf in categories.get(name, []):
                if tf in seen:
                    overlaps.add(tf)
                seen.add(tf)
        if overlaps:
            overlaps_list = ", ".join(sorted(overlaps))
            raise ValueError(
                f"campaign '{campaign.name}' forbids overlaps, but TFs appear in multiple categories: {overlaps_list}"
            )


def _apply_selectors(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    campaign: CampaignConfig,
    categories: Dict[str, list[str]],
    include_metrics: bool,
) -> tuple[Dict[str, list[str]], Dict[str, RegulatorMetrics]]:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    if not catalog_root.exists():
        raise ValueError(
            f"Catalog root not found at {catalog_root}. Run `cruncher fetch motifs/sites` before campaign selection."
        )
    catalog = CatalogIndex.load(catalog_root)
    store: Optional[CatalogMotifStore] = None
    if campaign.selectors.min_info_bits is not None:
        store = CatalogMotifStore(
            catalog_root,
            pwm_source=cfg.catalog.pwm_source,
            site_kinds=cfg.catalog.site_kinds,
            combine_sites=cfg.catalog.combine_sites,
            site_window_lengths=cfg.catalog.site_window_lengths,
            site_window_center=cfg.catalog.site_window_center,
            apply_pwm_window=False,
            min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
            allow_low_sites=cfg.catalog.allow_low_sites,
            pseudocounts=cfg.catalog.pseudocounts,
        )

    selected: Dict[str, list[str]] = {}
    metrics: Dict[str, RegulatorMetrics] = {}
    for category_name in campaign.categories:
        tfs = categories.get(category_name, [])
        filtered: list[str] = []
        for tf_name in tfs:
            entry = select_catalog_entry(
                catalog=catalog,
                tf_name=tf_name,
                pwm_source=cfg.catalog.pwm_source,
                site_kinds=cfg.catalog.site_kinds,
                combine_sites=cfg.catalog.combine_sites,
                source_preference=campaign.selectors.source_preference or cfg.catalog.source_preference,
                dataset_preference=campaign.selectors.dataset_preference or cfg.catalog.dataset_preference,
                dataset_map=cfg.catalog.dataset_map,
                allow_ambiguous=cfg.catalog.allow_ambiguous,
            )
            metric = _build_metrics(
                entry=entry,
                catalog=catalog,
                store=store,
                selectors=campaign.selectors,
                include_metrics=include_metrics,
                combine_sites=cfg.catalog.combine_sites,
                site_kinds=cfg.catalog.site_kinds,
            )
            if _passes_selectors(metric, campaign.selectors):
                filtered.append(tf_name)
            if include_metrics:
                metrics[tf_name] = metric
        if not filtered:
            raise ValueError(f"campaign '{campaign.name}' selectors removed all TFs in category '{category_name}'.")
        selected[category_name] = filtered
    return selected, metrics


def _build_metrics(
    *,
    entry: CatalogEntry,
    catalog: CatalogIndex,
    store: Optional[CatalogMotifStore],
    selectors: CampaignSelectorsConfig,
    include_metrics: bool,
    combine_sites: bool,
    site_kinds: Optional[list[str]],
) -> RegulatorMetrics:
    info_bits: Optional[float] = None
    if selectors.min_info_bits is not None:
        if store is None:
            raise ValueError("min_info_bits selector requires access to cached PWM data.")
        pwm = store.get_pwm(MotifRef(source=entry.source, motif_id=entry.motif_id))
        info_bits = pwm.information_bits()

    if include_metrics or selectors.min_site_count is not None:
        site_count, site_total, site_kind = _resolve_site_counts(
            catalog=catalog,
            entry=entry,
            combine_sites=combine_sites,
            site_kinds=site_kinds,
        )
    else:
        site_count = entry.site_count
        site_total = entry.site_total
        site_kind = entry.site_kind
    dataset_id = _resolve_dataset_id(
        catalog=catalog,
        entry=entry,
        combine_sites=combine_sites,
        site_kinds=site_kinds,
    )

    return RegulatorMetrics(
        tf_name=entry.tf_name,
        source=entry.source,
        motif_id=entry.motif_id,
        matrix_length=entry.matrix_length,
        site_count=site_count,
        site_total=site_total,
        site_kind=site_kind,
        dataset_id=dataset_id,
        info_bits=info_bits,
    )


def _resolve_site_counts(
    *,
    catalog: CatalogIndex,
    entry: CatalogEntry,
    combine_sites: bool,
    site_kinds: Optional[list[str]],
) -> tuple[int, int, Optional[str]]:
    if not combine_sites:
        return entry.site_count, entry.site_total, entry.site_kind
    entries = [
        candidate
        for candidate in catalog.entries.values()
        if candidate.tf_name.lower() == entry.tf_name.lower() and candidate.has_sites
    ]
    if site_kinds is not None:
        entries = [candidate for candidate in entries if candidate.site_kind in site_kinds]
    if not entries:
        return entry.site_count, entry.site_total, entry.site_kind
    site_count = sum(candidate.site_count for candidate in entries)
    site_total = sum(candidate.site_total for candidate in entries)
    site_kind = _merge_text(candidate.site_kind for candidate in entries)
    return site_count, site_total, site_kind


def _resolve_dataset_id(
    *,
    catalog: CatalogIndex,
    entry: CatalogEntry,
    combine_sites: bool,
    site_kinds: Optional[list[str]],
) -> Optional[str]:
    if not combine_sites:
        return entry.dataset_id
    entries = [
        candidate
        for candidate in catalog.entries.values()
        if candidate.tf_name.lower() == entry.tf_name.lower() and candidate.has_sites
    ]
    if site_kinds is not None:
        entries = [candidate for candidate in entries if candidate.site_kind in site_kinds]
    if not entries:
        return entry.dataset_id
    return _merge_text(candidate.dataset_id for candidate in entries)


def _merge_text(values: Iterable[Optional[str]]) -> Optional[str]:
    unique = {value for value in values if value}
    if not unique:
        return None
    if len(unique) == 1:
        return unique.pop()
    return "mixed"


def _passes_selectors(metric: RegulatorMetrics, selectors: CampaignSelectorsConfig) -> bool:
    if selectors.min_info_bits is not None:
        if metric.info_bits is None:
            raise ValueError(f"Missing info_bits for TF '{metric.tf_name}'.")
        if metric.info_bits < selectors.min_info_bits:
            return False
    if selectors.min_site_count is not None:
        if metric.site_count < selectors.min_site_count:
            return False
    if selectors.min_pwm_length is not None:
        if metric.matrix_length is None:
            raise ValueError(f"Missing matrix_length for TF '{metric.tf_name}'.")
        if metric.matrix_length < selectors.min_pwm_length:
            return False
    if selectors.max_pwm_length is not None:
        if metric.matrix_length is None:
            raise ValueError(f"Missing matrix_length for TF '{metric.tf_name}'.")
        if metric.matrix_length > selectors.max_pwm_length:
            return False
    return True


def select_catalog_entry(
    *,
    catalog: CatalogIndex,
    tf_name: str,
    pwm_source: str,
    site_kinds: Optional[list[str]],
    combine_sites: bool,
    source_preference: list[str],
    dataset_preference: list[str],
    dataset_map: dict[str, str],
    allow_ambiguous: bool,
) -> CatalogEntry:
    candidates = catalog.list(tf_name=tf_name, include_synonyms=True)
    if not candidates:
        if pwm_source == "sites":
            raise ValueError(f"No cached sites found for '{tf_name}'. Run `cruncher fetch sites` first.")
        raise ValueError(f"No cached motifs found for '{tf_name}'. Run `cruncher fetch motifs` first.")
    if pwm_source == "matrix":
        candidates = [c for c in candidates if c.has_matrix]
    elif pwm_source == "sites":
        candidates = [c for c in candidates if c.has_sites]
        if site_kinds:
            candidates = [c for c in candidates if c.site_kind in site_kinds]
    else:
        raise ValueError("pwm_source must be 'matrix' or 'sites'")
    if not candidates:
        raise ValueError(
            f"No cached data for '{tf_name}' compatible with pwm_source='{pwm_source}'. "
            "Fetch motifs/sites or change pwm_source."
        )
    candidates = sorted(candidates, key=lambda c: (c.source, c.motif_id))
    if dataset_map and tf_name in dataset_map:
        dataset_id = dataset_map[tf_name]
        candidates = [c for c in candidates if c.dataset_id == dataset_id]
        if not candidates:
            raise ValueError(f"Dataset '{dataset_id}' not found for TF '{tf_name}'.")
        return candidates[0]
    if dataset_preference:
        by_dataset = {c.dataset_id: c for c in candidates if c.dataset_id}
        for pref in dataset_preference:
            if pref in by_dataset:
                return by_dataset[pref]
    if source_preference:
        by_source = {c.source: c for c in candidates}
        for pref in source_preference:
            if pref in by_source:
                return by_source[pref]
    allow_ambiguous_effective = allow_ambiguous or (combine_sites and pwm_source == "sites")
    if len(candidates) == 1 or allow_ambiguous_effective:
        return candidates[0]
    options = ", ".join(f"{c.source}:{c.motif_id}" for c in candidates)
    raise ValueError(f"Ambiguous motif for '{tf_name}'. Candidates: {options}")


def _expand_within(categories: Dict[str, list[str]], campaign: CampaignConfig) -> list[list[str]]:
    output: list[list[str]] = []
    sizes = campaign.within_category.sizes if campaign.within_category is not None else []
    for category in campaign.categories:
        tfs = categories[category]
        for size in sizes:
            if size > len(tfs):
                raise ValueError(
                    f"campaign '{campaign.name}' cannot build within-category size={size} "
                    f"for '{category}' (size={len(tfs)})."
                )
            for combo in combinations(tfs, size):
                output.append(list(combo))
    return output


def _expand_across(categories: Dict[str, list[str]], campaign: CampaignConfig) -> list[list[str]]:
    if len(campaign.categories) < 2:
        raise ValueError(f"campaign '{campaign.name}' requires at least two categories for across_categories.")
    sizes = campaign.across_categories.sizes if campaign.across_categories is not None else []
    max_per_category = None
    if campaign.across_categories is not None:
        max_per_category = campaign.across_categories.max_per_category

    category_order = {name: idx for idx, name in enumerate(campaign.categories)}
    tf_to_categories: dict[str, list[str]] = {}
    union: list[str] = []
    for name in campaign.categories:
        for tf in categories[name]:
            tf_to_categories.setdefault(tf, []).append(name)
            if tf not in union:
                union.append(tf)

    tf_primary = {tf: sorted(cats, key=lambda n: category_order[n])[0] for tf, cats in tf_to_categories.items()}

    output: list[list[str]] = []
    for size in sizes:
        if size > len(union):
            raise ValueError(
                f"campaign '{campaign.name}' cannot build across-category size={size} "
                f"(only {len(union)} unique TFs available)."
            )
        for combo in combinations(union, size):
            categories_covered: set[str] = set()
            counts: dict[str, int] = {name: 0 for name in campaign.categories}
            for tf in combo:
                cats = tf_to_categories.get(tf, [])
                if campaign.distinct_across_categories:
                    primary = tf_primary[tf]
                    categories_covered.add(primary)
                    counts[primary] += 1
                else:
                    categories_covered.update(cats)
                    for cat in cats:
                        counts[cat] += 1
            if len(categories_covered) < 2:
                continue
            if max_per_category is not None:
                if any(count > max_per_category for count in counts.values()):
                    continue
            output.append(list(combo))
    return output


def _dedupe_sets(groups: Iterable[list[str]]) -> list[list[str]]:
    output: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(sorted(group))
        if key in seen:
            continue
        seen.add(key)
        output.append(group)
    return output


def _campaign_id(campaign: CampaignConfig, categories: Dict[str, list[str]]) -> str:
    payload = {
        "name": campaign.name,
        "categories": categories,
        "campaign": campaign.model_dump(mode="json"),
    }
    digest = sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest
