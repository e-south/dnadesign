"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/target_service.py

Resolve target regulator sets and catalog readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from dnadesign.cruncher.app.cache_readiness import cache_refresh_hint
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.ingest.site_windows import resolve_window_length
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif, read_lockfile
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path


@dataclass(frozen=True)
class TargetStatus:
    set_index: int
    tf_name: str
    source: Optional[str]
    motif_id: Optional[str]
    organism: Optional[dict[str, object]]
    has_matrix: bool
    has_sites: bool
    site_count: int
    site_total: int
    site_kind: Optional[str]
    dataset_id: Optional[str]
    matrix_source: Optional[str]
    pwm_source: str
    status: str
    message: Optional[str]


@dataclass(frozen=True)
class TargetCandidate:
    set_index: int
    tf_name: str
    candidates: list[CatalogEntry]


@dataclass(frozen=True)
class TargetStats:
    set_index: int
    tf_name: str
    source: str
    motif_id: str
    organism: Optional[dict[str, object]]
    matrix_length: Optional[int]
    matrix_source: Optional[str]
    site_count: int
    site_total: int
    site_kind: Optional[str]
    site_length_mean: Optional[float]
    site_length_min: Optional[int]
    site_length_max: Optional[int]
    site_length_source: Optional[str]
    dataset_id: Optional[str]
    dataset_method: Optional[str]
    reference_genome: Optional[str]


def resolve_category_targets(*, cfg: CruncherConfig, category_name: str) -> list[str]:
    tfs = cfg.regulator_categories.get(category_name)
    if tfs is None:
        available = ", ".join(sorted(cfg.regulator_categories.keys()))
        raise ValueError(f"category '{category_name}' not found. Available categories: {available or 'none'}")
    if not tfs:
        raise ValueError(f"category '{category_name}' is empty.")
    return list(tfs)


def _merge_text(values: Iterable[Optional[str]]) -> Optional[str]:
    unique = {value for value in values if value}
    if not unique:
        return None
    if len(unique) == 1:
        return unique.pop()
    return "mixed"


def _site_entries_for_target(
    *,
    catalog: CatalogIndex,
    entry: CatalogEntry,
    combine_sites: bool,
    site_kinds: Optional[list[str]],
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


def _needs_window_length(
    *,
    entries: list[CatalogEntry],
    window_lengths: dict[str, int],
) -> bool:
    if not entries:
        return False
    known_any = False
    variable = False
    lengths: set[int] = set()
    for entry in entries:
        min_len = entry.site_length_min
        max_len = entry.site_length_max
        if min_len is None or max_len is None:
            continue
        known_any = True
        if min_len != max_len:
            variable = True
        lengths.add(min_len)
        lengths.add(max_len)
    if not known_any:
        return False
    uniform = not variable and len(lengths) == 1
    if uniform:
        return False
    return not all(
        resolve_window_length(
            tf_name=entry.tf_name,
            dataset_id=entry.dataset_id,
            window_lengths=window_lengths,
        )
        is not None
        for entry in entries
    )


def list_targets(cfg: CruncherConfig) -> list[Tuple[int, str]]:
    targets: list[Tuple[int, str]] = []
    for idx, group in enumerate(cfg.regulator_sets, start=1):
        for tf in group:
            targets.append((idx, tf))
    return targets


def _status_for_entry(
    *,
    entry: CatalogEntry,
    pwm_source: str,
    min_sites: int,
    allow_low_sites: bool,
) -> Tuple[str, Optional[str]]:
    if pwm_source == "matrix":
        if not entry.has_matrix:
            return "missing-matrix", "No cached motif matrix available."
        return "ready", None
    if pwm_source == "sites":
        if not entry.has_sites:
            return "missing-sites", "No cached binding-site sequences available."
        if entry.site_count < min_sites:
            msg = f"Only {entry.site_count} sites available (min_sites_for_pwm={min_sites})."
            if allow_low_sites:
                return "warning", msg
            return "insufficient-sites", msg
        return "ready", None
    raise ValueError("pwm_source must be 'matrix' or 'sites'")


def _status_stub(
    *,
    set_index: int,
    tf_name: str,
    pwm_source: str,
    status: str,
    message: str,
    source: str | None = None,
    motif_id: str | None = None,
) -> TargetStatus:
    return TargetStatus(
        set_index=int(set_index),
        tf_name=tf_name,
        source=source,
        motif_id=motif_id,
        organism=None,
        has_matrix=False,
        has_sites=False,
        site_count=0,
        site_total=0,
        site_kind=None,
        dataset_id=None,
        matrix_source=None,
        pwm_source=pwm_source,
        status=status,
        message=message,
    )


def _lockfile_metadata(
    *,
    use_lockfile: bool,
    lock_path: Path,
) -> tuple[dict[str, LockedMotif], str | None, list[str] | None, bool | None]:
    if not (use_lockfile and lock_path.exists()):
        return {}, None, None, None
    lockfile = read_lockfile(lock_path)
    return lockfile.resolved, lockfile.pwm_source, lockfile.site_kinds, lockfile.combine_sites


def _lockfile_mismatch_message(
    *,
    targets: list[tuple[int, str]],
    lockmap: dict[str, LockedMotif],
    lockfile_pwm: str | None,
    lockfile_site_kinds: list[str] | None,
    lockfile_combine_sites: bool | None,
    expected_pwm_source: str,
    expected_site_kinds: list[str] | None,
    expected_combine_sites: bool,
) -> str | None:
    required = {tf for _, tf in targets}
    missing = required - set(lockmap.keys())
    extra = set(lockmap.keys()) - required
    mismatch: str | None = None
    if not lockfile_pwm:
        mismatch = "Lockfile missing pwm_source; re-run `cruncher lock <config>`."
    elif lockfile_pwm != expected_pwm_source:
        mismatch = (
            f"Lockfile pwm_source='{lockfile_pwm}' does not match expected pwm_source='{expected_pwm_source}'. "
            "Re-run `cruncher lock <config>`."
        )
    if expected_pwm_source == "sites":
        if expected_site_kinds is not None and lockfile_site_kinds is not None:
            if sorted(expected_site_kinds) != sorted(lockfile_site_kinds):
                mismatch = (
                    f"Lockfile site_kinds={lockfile_site_kinds} does not match expected site_kinds="
                    f"{expected_site_kinds}. "
                    "Re-run `cruncher lock <config>`."
                )
        if lockfile_combine_sites is not None and lockfile_combine_sites != expected_combine_sites:
            mismatch = (
                f"Lockfile combine_sites={lockfile_combine_sites} does not match expected combine_sites="
                f"{expected_combine_sites}. "
                "Re-run `cruncher lock <config>`."
            )
    if not (missing or extra or mismatch):
        return None
    details: list[str] = []
    if missing:
        details.append(f"missing: {', '.join(sorted(missing))}")
    if extra:
        details.append(f"extra: {', '.join(sorted(extra))}")
    if mismatch:
        details.append(mismatch)
    return " | ".join(details) if details else "Lockfile mismatch."


def _aggregate_site_entry_stats(
    entries: list[CatalogEntry],
) -> tuple[int, int, str | None, str | None]:
    if not entries:
        return 0, 0, None, None
    return (
        sum(candidate.site_count for candidate in entries),
        sum(candidate.site_total for candidate in entries),
        _merge_text(candidate.site_kind for candidate in entries),
        _merge_text(candidate.dataset_id for candidate in entries),
    )


def _resolve_target_catalog_entry(
    *,
    set_index: int,
    tf_name: str,
    cfg: CruncherConfig,
    catalog: CatalogIndex,
    lockmap: dict[str, LockedMotif],
    use_lockfile: bool,
    pwm_source: str,
    site_kinds: list[str] | None,
    combine_sites: bool,
    config_path: Path,
) -> tuple[CatalogEntry | None, LockedMotif | None, TargetStatus | None]:
    locked = lockmap.get(tf_name) if use_lockfile else None
    if locked is not None:
        entry = catalog.entries.get(f"{locked.source}:{locked.motif_id}")
        if entry is None:
            return (
                None,
                locked,
                _status_stub(
                    set_index=set_index,
                    tf_name=tf_name,
                    source=locked.source,
                    motif_id=locked.motif_id,
                    pwm_source=pwm_source,
                    status="missing-catalog",
                    message=f"Catalog entry missing for {locked.source}:{locked.motif_id}.",
                ),
            )
        return entry, locked, None
    if use_lockfile:
        return (
            None,
            None,
            _status_stub(
                set_index=set_index,
                tf_name=tf_name,
                pwm_source=pwm_source,
                status="missing-lock",
                message=f"Missing lock entry for '{tf_name}'. Run `cruncher lock -c {config_path.name}`.",
            ),
        )
    try:
        entry = select_catalog_entry(
            catalog=catalog,
            tf_name=tf_name,
            pwm_source=pwm_source,
            site_kinds=site_kinds,
            combine_sites=combine_sites,
            source_preference=cfg.catalog.source_preference,
            dataset_preference=cfg.catalog.dataset_preference,
            dataset_map=cfg.catalog.dataset_map,
            allow_ambiguous=cfg.catalog.allow_ambiguous,
        )
    except ValueError as exc:
        return (
            None,
            None,
            _status_stub(
                set_index=set_index,
                tf_name=tf_name,
                pwm_source=pwm_source,
                status="unresolved-target",
                message=str(exc),
            ),
        )
    return entry, None, None


def _status_from_site_counts(
    *,
    pwm_source: str,
    site_entries: list[CatalogEntry],
    site_count: int,
    min_sites_for_pwm: int,
    allow_low_sites: bool,
) -> tuple[str, str | None]:
    if pwm_source != "sites":
        raise ValueError("site-count status resolution requires pwm_source='sites'")
    if not site_entries:
        return "missing-sites", "No cached binding-site sequences available."
    if site_count < int(min_sites_for_pwm):
        msg = f"Only {site_count} sites available (cruncher.catalog.min_sites_for_pwm={int(min_sites_for_pwm)})."
        return ("warning", msg) if allow_low_sites else ("insufficient-sites", msg)
    return "ready", None


def _validate_status_artifacts(
    *,
    catalog_root: Path,
    entry: CatalogEntry,
    site_entries: list[CatalogEntry],
    pwm_source: str,
    status: str,
    message: str | None,
    site_entries_require_window: bool,
) -> tuple[str, str | None]:
    if status not in {"ready", "warning"}:
        return status, message
    if pwm_source == "matrix" and entry.has_matrix:
        motif_path = catalog_root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
        if not motif_path.exists():
            return "missing-matrix-file", f"Motif file missing: {motif_path}"
        return status, message
    if pwm_source != "sites":
        return status, message
    missing_paths: list[str] = []
    for candidate in site_entries:
        sites_path = catalog_root / "normalized" / "sites" / candidate.source / f"{candidate.motif_id}.jsonl"
        if not sites_path.exists():
            missing_paths.append(str(sites_path))
    if missing_paths:
        return "missing-sites-file", f"Sites file missing: {missing_paths[0]}"
    if site_entries_require_window:
        return (
            "needs-window",
            "Site lengths vary across cached entries; set cruncher.catalog.site_window_lengths for this TF "
            "or each dataset before building PWMs.",
        )
    return status, message


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
            raise ValueError(f"No cached sites found for '{tf_name}'. {cache_refresh_hint(pwm_source='sites')}")
        raise ValueError(f"No cached motifs found for '{tf_name}'. {cache_refresh_hint(pwm_source='matrix')}")
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


def target_statuses(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    pwm_source: Optional[str] = None,
    site_kinds: Optional[list[str]] = None,
    combine_sites: Optional[bool] = None,
    site_window_lengths: Optional[dict[str, int]] = None,
    use_lockfile: bool = True,
) -> list[TargetStatus]:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    lock_path = resolve_lock_path(config_path)
    targets = list_targets(cfg)
    effective_pwm_source = pwm_source or cfg.catalog.pwm_source
    effective_site_kinds = site_kinds if site_kinds is not None else cfg.catalog.site_kinds
    effective_combine_sites = combine_sites if combine_sites is not None else cfg.catalog.combine_sites
    if site_window_lengths is not None:
        effective_window_lengths = site_window_lengths
    else:
        effective_window_lengths = cfg.catalog.site_window_lengths
    lockmap, lockfile_pwm, lockfile_site_kinds, lockfile_combine_sites = _lockfile_metadata(
        use_lockfile=use_lockfile,
        lock_path=lock_path,
    )
    if use_lockfile and lock_path.exists():
        lock_mismatch = _lockfile_mismatch_message(
            targets=targets,
            lockmap=lockmap,
            lockfile_pwm=lockfile_pwm,
            lockfile_site_kinds=lockfile_site_kinds,
            lockfile_combine_sites=lockfile_combine_sites,
            expected_pwm_source=effective_pwm_source,
            expected_site_kinds=effective_site_kinds,
            expected_combine_sites=effective_combine_sites,
        )
        if lock_mismatch is not None:
            return [
                _status_stub(
                    set_index=set_index,
                    tf_name=tf,
                    pwm_source=effective_pwm_source,
                    status="mismatch-lock",
                    message=lock_mismatch,
                )
                for set_index, tf in targets
            ]

    catalog = CatalogIndex.load(catalog_root)
    statuses: list[TargetStatus] = []
    for set_index, tf in targets:
        entry, locked, unresolved_status = _resolve_target_catalog_entry(
            set_index=set_index,
            tf_name=tf,
            cfg=cfg,
            catalog=catalog,
            lockmap=lockmap,
            use_lockfile=use_lockfile,
            pwm_source=effective_pwm_source,
            site_kinds=effective_site_kinds,
            combine_sites=effective_combine_sites,
            config_path=config_path,
        )
        if unresolved_status is not None:
            statuses.append(unresolved_status)
            continue
        if entry is None:
            raise ValueError(f"Target resolution failed for '{tf}' without status.")
        status_site_entries = [entry]
        if effective_combine_sites:
            status_site_entries = _site_entries_for_target(
                catalog=catalog,
                entry=entry,
                combine_sites=effective_combine_sites,
                site_kinds=effective_site_kinds,
            )
        site_count, site_total, site_kind, dataset_id = _aggregate_site_entry_stats(status_site_entries)
        site_entries = status_site_entries

        if effective_pwm_source == "sites":
            status, message = _status_from_site_counts(
                pwm_source=effective_pwm_source,
                site_entries=site_entries,
                site_count=site_count,
                min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
                allow_low_sites=cfg.catalog.allow_low_sites,
            )
            if not site_entries:
                site_count = 0
                site_total = 0
                site_kind = None
                dataset_id = None
        else:
            status, message = _status_for_entry(
                entry=entry,
                pwm_source=effective_pwm_source,
                min_sites=cfg.catalog.min_sites_for_pwm,
                allow_low_sites=cfg.catalog.allow_low_sites,
            )

        if effective_pwm_source == "sites" and status in {"ready", "warning"} and site_total > site_count:
            missing = site_total - site_count
            seq_msg = (
                f"{missing} of {site_total} sites lack sequences. "
                "Configure ingest.genome_source=ncbi or pass --genome-fasta to hydrate sequences."
            )
            status = "missing-sequences"
            message = f"{message} {seq_msg}".strip() if message else seq_msg

        site_entries_require_window = False
        if effective_pwm_source == "sites" and status in {"ready", "warning"}:
            site_entries_require_window = _needs_window_length(
                entries=site_entries,
                window_lengths=effective_window_lengths,
            )
        status, message = _validate_status_artifacts(
            catalog_root=catalog_root,
            entry=entry,
            site_entries=site_entries,
            pwm_source=effective_pwm_source,
            status=status,
            message=message,
            site_entries_require_window=site_entries_require_window,
        )
        statuses.append(
            TargetStatus(
                set_index=set_index,
                tf_name=tf,
                source=locked.source if locked is not None else entry.source,
                motif_id=locked.motif_id if locked is not None else entry.motif_id,
                organism=entry.organism,
                has_matrix=entry.has_matrix,
                has_sites=any(candidate.has_sites for candidate in site_entries),
                site_count=site_count,
                site_total=site_total,
                site_kind=site_kind,
                dataset_id=dataset_id,
                matrix_source=entry.matrix_source,
                pwm_source=effective_pwm_source,
                status=status,
                message=message,
            )
        )
    return statuses


def has_blocking_target_errors(statuses: Iterable[TargetStatus]) -> bool:
    return any(
        status.status
        in {
            "missing-lock",
            "missing-catalog",
            "unresolved-target",
            "missing-matrix",
            "missing-matrix-file",
            "missing-sites",
            "missing-sites-file",
            "missing-sequences",
            "insufficient-sites",
            "needs-window",
            "mismatch-lock",
        }
        for status in statuses
    )


def target_candidates(*, cfg: CruncherConfig, config_path: Path) -> list[TargetCandidate]:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    candidates: list[TargetCandidate] = []
    for set_index, tf in list_targets(cfg):
        entries = catalog.list(tf_name=tf, include_synonyms=True)
        candidates.append(TargetCandidate(set_index=set_index, tf_name=tf, candidates=entries))
    return candidates


def target_candidates_fuzzy(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    min_score: float = 0.6,
    limit: int = 10,
) -> list[TargetCandidate]:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    candidates: list[TargetCandidate] = []
    for set_index, tf in list_targets(cfg):
        entries = catalog.search(query=tf, fuzzy=True, min_score=min_score, limit=limit)
        candidates.append(TargetCandidate(set_index=set_index, tf_name=tf, candidates=entries))
    return candidates


def target_stats(
    *,
    cfg: CruncherConfig,
    config_path: Path,
) -> list[TargetStats]:
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    stats: list[TargetStats] = []
    for set_index, tf in list_targets(cfg):
        entries = catalog.list(tf_name=tf, include_synonyms=True)
        for entry in entries:
            stats.append(
                TargetStats(
                    set_index=set_index,
                    tf_name=tf,
                    source=entry.source,
                    motif_id=entry.motif_id,
                    organism=entry.organism,
                    matrix_length=entry.matrix_length,
                    matrix_source=entry.matrix_source,
                    site_count=entry.site_count,
                    site_total=entry.site_total,
                    site_kind=entry.site_kind,
                    site_length_mean=entry.site_length_mean,
                    site_length_min=entry.site_length_min,
                    site_length_max=entry.site_length_max,
                    site_length_source=entry.site_length_source,
                    dataset_id=entry.dataset_id,
                    dataset_method=entry.dataset_method,
                    reference_genome=entry.reference_genome,
                )
            )
    return stats
