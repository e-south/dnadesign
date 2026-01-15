"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/services/target_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.ingest.site_windows import resolve_window_length
from dnadesign.cruncher.services.campaign_service import select_catalog_entry
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif, read_lockfile


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
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
    lock_path = catalog_root / "locks" / f"{config_path.stem}.lock.json"
    effective_pwm_source = pwm_source or cfg.motif_store.pwm_source
    effective_site_kinds = site_kinds if site_kinds is not None else cfg.motif_store.site_kinds
    effective_combine_sites = combine_sites if combine_sites is not None else cfg.motif_store.combine_sites
    if site_window_lengths is not None:
        effective_window_lengths = site_window_lengths
    else:
        effective_window_lengths = cfg.motif_store.site_window_lengths
    lockmap: dict[str, LockedMotif] = {}
    lockfile_pwm = None
    lockfile_site_kinds: Optional[list[str]] = None
    lockfile_combine_sites: Optional[bool] = None
    if use_lockfile and lock_path.exists():
        lockfile = read_lockfile(lock_path)
        lockmap = lockfile.resolved
        lockfile_pwm = lockfile.pwm_source
        lockfile_site_kinds = lockfile.site_kinds
        lockfile_combine_sites = lockfile.combine_sites

    required = {tf for _, tf in list_targets(cfg)}
    if use_lockfile and lock_path.exists():
        missing = required - set(lockmap.keys())
        extra = set(lockmap.keys()) - required
        mismatch = None
        if not lockfile_pwm:
            mismatch = "Lockfile missing pwm_source; re-run `cruncher lock <config>`."
        elif lockfile_pwm != effective_pwm_source:
            mismatch = (
                f"Lockfile pwm_source='{lockfile_pwm}' does not match expected pwm_source='{effective_pwm_source}'. "
                "Re-run `cruncher lock <config>`."
            )
        if effective_pwm_source == "sites":
            if effective_site_kinds is not None and lockfile_site_kinds is not None:
                if sorted(effective_site_kinds) != sorted(lockfile_site_kinds):
                    mismatch = (
                        f"Lockfile site_kinds={lockfile_site_kinds} does not match expected site_kinds="
                        f"{effective_site_kinds}. "
                        "Re-run `cruncher lock <config>`."
                    )
            if lockfile_combine_sites is not None and lockfile_combine_sites != effective_combine_sites:
                mismatch = (
                    f"Lockfile combine_sites={lockfile_combine_sites} does not match expected combine_sites="
                    f"{effective_combine_sites}. "
                    "Re-run `cruncher lock <config>`."
                )
        if missing or extra or mismatch:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"extra: {', '.join(sorted(extra))}")
            if mismatch:
                details.append(mismatch)
            msg = " | ".join(details) if details else "Lockfile mismatch."
            return [
                TargetStatus(
                    set_index=set_index,
                    tf_name=tf,
                    source=None,
                    motif_id=None,
                    organism=None,
                    has_matrix=False,
                    has_sites=False,
                    site_count=0,
                    site_total=0,
                    site_kind=None,
                    dataset_id=None,
                    matrix_source=None,
                    pwm_source=effective_pwm_source,
                    status="mismatch-lock",
                    message=msg,
                )
                for set_index, tf in list_targets(cfg)
            ]

    catalog = CatalogIndex.load(catalog_root)
    statuses: list[TargetStatus] = []
    for set_index, tf in list_targets(cfg):
        entry = None
        locked = lockmap.get(tf) if use_lockfile else None
        if locked is not None:
            entry = catalog.entries.get(f"{locked.source}:{locked.motif_id}")
            if entry is None:
                statuses.append(
                    TargetStatus(
                        set_index=set_index,
                        tf_name=tf,
                        source=locked.source,
                        motif_id=locked.motif_id,
                        organism=None,
                        has_matrix=False,
                        has_sites=False,
                        site_count=0,
                        site_total=0,
                        site_kind=None,
                        dataset_id=None,
                        matrix_source=None,
                        pwm_source=effective_pwm_source,
                        status="missing-catalog",
                        message=f"Catalog entry missing for {locked.source}:{locked.motif_id}.",
                    )
                )
                continue
        elif use_lockfile:
            statuses.append(
                TargetStatus(
                    set_index=set_index,
                    tf_name=tf,
                    source=None,
                    motif_id=None,
                    organism=None,
                    has_matrix=False,
                    has_sites=False,
                    site_count=0,
                    site_total=0,
                    site_kind=None,
                    dataset_id=None,
                    matrix_source=None,
                    pwm_source=effective_pwm_source,
                    status="missing-lock",
                    message=f"Missing lock entry for '{tf}'. Run `cruncher lock {config_path.name}`.",
                )
            )
            continue
        else:
            try:
                entry = select_catalog_entry(
                    catalog=catalog,
                    tf_name=tf,
                    pwm_source=effective_pwm_source,
                    site_kinds=effective_site_kinds,
                    combine_sites=effective_combine_sites,
                    source_preference=cfg.motif_store.source_preference,
                    dataset_preference=cfg.motif_store.dataset_preference,
                    dataset_map=cfg.motif_store.dataset_map,
                    allow_ambiguous=cfg.motif_store.allow_ambiguous,
                )
            except ValueError as exc:
                statuses.append(
                    TargetStatus(
                        set_index=set_index,
                        tf_name=tf,
                        source=None,
                        motif_id=None,
                        organism=None,
                        has_matrix=False,
                        has_sites=False,
                        site_count=0,
                        site_total=0,
                        site_kind=None,
                        dataset_id=None,
                        matrix_source=None,
                        pwm_source=effective_pwm_source,
                        status="unresolved-target",
                        message=str(exc),
                    )
                )
                continue
        status_site_entries = [entry]
        if effective_combine_sites:
            status_site_entries = _site_entries_for_target(
                catalog=catalog,
                entry=entry,
                combine_sites=effective_combine_sites,
                site_kinds=effective_site_kinds,
            )
        site_count = 0
        site_total = 0
        site_kind = None
        dataset_id = None
        if status_site_entries:
            site_count = sum(candidate.site_count for candidate in status_site_entries)
            site_total = sum(candidate.site_total for candidate in status_site_entries)
            site_kind = _merge_text(candidate.site_kind for candidate in status_site_entries)
            dataset_id = _merge_text(candidate.dataset_id for candidate in status_site_entries)
        site_entries = status_site_entries

        if effective_pwm_source == "sites":
            if not site_entries:
                status = "missing-sites"
                message = "No cached binding-site sequences available."
                site_count = 0
                site_total = 0
                site_kind = None
                dataset_id = None
            else:
                if site_count < cfg.motif_store.min_sites_for_pwm:
                    msg = f"Only {site_count} sites available (min_sites_for_pwm={cfg.motif_store.min_sites_for_pwm})."
                    if cfg.motif_store.allow_low_sites:
                        status = "warning"
                        message = msg
                    else:
                        status = "insufficient-sites"
                        message = msg
                else:
                    status = "ready"
                    message = None
        else:
            status, message = _status_for_entry(
                entry=entry,
                pwm_source=effective_pwm_source,
                min_sites=cfg.motif_store.min_sites_for_pwm,
                allow_low_sites=cfg.motif_store.allow_low_sites,
            )

        if effective_pwm_source == "sites" and status in {"ready", "warning"} and site_total > site_count:
            missing = site_total - site_count
            seq_msg = (
                f"{missing} of {site_total} sites lack sequences. "
                "Configure ingest.genome_source=ncbi or pass --genome-fasta to hydrate sequences."
            )
            status = "missing-sequences"
            message = f"{message} {seq_msg}".strip() if message else seq_msg

        if status in {"ready", "warning"}:
            if effective_pwm_source == "matrix" and entry.has_matrix:
                motif_path = catalog_root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
                if not motif_path.exists():
                    status = "missing-matrix-file"
                    message = f"Motif file missing: {motif_path}"
            if effective_pwm_source == "sites":
                missing_paths = []
                for candidate in site_entries:
                    sites_path = (
                        catalog_root / "normalized" / "sites" / candidate.source / f"{candidate.motif_id}.jsonl"
                    )
                    if not sites_path.exists():
                        missing_paths.append(str(sites_path))
                if missing_paths:
                    status = "missing-sites-file"
                    message = f"Sites file missing: {missing_paths[0]}"
                elif _needs_window_length(entries=site_entries, window_lengths=effective_window_lengths):
                    status = "needs-window"
                    message = (
                        "Site lengths vary across cached entries; set motif_store.site_window_lengths for this TF "
                        "or each dataset before building PWMs."
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
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
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
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
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
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
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
