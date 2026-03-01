"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/lock_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dnadesign.cruncher.app.cache_readiness import cache_refresh_hint
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.utils.hashing import sha256_lines, sha256_path


@dataclass(frozen=True)
class LockEntry:
    name: str
    source: str
    motif_id: str
    sha256: str
    dataset_id: Optional[str] = None


def _allow_ambiguous_effective(
    *, allow_ambiguous: bool, combine_sites: Optional[bool], pwm_source: Optional[str]
) -> bool:
    return bool(allow_ambiguous or (combine_sites and pwm_source == "sites"))


def _resolve_initial_candidates(
    *,
    catalog: CatalogIndex,
    tf_name: str,
    pwm_source: Optional[str],
) -> List[CatalogEntry]:
    candidates = catalog.list(tf_name=tf_name, include_synonyms=True)
    if candidates:
        return candidates
    if pwm_source == "sites":
        raise ValueError(f"No cached sites found for '{tf_name}'. {cache_refresh_hint(pwm_source='sites')}")
    if pwm_source == "matrix":
        raise ValueError(f"No cached motifs found for '{tf_name}'. {cache_refresh_hint(pwm_source='matrix')}")
    raise ValueError(f"No cached motifs or sites found for '{tf_name}'. {cache_refresh_hint(pwm_source=None)}")


def _filter_candidates_for_pwm_source(
    *,
    candidates: List[CatalogEntry],
    pwm_source: Optional[str],
    site_kinds: Optional[List[str]],
) -> List[CatalogEntry]:
    filtered = list(candidates)
    if not pwm_source:
        return filtered
    if pwm_source == "matrix":
        return [entry for entry in filtered if entry.has_matrix]
    if pwm_source == "sites":
        filtered = [entry for entry in filtered if entry.has_sites]
        if site_kinds:
            filtered = [entry for entry in filtered if entry.site_kind in site_kinds]
        return filtered
    raise ValueError("pwm_source must be 'matrix' or 'sites'")


def _choose_candidate(
    *,
    candidates: List[CatalogEntry],
    tf_name: str,
    source_preference: List[str],
    dataset_preference: Optional[List[str]],
    dataset_map: Optional[Dict[str, str]],
    allow_ambiguous_effective: bool,
) -> CatalogEntry:
    if not candidates:
        raise ValueError("No candidates available")
    selected = _apply_source_preference(candidates=list(candidates), source_preference=source_preference)
    if len(selected) == 1:
        return selected[0]
    selected = _apply_dataset_map_selection(selected=selected, tf_name=tf_name, dataset_map=dataset_map)
    if len(selected) == 1:
        return selected[0]
    preferred_dataset = _pick_by_dataset_preference(selected=selected, dataset_preference=dataset_preference)
    if preferred_dataset is not None:
        return preferred_dataset
    if not allow_ambiguous_effective and not source_preference:
        raise ValueError("Multiple candidates found; set source_preference or allow_ambiguous")
    preferred_source = _pick_by_source_preference(selected=selected, source_preference=source_preference)
    if preferred_source is not None:
        return preferred_source
    if allow_ambiguous_effective:
        return selected[0]
    raise ValueError("Multiple candidates found; no matching source_preference")


def _apply_source_preference(*, candidates: List[CatalogEntry], source_preference: List[str]) -> List[CatalogEntry]:
    if not source_preference:
        return list(candidates)
    preferred: list[CatalogEntry] = []
    for pref in source_preference:
        preferred.extend([entry for entry in candidates if entry.source == pref])
    if not preferred:
        raise ValueError("Multiple candidates found; no matching source_preference")
    return preferred


def _apply_dataset_map_selection(
    *,
    selected: List[CatalogEntry],
    tf_name: str,
    dataset_map: Optional[Dict[str, str]],
) -> List[CatalogEntry]:
    if not dataset_map or tf_name not in dataset_map:
        return list(selected)
    dataset_id = dataset_map[tf_name]
    filtered = [entry for entry in selected if entry.dataset_id == dataset_id]
    if filtered:
        return filtered
    options = ", ".join(f"{c.source}:{c.motif_id}" for c in selected)
    raise ValueError(f"Dataset '{dataset_id}' not found for TF '{tf_name}'. Candidates: {options}.")


def _pick_by_dataset_preference(
    *,
    selected: List[CatalogEntry],
    dataset_preference: Optional[List[str]],
) -> CatalogEntry | None:
    if not dataset_preference:
        return None
    by_dataset = {entry.dataset_id: entry for entry in selected if entry.dataset_id}
    for pref in dataset_preference:
        if pref in by_dataset:
            return by_dataset[pref]
    return None


def _pick_by_source_preference(
    *,
    selected: List[CatalogEntry],
    source_preference: List[str],
) -> CatalogEntry | None:
    if not source_preference:
        return None
    by_source = {entry.source: entry for entry in selected}
    for pref in source_preference:
        if pref in by_source:
            return by_source[pref]
    return None


def _combined_sites_checksum(
    *,
    catalog: CatalogIndex,
    catalog_root: Path,
    tf_name: str,
    site_kinds: Optional[List[str]],
) -> str:
    entries = [
        entry for entry in catalog.entries.values() if entry.tf_name.lower() == tf_name.lower() and entry.has_sites
    ]
    if site_kinds:
        entries = [entry for entry in entries if entry.site_kind in site_kinds]
    if not entries:
        raise ValueError(
            f"No cached site entries found for TF '{tf_name}'. Fetch sites or adjust cruncher.catalog.site_kinds."
        )
    lines: list[str] = []
    for entry in sorted(entries, key=lambda e: (e.source, e.motif_id)):
        sites_path = catalog_root / "normalized" / "sites" / entry.source / f"{entry.motif_id}.jsonl"
        if not sites_path.exists():
            raise ValueError(f"Missing sites cache for '{entry.source}:{entry.motif_id}'")
        lines.append(f"{entry.source}:{entry.motif_id}:{sha256_path(sites_path)}")
    return sha256_lines(lines)


def _resolve_candidate_checksum(
    *,
    candidate: CatalogEntry,
    catalog: CatalogIndex,
    catalog_root: Path,
    pwm_source: Optional[str],
    combine_sites: Optional[bool],
    site_kinds: Optional[List[str]],
) -> str:
    motif_path = catalog_root / "normalized" / "motifs" / candidate.source / f"{candidate.motif_id}.json"
    sites_path = catalog_root / "normalized" / "sites" / candidate.source / f"{candidate.motif_id}.jsonl"
    if pwm_source == "sites":
        if combine_sites:
            return _combined_sites_checksum(
                catalog=catalog,
                catalog_root=catalog_root,
                tf_name=candidate.tf_name,
                site_kinds=site_kinds,
            )
        if not sites_path.exists():
            raise ValueError(f"Missing sites cache for '{candidate.source}:{candidate.motif_id}'")
        return sha256_path(sites_path)
    if not motif_path.exists():
        raise ValueError(f"Missing motif cache for '{candidate.source}:{candidate.motif_id}'")
    payload = json.loads(motif_path.read_text())
    checksum = payload.get("checksums", {}).get("sha256_norm", "")
    if checksum:
        return checksum
    return sha256_path(motif_path)


def _write_lock_payload(
    *,
    resolved: Dict[str, LockEntry],
    pwm_source: Optional[str],
    site_kinds: Optional[List[str]],
    combine_sites: Optional[bool],
    lock_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pwm_source": pwm_source,
        "site_kinds": site_kinds,
        "combine_sites": combine_sites,
        "resolved": {
            k: {
                "source": v.source,
                "motif_id": v.motif_id,
                "sha256": v.sha256,
                "dataset_id": v.dataset_id,
            }
            for k, v in resolved.items()
        },
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2))


def resolve_lock(
    *,
    names: Iterable[str],
    catalog_root: Path,
    source_preference: Optional[List[str]] = None,
    allow_ambiguous: bool = False,
    pwm_source: Optional[str] = None,
    site_kinds: Optional[List[str]] = None,
    combine_sites: Optional[bool] = None,
    dataset_preference: Optional[List[str]] = None,
    dataset_map: Optional[Dict[str, str]] = None,
    lock_path: Path,
) -> Dict[str, LockEntry]:
    resolved: Dict[str, LockEntry] = {}
    catalog = CatalogIndex.load(catalog_root)
    preference = source_preference or []
    allow_ambiguous_effective = _allow_ambiguous_effective(
        allow_ambiguous=allow_ambiguous,
        combine_sites=combine_sites,
        pwm_source=pwm_source,
    )

    for name in names:
        candidates = _resolve_initial_candidates(catalog=catalog, tf_name=name, pwm_source=pwm_source)
        candidates = _filter_candidates_for_pwm_source(
            candidates=candidates,
            pwm_source=pwm_source,
            site_kinds=site_kinds,
        )
        if not candidates:
            raise ValueError(
                f"No cached data for '{name}' compatible with pwm_source='{pwm_source}'. "
                "Fetch motifs/sites or change pwm_source."
            )
        if len(candidates) > 1 and not allow_ambiguous_effective and not preference:
            options = ", ".join(f"{c.source}:{c.motif_id}" for c in candidates)
            raise ValueError(f"Ambiguous motif for '{name}'. Candidates: {options}")
        candidate = _choose_candidate(
            candidates=candidates,
            tf_name=name,
            source_preference=preference,
            dataset_preference=dataset_preference,
            dataset_map=dataset_map,
            allow_ambiguous_effective=allow_ambiguous_effective,
        )
        checksum = _resolve_candidate_checksum(
            candidate=candidate,
            catalog=catalog,
            catalog_root=catalog_root,
            pwm_source=pwm_source,
            combine_sites=combine_sites,
            site_kinds=site_kinds,
        )
        resolved[name] = LockEntry(
            name=name,
            source=candidate.source,
            motif_id=candidate.motif_id,
            sha256=checksum,
            dataset_id=candidate.dataset_id,
        )
    _write_lock_payload(
        resolved=resolved,
        pwm_source=pwm_source,
        site_kinds=site_kinds,
        combine_sites=combine_sites,
        lock_path=lock_path,
    )
    return resolved
