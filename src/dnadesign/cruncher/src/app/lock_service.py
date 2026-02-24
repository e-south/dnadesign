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
    allow_ambiguous_effective = allow_ambiguous or (combine_sites and pwm_source == "sites")

    def _combined_sites_checksum(tf_name: str, entries: List[CatalogEntry]) -> str:
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

    def _choose(candidates: List[CatalogEntry], *, tf_name: str) -> CatalogEntry:
        if not candidates:
            raise ValueError("No candidates available")
        if preference:
            preferred: list[CatalogEntry] = []
            for pref in preference:
                preferred.extend([entry for entry in candidates if entry.source == pref])
            if not preferred:
                raise ValueError("Multiple candidates found; no matching source_preference")
            candidates = preferred
        if len(candidates) == 1:
            return candidates[0]
        if dataset_map and tf_name in dataset_map:
            dataset_id = dataset_map[tf_name]
            filtered = [entry for entry in candidates if entry.dataset_id == dataset_id]
            if not filtered:
                options = ", ".join(f"{c.source}:{c.motif_id}" for c in candidates)
                raise ValueError(f"Dataset '{dataset_id}' not found for TF '{tf_name}'. Candidates: {options}.")
            candidates = filtered
            if len(candidates) == 1:
                return candidates[0]
        if dataset_preference:
            by_dataset = {entry.dataset_id: entry for entry in candidates if entry.dataset_id}
            for pref in dataset_preference:
                if pref in by_dataset:
                    return by_dataset[pref]
        if not allow_ambiguous_effective and not preference:
            raise ValueError("Multiple candidates found; set source_preference or allow_ambiguous")
        if preference:
            by_source = {entry.source: entry for entry in candidates}
            for pref in preference:
                if pref in by_source:
                    return by_source[pref]
        if allow_ambiguous_effective:
            return candidates[0]
        raise ValueError("Multiple candidates found; no matching source_preference")

    for name in names:
        candidates = catalog.list(tf_name=name, include_synonyms=True)
        if not candidates:
            if pwm_source == "sites":
                raise ValueError(f"No cached sites found for '{name}'. {cache_refresh_hint(pwm_source='sites')}")
            if pwm_source == "matrix":
                raise ValueError(f"No cached motifs found for '{name}'. {cache_refresh_hint(pwm_source='matrix')}")
            raise ValueError(f"No cached motifs or sites found for '{name}'. {cache_refresh_hint(pwm_source=None)}")
        if pwm_source:
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
                f"No cached data for '{name}' compatible with pwm_source='{pwm_source}'. "
                "Fetch motifs/sites or change pwm_source."
            )
        if len(candidates) > 1 and not allow_ambiguous_effective and not preference:
            options = ", ".join(f"{c.source}:{c.motif_id}" for c in candidates)
            raise ValueError(f"Ambiguous motif for '{name}'. Candidates: {options}")
        candidate = _choose(candidates, tf_name=name)
        checksum = ""
        motif_path = catalog_root / "normalized" / "motifs" / candidate.source / f"{candidate.motif_id}.json"
        sites_path = catalog_root / "normalized" / "sites" / candidate.source / f"{candidate.motif_id}.jsonl"
        if pwm_source == "sites":
            if combine_sites:
                entries = [
                    entry
                    for entry in catalog.entries.values()
                    if entry.tf_name.lower() == candidate.tf_name.lower() and entry.has_sites
                ]
                if site_kinds:
                    entries = [entry for entry in entries if entry.site_kind in site_kinds]
                checksum = _combined_sites_checksum(candidate.tf_name, entries)
            else:
                if not sites_path.exists():
                    raise ValueError(f"Missing sites cache for '{candidate.source}:{candidate.motif_id}'")
                checksum = sha256_path(sites_path)
        else:
            if not motif_path.exists():
                raise ValueError(f"Missing motif cache for '{candidate.source}:{candidate.motif_id}'")
            payload = json.loads(motif_path.read_text())
            checksum = payload.get("checksums", {}).get("sha256_norm", "")
            if not checksum:
                checksum = sha256_path(motif_path)
        resolved[name] = LockEntry(
            name=name,
            source=candidate.source,
            motif_id=candidate.motif_id,
            sha256=checksum,
            dataset_id=candidate.dataset_id,
        )
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
    return resolved
