"""Lockfile utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.utils.hashing import sha256_lines, sha256_path


@dataclass(frozen=True)
class LockedMotif:
    source: str
    motif_id: str
    sha256: str
    dataset_id: Optional[str] = None


@dataclass(frozen=True)
class Lockfile:
    resolved: dict[str, LockedMotif]
    pwm_source: Optional[str]
    site_kinds: Optional[list[str]]
    combine_sites: Optional[bool]
    generated_at: Optional[str]


def read_lockfile(path: Path) -> Lockfile:
    payload = json.loads(path.read_text())
    resolved = {}
    for name, data in payload.get("resolved", {}).items():
        resolved[name] = LockedMotif(
            source=data["source"],
            motif_id=data["motif_id"],
            sha256=data.get("sha256", ""),
            dataset_id=data.get("dataset_id"),
        )
    return Lockfile(
        resolved=resolved,
        pwm_source=payload.get("pwm_source"),
        site_kinds=payload.get("site_kinds"),
        combine_sites=payload.get("combine_sites"),
        generated_at=payload.get("generated_at"),
    )


def load_lockfile(path: Path) -> dict[str, LockedMotif]:
    return read_lockfile(path).resolved


def validate_lockfile(
    lockfile: Lockfile,
    *,
    expected_pwm_source: str,
    expected_site_kinds: Optional[list[str]] = None,
    expected_combine_sites: Optional[bool] = None,
    required_tfs: Iterable[str],
) -> None:
    if not lockfile.pwm_source:
        raise ValueError("Lockfile is missing pwm_source metadata. Re-run `cruncher lock <config>`.")
    if lockfile.pwm_source != expected_pwm_source:
        raise ValueError(
            f"Lockfile pwm_source='{lockfile.pwm_source}' does not match config pwm_source='{expected_pwm_source}'. "
            "Re-run `cruncher lock <config>`."
        )
    if expected_pwm_source == "sites":
        if expected_site_kinds is not None and lockfile.site_kinds is not None:
            if sorted(lockfile.site_kinds) != sorted(expected_site_kinds):
                raise ValueError(
                    f"Lockfile site_kinds={lockfile.site_kinds} does not match config site_kinds="
                    f"{expected_site_kinds}. "
                    "Re-run `cruncher lock <config>`."
                )
        if expected_combine_sites is not None and lockfile.combine_sites is not None:
            if lockfile.combine_sites != expected_combine_sites:
                raise ValueError(
                    f"Lockfile combine_sites={lockfile.combine_sites} does not match config combine_sites="
                    f"{expected_combine_sites}. "
                    "Re-run `cruncher lock <config>`."
                )
    required = set(required_tfs)
    resolved = set(lockfile.resolved.keys())
    missing = required - resolved
    extra = resolved - required
    if missing:
        raise ValueError(f"Lockfile missing TFs: {', '.join(sorted(missing))}. Re-run `cruncher lock <config>`.")
    if extra:
        raise ValueError(
            f"Lockfile contains TFs not in config: {', '.join(sorted(extra))}. Re-run `cruncher lock <config>`."
        )


def verify_lockfile_hashes(
    *,
    lockfile: Lockfile,
    catalog_root: Path,
    expected_pwm_source: str,
) -> None:
    catalog = CatalogIndex.load(catalog_root)
    for tf_name, locked in lockfile.resolved.items():
        entry = catalog.entries.get(f"{locked.source}:{locked.motif_id}")
        if entry is None:
            raise ValueError(f"Catalog entry missing for {locked.source}:{locked.motif_id}")
        if expected_pwm_source == "matrix":
            motif_path = catalog_root / "normalized" / "motifs" / locked.source / f"{locked.motif_id}.json"
            if not motif_path.exists():
                raise FileNotFoundError(f"Missing motif cache file: {motif_path}")
            payload = json.loads(motif_path.read_text())
            checksum = payload.get("checksums", {}).get("sha256_norm") or sha256_path(motif_path)
            if locked.sha256 and checksum != locked.sha256:
                raise ValueError(
                    f"Lockfile checksum mismatch for {tf_name} ({locked.source}:{locked.motif_id}). "
                    "Re-run `cruncher lock <config>`."
                )
        elif expected_pwm_source == "sites":
            if lockfile.combine_sites:
                entries = [
                    candidate
                    for candidate in catalog.entries.values()
                    if candidate.tf_name.lower() == entry.tf_name.lower() and candidate.has_sites
                ]
                if lockfile.site_kinds:
                    entries = [candidate for candidate in entries if candidate.site_kind in lockfile.site_kinds]
                if not entries:
                    raise ValueError(
                        f"No cached site entries found for TF '{entry.tf_name}'. "
                        "Fetch sites or adjust motif_store.site_kinds."
                    )
                lines = []
                for candidate in sorted(entries, key=lambda e: (e.source, e.motif_id)):
                    sites_path = (
                        catalog_root / "normalized" / "sites" / candidate.source / f"{candidate.motif_id}.jsonl"
                    )
                    if not sites_path.exists():
                        raise FileNotFoundError(f"Missing sites cache file: {sites_path}")
                    lines.append(f"{candidate.source}:{candidate.motif_id}:{sha256_path(sites_path)}")
                checksum = sha256_lines(lines)
            else:
                sites_path = catalog_root / "normalized" / "sites" / locked.source / f"{locked.motif_id}.jsonl"
                if not sites_path.exists():
                    raise FileNotFoundError(f"Missing sites cache file: {sites_path}")
                checksum = sha256_path(sites_path)
            if locked.sha256 and checksum != locked.sha256:
                raise ValueError(
                    f"Lockfile checksum mismatch for {tf_name} ({locked.source}:{locked.motif_id}). "
                    "Re-run `cruncher lock <config>`."
                )
            if locked.dataset_id and entry.dataset_id and locked.dataset_id != entry.dataset_id:
                raise ValueError(
                    f"Lockfile dataset_id='{locked.dataset_id}' does not match cached entry dataset_id="
                    f"'{entry.dataset_id}'. "
                    "Re-run `cruncher lock <config>`."
                )
        else:
            raise ValueError("pwm_source must be 'matrix' or 'sites'")
