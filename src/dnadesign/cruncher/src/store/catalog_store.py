"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/store/catalog_store.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.pwm_window import select_pwm_window
from dnadesign.cruncher.ingest.models import MotifDescriptor, MotifQuery
from dnadesign.cruncher.ingest.normalize import compute_pwm_from_sites
from dnadesign.cruncher.ingest.site_windows import (
    resolve_window_length,
    window_sequence,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex, build_descriptor
from dnadesign.cruncher.store.motif_store import MotifRef, MotifStore


def _parse_int_tag(tags: dict[str, object], key: str) -> int | None:
    value = tags.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_nsites(payload: dict[str, object], entry: CatalogEntry | None) -> int | None:
    tags: dict[str, object] = {}
    descriptor = payload.get("descriptor")
    if isinstance(descriptor, dict):
        tags.update(descriptor.get("tags") or {})
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        tags.update(provenance.get("tags") or {})
    for key in ("discovery_nsites", "meme_nsites", "site_count", "nsites"):
        parsed = _parse_int_tag(tags, key)
        if parsed is not None:
            return parsed
    matrix_source = tags.get("matrix_source")
    if matrix_source == "sites" and entry is not None and entry.site_count:
        return entry.site_count
    return None


def iter_site_sequences(
    *,
    root: Path,
    entries: list[CatalogEntry],
    site_window_lengths: dict[str, int],
    site_window_center: str,
    allow_variable_lengths: bool = False,
) -> Iterable[str]:
    for site_entry in entries:
        sites_path = root / "normalized" / "sites" / site_entry.source / f"{site_entry.motif_id}.jsonl"
        if not sites_path.exists():
            raise FileNotFoundError(f"Site records not found: {sites_path}")
        window_length = resolve_window_length(
            tf_name=site_entry.tf_name,
            dataset_id=site_entry.dataset_id,
            window_lengths=site_window_lengths,
        )
        if site_window_center == "summit":
            raise ValueError(
                "summit-centered windows require per-site summit metadata; "
                "use site_window_center='midpoint' or supply summit-aware sequences."
            )
        lengths: set[int] = set()
        with sites_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload = json.loads(line)
                seq = payload.get("sequence")
                if not seq:
                    site_id = payload.get("site_id") or "unknown"
                    motif_ref = payload.get("motif_ref") or f"{site_entry.source}:{site_entry.motif_id}"
                    raise ValueError(
                        "Cached site record is missing a sequence for "
                        f"TF '{site_entry.tf_name}' (motif={motif_ref}, site_id={site_id}). "
                        "Run `cruncher fetch sites --hydrate <config>` or re-fetch with "
                        "ingest.genome_source/--genome-fasta to hydrate sequences."
                    )
                if window_length is None:
                    lengths.add(len(seq))
                    if not allow_variable_lengths and len(lengths) > 1:
                        raise ValueError(
                            f"Site lengths vary for TF '{site_entry.tf_name}'. "
                            "Set motif_store.site_window_lengths to build a PWM."
                        )
                else:
                    seq = window_sequence(seq, window_length, center=site_window_center)
                yield seq


class CatalogMotifStore(MotifStore):
    def __init__(
        self,
        root: Path,
        *,
        pwm_source: str = "matrix",
        site_kinds: list[str] | None = None,
        combine_sites: bool = False,
        site_window_lengths: dict[str, int] | None = None,
        site_window_center: str = "midpoint",
        pwm_window_lengths: dict[str, int] | None = None,
        pwm_window_strategy: str = "max_info",
        min_sites_for_pwm: int = 2,
        allow_low_sites: bool = False,
        pseudocounts: float = 0.5,
    ) -> None:
        self.root = root
        self.pwm_source = pwm_source
        self.site_kinds = site_kinds
        self.combine_sites = combine_sites
        self.site_window_lengths = site_window_lengths or {}
        self.site_window_center = site_window_center
        self.pwm_window_lengths = pwm_window_lengths or {}
        self.pwm_window_strategy = pwm_window_strategy
        self.min_sites_for_pwm = min_sites_for_pwm
        self.allow_low_sites = allow_low_sites
        self.pseudocounts = pseudocounts

    def list(self, query: MotifQuery) -> list[MotifDescriptor]:
        catalog = CatalogIndex.load(self.root)
        org = None
        if query.organism is not None:
            org = {
                "taxon": query.organism.taxon,
                "name": query.organism.name,
                "strain": query.organism.strain,
                "assembly": query.organism.assembly,
            }
        entries = catalog.list(tf_name=query.tf_name, source=query.source, organism=org)
        return [build_descriptor(entry) for entry in entries]

    def get_pwm(self, ref: MotifRef) -> PWM:
        source = ref.source
        if source == "catalog":
            # default namespace if caller doesn't specify
            raise ValueError("catalog MotifRef requires explicit source namespace")
        if self.pwm_source == "matrix":
            norm_path = self.root / "normalized" / "motifs" / source / f"{ref.motif_id}.json"
            if not norm_path.exists():
                raise FileNotFoundError(f"Motif record not found: {norm_path}")
            payload = json.loads(norm_path.read_text())
            matrix = np.array(payload["matrix"], dtype=float)
            log_odds = payload.get("log_odds_matrix")
            log_odds_matrix = np.array(log_odds, dtype=float) if log_odds is not None else None
            catalog = CatalogIndex.load(self.root)
            entry = catalog.entries.get(f"{ref.source}:{ref.motif_id}")
            nsites = _extract_nsites(payload, entry)
            pwm = PWM(
                name=payload["descriptor"]["tf_name"],
                matrix=matrix,
                nsites=nsites,
                log_odds_matrix=log_odds_matrix,
            )
            if entry is None:
                return pwm
            return self._apply_pwm_window(pwm, entry)
        if self.pwm_source == "sites":
            catalog = CatalogIndex.load(self.root)
            entry = catalog.entries.get(f"{ref.source}:{ref.motif_id}")
            if entry is None:
                raise FileNotFoundError(f"Catalog entry not found for {ref.source}:{ref.motif_id}")

            entries = [entry]
            if self.combine_sites:
                entries = [
                    e for e in catalog.entries.values() if e.tf_name.lower() == entry.tf_name.lower() and e.has_sites
                ]
            if self.site_kinds is not None:
                entries = [e for e in entries if e.site_kind in self.site_kinds]
            if not entries:
                raise ValueError(
                    f"No site entries available for TF '{entry.tf_name}' with site_kinds={self.site_kinds}."
                )
            matrix, site_count = compute_pwm_from_sites(
                iter_site_sequences(
                    root=self.root,
                    entries=entries,
                    site_window_lengths=self.site_window_lengths,
                    site_window_center=self.site_window_center,
                ),
                min_sites=self.min_sites_for_pwm,
                return_count=True,
                strict_min_sites=not self.allow_low_sites,
                pseudocounts=self.pseudocounts,
            )
            matrix = np.array(matrix, dtype=float)
            tf_name = entry.tf_name
            pwm = PWM(name=tf_name, matrix=matrix, nsites=site_count)
            return self._apply_pwm_window(pwm, entry)
        raise ValueError("pwm_source must be 'matrix' or 'sites'")

    def _apply_pwm_window(self, pwm: PWM, entry: CatalogEntry) -> PWM:
        window_length = resolve_window_length(
            tf_name=entry.tf_name,
            dataset_id=entry.dataset_id,
            window_lengths=self.pwm_window_lengths,
        )
        if window_length is None:
            return pwm
        return select_pwm_window(pwm, length=window_length, strategy=self.pwm_window_strategy)
