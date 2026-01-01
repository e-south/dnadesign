"""Catalog-backed MotifStore implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.ingest.models import MotifDescriptor, MotifQuery
from dnadesign.cruncher.ingest.normalize import compute_pwm_from_sites
from dnadesign.cruncher.ingest.site_windows import resolve_window_length, window_sequence
from dnadesign.cruncher.store.catalog_index import CatalogIndex, build_descriptor
from dnadesign.cruncher.store.motif_store import MotifRef, MotifStore


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
        min_sites_for_pwm: int = 2,
        allow_low_sites: bool = False,
    ) -> None:
        self.root = root
        self.pwm_source = pwm_source
        self.site_kinds = site_kinds
        self.combine_sites = combine_sites
        self.site_window_lengths = site_window_lengths or {}
        self.site_window_center = site_window_center
        self.min_sites_for_pwm = min_sites_for_pwm
        self.allow_low_sites = allow_low_sites

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
            return PWM(name=payload["descriptor"]["tf_name"], matrix=matrix)
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

            def _iter_sequences() -> Iterable[str]:
                for site_entry in entries:
                    sites_path = self.root / "normalized" / "sites" / site_entry.source / f"{site_entry.motif_id}.jsonl"
                    if not sites_path.exists():
                        raise FileNotFoundError(f"Site records not found: {sites_path}")
                    window_length = resolve_window_length(
                        tf_name=site_entry.tf_name,
                        dataset_id=site_entry.dataset_id,
                        window_lengths=self.site_window_lengths,
                    )
                    if self.site_window_center == "summit":
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
                                continue
                            if window_length is None:
                                lengths.add(len(seq))
                                if len(lengths) > 1:
                                    raise ValueError(
                                        f"Site lengths vary for TF '{site_entry.tf_name}'. "
                                        "Set motif_store.site_window_lengths to build a PWM."
                                    )
                            else:
                                seq = window_sequence(seq, window_length, center=self.site_window_center)
                            yield seq

            matrix, site_count = compute_pwm_from_sites(
                _iter_sequences(),
                min_sites=self.min_sites_for_pwm,
                return_count=True,
                strict_min_sites=not self.allow_low_sites,
            )
            matrix = np.array(matrix, dtype=float)
            tf_name = entry.tf_name
            return PWM(name=tf_name, matrix=matrix, nsites=site_count)
        raise ValueError("pwm_source must be 'matrix' or 'sites'")
