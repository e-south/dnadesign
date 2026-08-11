"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/economics/contracts.py

Define typed contracts for dated synthesis-price snapshots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class GeneFragmentBand:
    minimum_length_nt: int
    maximum_length_nt: int
    with_adapters_usd_per_bp: Decimal
    without_adapters_usd_per_bp: Decimal

    def __post_init__(self) -> None:
        _positive_span(self.minimum_length_nt, self.maximum_length_nt, label="gene-fragment length")
        if self.with_adapters_usd_per_bp <= 0 or self.without_adapters_usd_per_bp <= 0:
            raise ValueError("gene-fragment rates must be positive")

    def contains(self, length_nt: int) -> bool:
        return self.minimum_length_nt <= length_nt <= self.maximum_length_nt


@dataclass(frozen=True, slots=True)
class OligoLengthBand:
    band_id: str
    minimum_length_nt: int
    maximum_length_nt: int

    def __post_init__(self) -> None:
        if not self.band_id.strip():
            raise ValueError("oligo length-band id must not be empty")
        _positive_span(self.minimum_length_nt, self.maximum_length_nt, label="oligo length")

    def contains(self, minimum_length_nt: int, maximum_length_nt: int) -> bool:
        return self.minimum_length_nt <= minimum_length_nt <= maximum_length_nt <= self.maximum_length_nt


@dataclass(frozen=True, slots=True)
class OligoPoolTier:
    minimum_oligos: int
    maximum_oligos: int
    prices_usd: dict[str, Decimal]

    def __post_init__(self) -> None:
        _positive_span(self.minimum_oligos, self.maximum_oligos, label="oligo-count tier")
        if not self.prices_usd or any(not key.strip() or value <= 0 for key, value in self.prices_usd.items()):
            raise ValueError("oligo-pool tier prices must be positive and keyed by length band")

    def contains(self, oligo_count: int) -> bool:
        return self.minimum_oligos <= oligo_count <= self.maximum_oligos


@dataclass(frozen=True, slots=True)
class PricingSnapshot:
    supplier: str
    price_context: str
    currency: str
    retrieved_on: date
    source_url: str
    n_nucleotide_surcharge_fraction: Decimal
    gene_fragment_bands: tuple[GeneFragmentBand, ...]
    oligo_length_bands: tuple[OligoLengthBand, ...]
    oligo_pool_tiers: tuple[OligoPoolTier, ...]

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.supplier, self.price_context, self.currency, self.source_url)):
            raise ValueError("pricing snapshot identity fields must not be empty")
        if self.currency != "USD":
            raise ValueError("pricing snapshot must use USD because price fields are USD-denominated")
        if self.n_nucleotide_surcharge_fraction < 0:
            raise ValueError("N-nucleotide surcharge must be non-negative")
        if not self.gene_fragment_bands or not self.oligo_length_bands or not self.oligo_pool_tiers:
            raise ValueError("pricing snapshot bands and tiers must not be empty")
        _require_non_overlapping_bands(self.gene_fragment_bands, label="gene-fragment")
        _require_non_overlapping_bands(self.oligo_length_bands, label="oligo")
        expected_minimum = self.oligo_pool_tiers[0].minimum_oligos
        declared_band_ids = [band.band_id for band in self.oligo_length_bands]
        band_ids = set(declared_band_ids)
        if len(band_ids) != len(declared_band_ids):
            raise ValueError("oligo length-band ids must be unique")
        for tier in self.oligo_pool_tiers:
            if tier.minimum_oligos != expected_minimum:
                raise ValueError("oligo-pool tiers must be contiguous")
            if set(tier.prices_usd) != band_ids:
                raise ValueError("every oligo-pool tier must price every declared length band")
            expected_minimum = tier.maximum_oligos + 1

    @property
    def minimum_oligos_per_pool(self) -> int:
        return self.oligo_pool_tiers[0].minimum_oligos

    @property
    def maximum_oligos_per_pool(self) -> int:
        return self.oligo_pool_tiers[-1].maximum_oligos

    def gene_fragment_band(self, length_nt: int) -> GeneFragmentBand:
        matches = [band for band in self.gene_fragment_bands if band.contains(length_nt)]
        if len(matches) != 1:
            raise ValueError(f"target length {length_nt} nt must match exactly one gene-fragment price band")
        return matches[0]

    def oligo_length_band(self, minimum_length_nt: int, maximum_length_nt: int) -> OligoLengthBand:
        matches = [band for band in self.oligo_length_bands if band.contains(minimum_length_nt, maximum_length_nt)]
        if len(matches) != 1:
            raise ValueError("physical oligo length range must fit exactly one price band")
        return matches[0]

    def oligo_pool_price(self, *, oligo_count: int, length_band_id: str) -> Decimal:
        matches = [tier for tier in self.oligo_pool_tiers if tier.contains(oligo_count)]
        if len(matches) != 1:
            raise ValueError(f"oligo count {oligo_count} is outside the pricing snapshot")
        return matches[0].prices_usd[length_band_id]


@dataclass(frozen=True, slots=True)
class SynthesisScenario:
    target_length_nt: int
    oligos_per_target: int
    minimum_oligo_length_nt: int
    maximum_oligo_length_nt: int
    uses_n_nucleotide: bool = False

    def __post_init__(self) -> None:
        if self.target_length_nt <= 0 or self.oligos_per_target <= 0:
            raise ValueError("target length and oligos per target must be positive")
        if not isinstance(self.uses_n_nucleotide, bool):
            raise ValueError("uses_n_nucleotide must be a boolean")
        _positive_span(self.minimum_oligo_length_nt, self.maximum_oligo_length_nt, label="physical oligo length")


def load_pricing_snapshot(path: Path) -> PricingSnapshot:
    """Load and validate one pricing snapshot."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "junction.synthesis-pricing-snapshot/v1":
        raise ValueError("unsupported synthesis pricing snapshot schema")
    gene_bands = _list_of_mappings(raw, "gene_fragment_bands")
    oligo_bands = _list_of_mappings(raw, "oligo_length_bands")
    tiers = _list_of_mappings(raw, "oligo_pool_tiers")
    retrieved = raw.get("retrieved_on")
    return PricingSnapshot(
        supplier=_text(raw, "supplier"),
        price_context=_text(raw, "price_context"),
        currency=_text(raw, "currency"),
        retrieved_on=retrieved if isinstance(retrieved, date) else date.fromisoformat(str(retrieved)),
        source_url=_text(raw, "source_url"),
        n_nucleotide_surcharge_fraction=Decimal(str(_required(raw, "n_nucleotide_surcharge_fraction"))),
        gene_fragment_bands=tuple(
            GeneFragmentBand(
                minimum_length_nt=int(_required(item, "minimum_length_nt")),
                maximum_length_nt=int(_required(item, "maximum_length_nt")),
                with_adapters_usd_per_bp=Decimal(str(_required(item, "with_adapters_usd_per_bp"))),
                without_adapters_usd_per_bp=Decimal(str(_required(item, "without_adapters_usd_per_bp"))),
            )
            for item in gene_bands
        ),
        oligo_length_bands=tuple(
            OligoLengthBand(
                band_id=_text(item, "id"),
                minimum_length_nt=int(_required(item, "minimum_length_nt")),
                maximum_length_nt=int(_required(item, "maximum_length_nt")),
            )
            for item in oligo_bands
        ),
        oligo_pool_tiers=tuple(
            OligoPoolTier(
                minimum_oligos=int(_required(item, "minimum_oligos")),
                maximum_oligos=int(_required(item, "maximum_oligos")),
                prices_usd={str(key): Decimal(str(value)) for key, value in _mapping(item, "prices_usd").items()},
            )
            for item in tiers
        ),
    )


def default_pricing_snapshot_path() -> Path:
    return Path(__file__).with_name("data") / "twist-academic-2026-08-11.yaml"


def _positive_span(minimum: int, maximum: int, *, label: str) -> None:
    if minimum <= 0 or maximum <= 0 or minimum > maximum:
        raise ValueError(f"{label} span must be positive and ordered")


def _require_non_overlapping_bands(bands: tuple[Any, ...], *, label: str) -> None:
    ordered = sorted(bands, key=lambda band: band.minimum_length_nt)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.minimum_length_nt <= previous.maximum_length_nt:
            raise ValueError(f"{label} length bands must not overlap")


def _required(parent: dict[str, Any], key: str) -> Any:
    if key not in parent:
        raise ValueError(f"missing required pricing field: {key}")
    return parent[key]


def _text(parent: dict[str, Any], key: str) -> str:
    value = _required(parent, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pricing field {key} must be a non-empty string")
    return value.strip()


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(parent, key)
    if not isinstance(value, dict):
        raise ValueError(f"pricing field {key} must be a mapping")
    return value


def _list_of_mappings(parent: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = _required(parent, key)
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"pricing field {key} must be a non-empty list of mappings")
    return value


__all__ = [
    "GeneFragmentBand",
    "OligoLengthBand",
    "OligoPoolTier",
    "PricingSnapshot",
    "SynthesisScenario",
    "default_pricing_snapshot_path",
    "load_pricing_snapshot",
]
