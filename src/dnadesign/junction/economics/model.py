"""Calculate purchase-price curves from a dated snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .contracts import PricingSnapshot, SynthesisScenario


@dataclass(frozen=True, slots=True)
class PurchasePriceRow:
    target_count: int
    gene_fragments_usd: Decimal
    oligo_pool_usd: Decimal
    oligo_count: int


def build_purchase_price_rows(
    snapshot: PricingSnapshot,
    scenario: SynthesisScenario,
) -> tuple[PurchasePriceRow, ...]:
    """Price every complete target count supported by the oligo-pool snapshot."""

    gene_band = snapshot.gene_fragment_band(scenario.target_length_nt)
    oligo_band = snapshot.oligo_length_band(
        scenario.minimum_oligo_length_nt,
        scenario.maximum_oligo_length_nt,
    )
    minimum_targets = (snapshot.minimum_oligos_per_pool + scenario.oligos_per_target - 1) // scenario.oligos_per_target
    maximum_targets = snapshot.maximum_oligos_per_pool // scenario.oligos_per_target
    if minimum_targets > maximum_targets:
        raise ValueError("oligo-pool snapshot cannot hold one complete target")
    rows: list[PurchasePriceRow] = []
    gene_price_per_target = Decimal(scenario.target_length_nt) * gene_band.with_adapters_usd_per_bp
    for target_count in range(minimum_targets, maximum_targets + 1):
        oligo_count = target_count * scenario.oligos_per_target
        pool_price = snapshot.oligo_pool_price(
            oligo_count=oligo_count,
            length_band_id=oligo_band.band_id,
        )
        if scenario.uses_n_nucleotide:
            pool_price *= Decimal("1") + snapshot.n_nucleotide_surcharge_fraction
        rows.append(
            PurchasePriceRow(
                target_count=target_count,
                gene_fragments_usd=Decimal(target_count) * gene_price_per_target,
                oligo_pool_usd=pool_price,
                oligo_count=oligo_count,
            )
        )
    return tuple(rows)


def stable_oligo_pool_advantage(rows: tuple[PurchasePriceRow, ...]) -> int | None:
    """Return the first target count after which the oligo pool stays cheaper."""

    if not rows:
        raise ValueError("purchase-price rows must not be empty")
    last_gene_advantage = max(
        (row.target_count for row in rows if row.oligo_pool_usd >= row.gene_fragments_usd),
        default=0,
    )
    candidate = last_gene_advantage + 1
    return candidate if candidate <= rows[-1].target_count else None


__all__ = ["PurchasePriceRow", "build_purchase_price_rows", "stable_oligo_pool_advantage"]
