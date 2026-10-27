"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/economics/__init__.py

Expose synthesis purchase-price comparison contracts and services.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .contracts import PricingSnapshot, SynthesisScenario, load_pricing_snapshot
from .model import PurchasePriceRow, build_purchase_price_rows, stable_oligo_pool_advantage
from .render import RenderedPurchasePriceComparison, render_purchase_price_comparison

__all__ = [
    "PricingSnapshot",
    "PurchasePriceRow",
    "RenderedPurchasePriceComparison",
    "SynthesisScenario",
    "build_purchase_price_rows",
    "load_pricing_snapshot",
    "render_purchase_price_comparison",
    "stable_oligo_pool_advantage",
]
