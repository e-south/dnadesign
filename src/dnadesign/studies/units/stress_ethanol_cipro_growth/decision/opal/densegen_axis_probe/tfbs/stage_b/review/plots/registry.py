"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/review/plots/registry.py

Fail-fast renderer registry helpers for Stage B review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .contracts import RealizedReviewRenderer


def build_realized_review_renderer_registry(
    renderers: Mapping[str, RealizedReviewRenderer],
) -> Mapping[str, RealizedReviewRenderer]:
    """Build a realized-label renderer registry keyed by visual kind."""

    registry: dict[str, RealizedReviewRenderer] = {}
    for kind, renderer in renderers.items():
        token = str(kind).strip()
        if not token:
            raise ValueError("Stage B realized review renderer kind must be nonempty")
        if token in registry:
            raise ValueError(f"Duplicate Stage B realized review renderer kind: {token}")
        registry[token] = renderer
    if not registry:
        raise ValueError("Stage B realized review renderer registry must not be empty")
    return MappingProxyType(registry)
