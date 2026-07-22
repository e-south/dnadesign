"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/strategy.py

Cloning-strategy config loading for synthesis handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.opal import RestrictionSiteSpec

from .contracts import CloningStrategy


def _restriction_sites(raw: Any) -> tuple[RestrictionSiteSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("restriction_sites must be a list when provided")
    return tuple(RestrictionSiteSpec.from_mapping(item) for item in raw)


def load_cloning_strategy(path: str | Path) -> CloningStrategy:
    """Load a versioned cloning strategy from YAML."""

    strategy_path = Path(path)
    with strategy_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"strategy config must be a mapping: {strategy_path}")
    return CloningStrategy(
        name=str(raw["name"]),
        version=str(raw["version"]),
        left_flank=str(raw["left_flank"]),
        right_flank=str(raw["right_flank"]),
        expected_core_length=raw["expected_core_length"],
        restriction_sites=_restriction_sites(raw.get("restriction_sites")),
    )
