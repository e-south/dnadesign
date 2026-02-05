"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/optimizer_config.py

Resolve optimizer configuration and summarize sampling settings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.config.schema_v3 import SamplePtAdaptConfig, SamplePtConfig

DEFAULT_TEMP_MAX: float = 20.0
DEFAULT_N_TEMPS: int = 6
DEFAULT_SWAP_STRIDE: int = 1
DEFAULT_ADAPTIVE_SWAP: dict[str, object] = SamplePtAdaptConfig(
    enabled=True,
    target_swap=0.25,
    window=50,
    k=0.50,
    min_scale=0.25,
    max_scale=50.0,
    stop_after_tune=True,
).model_dump()


def _resolve_optimizer_kind() -> str:
    return "pt"


def _effective_chain_count(pt_cfg: SamplePtConfig | None = None) -> int:
    if pt_cfg is None:
        return DEFAULT_N_TEMPS
    return int(pt_cfg.n_temps)


def _resolve_pt_defaults() -> dict[str, Any]:
    return {
        "n_temps": DEFAULT_N_TEMPS,
        "temp_max": DEFAULT_TEMP_MAX,
        "swap_stride": DEFAULT_SWAP_STRIDE,
        "adaptive_swap": dict(DEFAULT_ADAPTIVE_SWAP),
    }
