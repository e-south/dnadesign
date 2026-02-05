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

from dnadesign.cruncher.config.schema_v2 import AdaptiveSwapConfig

DEFAULT_BETA_LADDER: list[float] = [0.2, 1.0, 5.0, 25.0]
DEFAULT_SWAP_PROB: float = 0.10
DEFAULT_ADAPTIVE_SWAP: dict[str, object] = AdaptiveSwapConfig(
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


def _effective_chain_count() -> int:
    return len(DEFAULT_BETA_LADDER)


def _resolve_pt_defaults() -> dict[str, Any]:
    return {
        "kind": "geometric",
        "beta": list(DEFAULT_BETA_LADDER),
        "swap_prob": DEFAULT_SWAP_PROB,
        "adaptive_swap": dict(DEFAULT_ADAPTIVE_SWAP),
    }
