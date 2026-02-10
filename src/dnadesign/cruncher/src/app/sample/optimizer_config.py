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

from dnadesign.cruncher.config.schema_v3 import SampleOptimizerConfig

DEFAULT_OPTIMIZER_KIND: str = "gibbs_anneal"
DEFAULT_CHAIN_COUNT: int = 1
DEFAULT_COOLING: dict[str, object] = {"kind": "fixed", "beta": 1.0}


def _resolve_optimizer_kind() -> str:
    return DEFAULT_OPTIMIZER_KIND


def _effective_chain_count(optimizer_cfg: SampleOptimizerConfig | None = None) -> int:
    if optimizer_cfg is None:
        return DEFAULT_CHAIN_COUNT
    return int(optimizer_cfg.chains)


def _resolve_optimizer_defaults() -> dict[str, Any]:
    return {
        "kind": DEFAULT_OPTIMIZER_KIND,
        "chains": DEFAULT_CHAIN_COUNT,
        "cooling": dict(DEFAULT_COOLING),
    }
