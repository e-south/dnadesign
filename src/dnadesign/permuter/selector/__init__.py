"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/__init__.py

Selector Orchestrator - wires Objective + Strategy plugins.

Public entry:
    select(variants, *, metrics_cfg, select_cfg, job_ctx=None) -> elites

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .objectives import get as get_objective
from .strategies import get as get_strategy


def select(
    variants: List[Dict],
    *,
    metrics_cfg: List[Dict],
    select_cfg: Dict,
    job_ctx: Optional[Dict] = None,
) -> List[Dict]:
    """
    1) Objective: compute per-variant objective_score, normalized metrics, and
       objective_meta (including sidecar id for normalization stats).
    2) Strategy: pick elites purely by objective_score.

    Returns:
      A new list of selected variant dicts.
    """
    if not variants:
        return []  # trivial fast-path

    objective_cfg = select_cfg.get("objective", {})
    strategy_cfg = select_cfg.get("strategy", {})

    # Objective stage
    objective_type = objective_cfg.get("type")
    objective_cls = get_objective(objective_type)
    objective = objective_cls()
    objective.compute(
        variants,
        metrics_cfg=metrics_cfg,
        objective_cfg=objective_cfg,
        job_ctx=job_ctx,
    )

    # Strategy stage
    strategy_type = strategy_cfg.get("type")
    strategy_cls = get_strategy(strategy_type)
    strategy = strategy_cls()
    elites = strategy.select(variants, cfg=strategy_cfg)
    return elites
