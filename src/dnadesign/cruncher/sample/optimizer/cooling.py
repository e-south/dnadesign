"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/optimizer/cooling.py

All cooling-schedule factories used by optimizers:
    - make_beta_scheduler()   → for single-chain (fixed|linear|piecewise)
    - make_beta_ladder()      → for parallel-tempering (“geometric” or singleton)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from typing import Any, Dict, List

import numpy as np


def make_beta_scheduler(cooling_cfg: Dict[str, Any], total_sweeps: int):
    """
    Return a function beta_of(iter_idx: int) → float, based on cooling_cfg.

    cooling_cfg keys:
      - kind == "fixed"     → {'kind':'fixed', 'beta': float}
      - kind == "linear"    → {'kind':'linear', 'beta': (β_start, β_end)}
      - kind == "piecewise" → {'kind':'piecewise', 'stages': [ {'sweeps':int, 'beta':float}, ... ]}

    total_sweeps = tune + draws

    Usage:
        beta_of = make_beta_scheduler(cfg["cooling"], total_sweeps)
        β_at_iter_100 = beta_of(100)
    """
    kind = cooling_cfg["kind"]

    if kind == "fixed":
        β_const = float(cooling_cfg["beta"])

        def beta_of(_it: int) -> float:
            return β_const

        return beta_of

    if kind == "linear":
        β_start, β_end = cooling_cfg["beta"]

        def beta_of(it: int) -> float:
            frac = min(it / total_sweeps, 1.0)
            return β_start + frac * (β_end - β_start)

        return beta_of

    # kind == "piecewise"
    stages = sorted(cooling_cfg["stages"], key=lambda s: s["sweeps"])
    sweep_pts = [s["sweeps"] for s in stages]
    betas = [s["beta"] for s in stages]

    def beta_of(it: int) -> float:
        # If beyond last stage, return last β
        if it >= sweep_pts[-1]:
            return betas[-1]
        j = np.searchsorted(sweep_pts, it, side="right") - 1
        t0, t1 = sweep_pts[j], sweep_pts[j + 1]
        b0, b1 = betas[j], betas[j + 1]
        return b0 + (b1 - b0) * ((it - t0) / (t1 - t0))

    return beta_of


def make_beta_ladder(cooling_cfg: Dict[str, Any]) -> List[float]:
    """
    For PT, build a “β-ladder” (a list of β values):
      - If kind=="geometric": return copy of cooling_cfg["beta"] (list of floats)
      - If kind=="fixed": return [cooling_cfg["beta"]]
      - Otherwise: error
    """
    kind = cooling_cfg["kind"]
    if kind == "geometric":
        return list(cooling_cfg["beta"])
    if kind == "fixed":
        return [cooling_cfg["beta"]]
    raise ValueError("For PT, cooling.kind must be 'geometric' or 'fixed'")
