"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/strategies/threshold.py

Threshold selection strategy.

Supports:
  - target: objective | metric (with metric_id)
  - exactly one of: threshold (numeric) | percentile (0-100]
  - for metric targets, use_normalized defaults to True. Percentile requires True.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import Strategy


class ThresholdStrategy(Strategy):
    def select(self, variants: List[Dict], *, cfg: Dict) -> List[Dict]:
        if not variants:
            return []

        target = cfg.get("target", "objective")
        has_thr = "threshold" in cfg
        has_pct = "percentile" in cfg

        if has_thr == has_pct:
            raise ValueError(
                "threshold strategy requires exactly one of 'threshold' or 'percentile'"
            )

        # choose values to threshold on
        if target == "objective":
            values = np.array(
                [float(v["objective_score"]) for v in variants], dtype=float
            )
        elif target == "metric":
            mid = cfg.get("metric_id")
            if not mid:
                raise ValueError("threshold.target=metric requires 'metric_id'")
            use_norm = bool(cfg.get("use_normalized", True))
            if use_norm:
                # goal-aware 0..1, higher-is-better provided by objective stage
                values = np.array(
                    [float(v["norm_metrics"][mid]) for v in variants], dtype=float
                )
            else:
                # raw metric values (assume higher-is-better for numeric threshold)
                if has_pct:
                    raise ValueError(
                        "percentile requires use_normalized: true when target=metric"
                    )
                values = np.array(
                    [float(v["metrics"][mid]) for v in variants], dtype=float
                )
        else:  # pragma: no cover
            raise ValueError("threshold.target must be 'objective' or 'metric'")

        if has_thr:
            thr = float(cfg["threshold"])
            mask = values >= thr
        else:
            pct = float(cfg["percentile"])
            cutoff = float(np.percentile(values, pct))
            mask = values >= cutoff

        elites = [v for v, keep in zip(variants, mask) if keep]
        return elites
