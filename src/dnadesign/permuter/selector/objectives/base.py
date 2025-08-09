"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/selector/objectives/base.py

Objective base class.

An Objective:
  • validates configuration (e.g., weights)
  • normalizes raw metrics per variant
  • computes a single `objective_score` per variant
  • records sidecar metadata for reproducibility

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional


class Objective(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        variants: List[Dict],
        *,
        metrics_cfg: List[Dict],
        objective_cfg: Dict,
        job_ctx: Optional[Dict] = None,
    ) -> None:
        """
        Mutates each variant dict to add:
          - "norm_metrics": {metric_id: float in [0,1]}
          - "objective_score": float
          - "objective_meta": {
                "type": "<objective_name>",
                "weights": {...},
                "norm_scope": "...",
                "norm_stats_id": "r{N}_<hash>"
            }

        Must hard-fail if:
          - config is invalid for this objective
          - any required raw metric is missing or non-finite
        """
        ...
