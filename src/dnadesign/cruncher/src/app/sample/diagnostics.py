"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/sample/diagnostics.py

Aggregate diagnostics and scoring summaries for sampling workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.objectives.models import ObjectiveResult
from dnadesign.cruncher.core.scoring import Scorer


@dataclass
class _EliteCandidate:
    seq_arr: np.ndarray
    chain_id: int
    draw_idx: int
    combined_score: float
    min_norm: float
    sum_norm: float
    per_tf_map: dict[str, float]
    norm_map: dict[str, float]
    per_tf_hits: dict[str, dict[str, object]] | None
    objective_results: dict[str, ObjectiveResult] | None = None


def _norm_map_for_elites(
    seq_arr: np.ndarray,
    per_tf_map: dict[str, float],
    *,
    objective_results: dict[str, object] | None = None,
    scorer: Scorer,
    score_scale: str,
) -> dict[str, float]:
    if objective_results is not None:
        out: dict[str, float] = {}
        for objective_id, result in objective_results.items():
            diagnostics = getattr(result, "diagnostics", None)
            if isinstance(diagnostics, dict) and "normalized_scalar" in diagnostics:
                out[str(objective_id)] = float(diagnostics["normalized_scalar"])
        if out:
            return out
    if score_scale.lower() == "normalized-llr":
        missing = [tf for tf in scorer.tf_names if tf not in per_tf_map]
        if missing:
            raise ValueError(f"Per-TF scores missing for normalized-llr: {missing}")
        return {tf: float(per_tf_map[tf]) for tf in scorer.tf_names}
    return scorer.normalized_llr_map(seq_arr)


def resolve_dsdna_mode(*, elites_cfg: object, bidirectional: bool) -> bool:
    _ = elites_cfg
    return bool(bidirectional)


def dsdna_equivalence_enabled(sample_cfg: SampleConfig) -> bool:
    return bool(sample_cfg.objective.bidirectional)
