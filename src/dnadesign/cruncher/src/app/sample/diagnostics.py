"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/diagnostics.py

Aggregate diagnostics and scoring summaries for sampling workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnadesign.cruncher.config.schema_v2 import SampleConfig
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


def _norm_map_for_elites(
    seq_arr: np.ndarray,
    per_tf_map: dict[str, float],
    *,
    scorer: Scorer,
    score_scale: str,
) -> dict[str, float]:
    if score_scale.lower() == "normalized-llr":
        missing = [tf for tf in scorer.tf_names if tf not in per_tf_map]
        if missing:
            raise ValueError(f"Per-TF scores missing for normalized-llr: {missing}")
        return {tf: float(per_tf_map[tf]) for tf in scorer.tf_names}
    return scorer.normalized_llr_map(seq_arr)


def _elite_filter_passes(
    *,
    norm_map: dict[str, float],
    min_norm: float,
    min_per_tf_norm: float | None,
    require_all_tfs_over_min_norm: bool,
) -> bool:
    if min_per_tf_norm is not None:
        if require_all_tfs_over_min_norm:
            if not all(score >= min_per_tf_norm for score in norm_map.values()):
                return False
        else:
            if min_norm < min_per_tf_norm:
                return False
    return True


def resolve_dsdna_mode(*, elites_cfg: object, bidirectional: bool) -> bool:
    _ = elites_cfg
    return bool(bidirectional)


def dsdna_equivalence_enabled(sample_cfg: SampleConfig) -> bool:
    return resolve_dsdna_mode(elites_cfg=sample_cfg.elites, bidirectional=sample_cfg.objective.bidirectional)
