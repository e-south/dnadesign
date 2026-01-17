"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/objective.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from dnadesign.cruncher.core.sequence import canon_string


def compute_objective_components(
    sequences_df: pd.DataFrame,
    tf_names: Iterable[str],
    *,
    top_k: int | None = None,
    dsdna_canonicalize: bool | None = None,
    overlap_total_bp_median: float | None = None,
) -> dict[str, object]:
    tf_list = list(tf_names)
    df = sequences_df.copy()
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"].copy()

    result: dict[str, object] = {
        "best_score_final": None,
        "top_k_median_final": None,
        "median_min_scaled_tf": None,
        "p10_min_scaled_tf": None,
        "p50_min_scaled_tf": None,
        "p90_min_scaled_tf": None,
        "worst_tf_frequency": {},
        "unique_fraction_raw": None,
        "unique_fraction_canonical": None,
        "canonicalization_enabled": bool(dsdna_canonicalize),
        "overlap_total_bp_median": overlap_total_bp_median,
    }

    if "combined_score_final" in df.columns:
        scores = pd.to_numeric(df["combined_score_final"], errors="coerce").dropna()
        if not scores.empty:
            result["best_score_final"] = float(scores.max())
            if top_k is not None and top_k > 0:
                top_scores = scores.nlargest(min(int(top_k), len(scores)))
                result["top_k_median_final"] = float(np.median(top_scores))

    score_cols = [f"score_{tf}" for tf in tf_list]
    if score_cols and all(col in df.columns for col in score_cols) and not df.empty:
        scores = df[score_cols].to_numpy(dtype=float)
        min_scaled = np.min(scores, axis=1)
        if min_scaled.size:
            result["median_min_scaled_tf"] = float(np.median(min_scaled))
            result["p10_min_scaled_tf"] = float(np.percentile(min_scaled, 10))
            result["p50_min_scaled_tf"] = float(np.percentile(min_scaled, 50))
            result["p90_min_scaled_tf"] = float(np.percentile(min_scaled, 90))
        argmin_idx = np.argmin(scores, axis=1)
        counts = {tf: 0 for tf in tf_list}
        for idx in argmin_idx:
            tf = tf_list[int(idx)]
            counts[tf] += 1
        if counts:
            result["worst_tf_frequency"] = counts

    if "sequence" in df.columns and not df.empty:
        total = int(len(df))
        raw_unique = int(df["sequence"].nunique())
        result["unique_fraction_raw"] = raw_unique / float(total) if total else None
        if "canonical_sequence" in df.columns:
            canon_unique = int(df["canonical_sequence"].astype(str).nunique())
            result["unique_fraction_canonical"] = canon_unique / float(total) if total else None
        elif dsdna_canonicalize:
            canon = df["sequence"].astype(str).map(canon_string)
            canon_unique = int(canon.nunique())
            result["unique_fraction_canonical"] = canon_unique / float(total) if total else None

    return result
