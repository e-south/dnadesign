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


def compute_objective_components(
    sequences_df: pd.DataFrame,
    tf_names: Iterable[str],
    *,
    top_k: int | None = None,
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
        "canonicalization_enabled": False,
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
            result["canonicalization_enabled"] = True

    learning: dict[str, object] = {}
    required_cols = {"combined_score_final", "draw", "chain"}
    if required_cols.issubset(df.columns) and not df.empty:
        score_df = df[list(required_cols)].copy()
        score_df["combined_score_final"] = pd.to_numeric(score_df["combined_score_final"], errors="coerce")
        score_df["draw"] = pd.to_numeric(score_df["draw"], errors="coerce")
        score_df["chain"] = pd.to_numeric(score_df["chain"], errors="coerce")
        score_df = score_df.dropna(subset=["combined_score_final", "draw", "chain"])
        if not score_df.empty:
            best_idx = score_df["combined_score_final"].idxmax()
            best_row = score_df.loc[best_idx]
            best_draw = int(best_row["draw"])
            best_chain = int(best_row["chain"])
            max_draw = int(score_df["draw"].max())
            learning["best_score_draw"] = best_draw
            learning["best_score_chain"] = best_chain
            learning["best_score_fraction"] = best_draw / float(max_draw) if max_draw > 0 else None

            last_improve_by_chain: dict[int, int] = {}
            for chain_value, chain_df in score_df.groupby("chain"):
                chain_df = chain_df.sort_values("draw")
                best_local = None
                last_improve = None
                for draw, score in chain_df[["draw", "combined_score_final"]].itertuples(index=False):
                    draw_int = int(draw)
                    if best_local is None or score > best_local:
                        best_local = float(score)
                        last_improve = draw_int
                if last_improve is not None:
                    last_improve_by_chain[int(chain_value)] = last_improve

            if last_improve_by_chain:
                last_improve_draw = max(last_improve_by_chain.values())
                learning["last_improvement_draw"] = last_improve_draw
                learning["plateau_draws"] = max_draw - last_improve_draw if max_draw >= last_improve_draw else 0

    if learning:
        result["learning"] = learning

    return result
