"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/metrics.py

Metric extraction utilities for Study trial run directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.diversity import compute_elites_full_sequence_nn_table
from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.artifacts.layout import elites_path


def _resolve_joint_score(elites_df: pd.DataFrame) -> pd.Series:
    for column in ("combined_score_final", "objective_scalar"):
        if column in elites_df.columns:
            values = pd.to_numeric(elites_df[column], errors="coerce")
            if values.notna().any():
                return values
    score_cols = [column for column in elites_df.columns if column.startswith("score_")]
    if score_cols:
        score_matrix = elites_df[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if score_matrix.size:
            return pd.Series(np.nanmin(score_matrix, axis=1), index=elites_df.index, dtype=float)
    raise ValueError(
        "Unable to resolve joint score from elites.parquet (expected combined_score_final/objective_scalar/score_*)."
    )


def _identity_maps(elites_df: pd.DataFrame) -> tuple[dict[str, str] | None, dict[str, int] | None]:
    if "id" not in elites_df.columns:
        return None, None
    ids = elites_df["id"].astype(str)
    sequence_column = "canonical_sequence" if "canonical_sequence" in elites_df.columns else "sequence"
    if sequence_column not in elites_df.columns:
        return None, None
    identities = elites_df[sequence_column].astype(str)
    identity_by_elite_id = {str(elite_id): str(seq) for elite_id, seq in zip(ids, identities, strict=False)}
    rank_map: dict[str, int] = {}
    if "rank" in elites_df.columns:
        rank_values = pd.to_numeric(elites_df["rank"], errors="coerce")
        for elite_id, rank in zip(ids, rank_values, strict=False):
            if pd.notna(rank):
                rank_map[str(elite_id)] = int(rank)
    return identity_by_elite_id, (rank_map or None)


def extract_elite_metrics(run_dir: Path) -> dict[str, float | int | None]:
    elites_file = elites_path(run_dir)
    if not elites_file.exists():
        raise FileNotFoundError(f"Missing elites parquet for study metrics: {elites_file}")
    elites_df = read_parquet(elites_file)
    if elites_df.empty:
        raise ValueError(f"Elites parquet is empty for study metrics: {elites_file}")
    if "sequence" not in elites_df.columns:
        raise ValueError(f"Elites parquet missing sequence column for study metrics: {elites_file}")

    scores = _resolve_joint_score(elites_df)
    if scores.dropna().empty:
        raise ValueError(f"Elites parquet has no finite joint scores: {elites_file}")

    identity_by_elite_id, rank_map = _identity_maps(elites_df)
    _, full_summary = compute_elites_full_sequence_nn_table(
        elites_df,
        identity_mode="canonical" if "canonical_sequence" in elites_df.columns else "raw",
        identity_by_elite_id=identity_by_elite_id,
        rank_by_elite_id=rank_map,
    )
    sequence_length = int(elites_df["sequence"].astype(str).str.len().iloc[0])
    return {
        "n_elites": int(len(elites_df)),
        "mean_score": float(scores.mean()),
        "median_score": float(scores.median()),
        "best_score": float(scores.max()),
        "sequence_length": int(full_summary.get("sequence_length_bp") or sequence_length),
        "mean_pairwise_full_bp": full_summary.get("mean_pairwise_full_bp"),
        "min_pairwise_full_bp": full_summary.get("min_pairwise_full_bp"),
        "median_nn_full_bp": full_summary.get("median_nn_full_bp"),
        "mean_pairwise_full_distance": full_summary.get("mean_pairwise_full_distance"),
        "min_pairwise_full_distance": full_summary.get("min_pairwise_full_distance"),
        "median_nn_full_distance": full_summary.get("median_nn_full_distance"),
    }
