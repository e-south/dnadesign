"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/summary.py

Table helpers for the curated analysis suite.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def score_frame_from_df(df: pd.DataFrame, tf_names: list[str]) -> pd.DataFrame:
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"].copy()
    cols = _score_columns(tf_names)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns in sequences.parquet: {missing}")
    return df[cols].copy()


def load_score_frame(seq_path: Path, tf_names: list[str]) -> pd.DataFrame:
    df = read_parquet(seq_path)
    return score_frame_from_df(df, tf_names)


def write_score_summary(score_df: pd.DataFrame, tf_names: list[str], out_path: Path) -> None:
    summary = score_df.agg(["mean", "median", "std", "min", "max"]).T
    summary.insert(0, "tf", [name.replace("score_", "") for name in summary.index])
    summary.reset_index(drop=True, inplace=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        write_parquet(summary, out_path)
    else:
        summary.to_csv(out_path, index=False)


def write_elite_topk(elites_df: pd.DataFrame, tf_names: list[str], out_path: Path, top_k: int) -> None:
    if "sequence" not in elites_df.columns:
        raise ValueError("Elites parquet missing required 'sequence' column.")
    cols = _score_columns(tf_names)
    missing = [col for col in cols if col not in elites_df.columns]
    if missing:
        raise ValueError(f"Missing score columns in elites parquet: {missing}")
    df = elites_df.copy()
    if "rank" in df.columns:
        df = df.nsmallest(top_k, "rank")
    elif "norm_sum" in df.columns:
        df = df.nlargest(top_k, "norm_sum")
    else:
        df = df.head(top_k)
    keep_cols = ["sequence"] + [c for c in ("rank", "norm_sum") if c in df.columns] + cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        write_parquet(df[keep_cols], out_path)
    else:
        df[keep_cols].to_csv(out_path, index=False)


def _safe_max(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.max())


def _pareto_front_mask(scores: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    scores = np.nan_to_num(scores, nan=-np.inf)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        better_or_equal = (scores >= scores[i]).all(axis=1)
        strictly_better = (scores > scores[i]).any(axis=1)
        if np.any(better_or_equal & strictly_better):
            dominated[i] = True
    return ~dominated


def write_joint_metrics(elites_df: pd.DataFrame, tf_names: list[str], out_path: Path) -> None:
    cols = _score_columns(tf_names)
    missing = [col for col in cols if col not in elites_df.columns]
    if missing:
        raise ValueError(f"Missing score columns in elites parquet: {missing}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tf_blob = ",".join(tf_names)
    if elites_df.empty:
        payload = {
            "tf_names": tf_blob,
            "joint_min": None,
            "joint_mean": None,
            "joint_hmean": None,
            "balance_index": None,
            "pareto_front_size": 0,
            "pareto_fraction": 0.0,
        }
        df = pd.DataFrame([payload])
        if out_path.suffix == ".parquet":
            write_parquet(df, out_path)
        else:
            df.to_csv(out_path, index=False)
        return

    scores = elites_df[cols].to_numpy(dtype=float)
    joint_min = scores.min(axis=1)
    joint_mean = scores.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        joint_hmean = scores.shape[1] / np.sum(1.0 / scores, axis=1)
    zero_mean = np.isfinite(joint_mean) & (joint_mean == 0)
    zero_mean_count = int(zero_mean.sum())
    balance_index = np.divide(
        joint_min,
        joint_mean,
        out=np.full_like(joint_min, np.nan, dtype=float),
        where=joint_mean != 0,
    )

    pareto_mask = _pareto_front_mask(scores)
    pareto_front_size = int(pareto_mask.sum())
    pareto_fraction = pareto_front_size / float(scores.shape[0]) if scores.shape[0] else 0.0

    payload = {
        "tf_names": tf_blob,
        "joint_min": _safe_max(joint_min),
        "joint_mean": _safe_max(joint_mean),
        "joint_hmean": _safe_max(joint_hmean),
        "balance_index": _safe_max(balance_index),
        "joint_mean_zero_count": zero_mean_count,
        "pareto_front_size": pareto_front_size,
        "pareto_fraction": pareto_fraction,
    }
    df = pd.DataFrame([payload])
    if out_path.suffix == ".parquet":
        write_parquet(df, out_path)
    else:
        df.to_csv(out_path, index=False)
