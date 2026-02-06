"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/overlap.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


def _resolve_hist_bins(hist_bins: list[int] | None, max_value: float) -> list[float]:
    if hist_bins is None:
        hist_bins = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    bins = sorted({float(v) for v in hist_bins if v > 0})
    if not bins:
        bins = [1.0, 2.0, 3.0, 4.0, 5.0]
    if max_value > bins[-1]:
        bins.append(float(max_value))
    return bins


def compute_overlap_tables(
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    tf_names: Iterable[str],
    pwm_widths: dict[str, int] | None = None,
    hist_bins: list[int] | None = None,
    *,
    include_sequences: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | None]]:
    tf_list = list(tf_names)
    pair_keys: list[tuple[str, str]] = []
    pair_stats: dict[tuple[str, str], dict[str, object]] = {}
    for i, tf_i in enumerate(tf_list):
        for tf_j in tf_list[i + 1 :]:
            key = (tf_i, tf_j)
            pair_keys.append(key)
            pair_stats[key] = {
                "pair_count": 0,
                "overlap_count": 0,
                "overlap_bp": [],
                "delta": [],
                "strand_counts": Counter(),
            }

    elite_rows: list[dict[str, object]] = []
    if elites_df is None or elites_df.empty:
        pair_df = pd.DataFrame(
            [
                {
                    "tf_i": tf_i,
                    "tf_j": tf_j,
                    "pair_count": 0,
                    "overlap_count": 0,
                    "overlap_rate": None,
                    "overlap_bp_mean": None,
                    "overlap_bp_median": None,
                    "overlap_bp_hist": None,
                    "offset_delta_mean": None,
                    "offset_delta_median": None,
                    "strand_pp": 0,
                    "strand_pm": 0,
                    "strand_mp": 0,
                    "strand_mm": 0,
                }
                for tf_i, tf_j in pair_keys
            ]
        )
        elite_cols = ["id", "rank", "sequence_hash", "overlap_total_bp", "overlap_pair_count"]
        if include_sequences:
            elite_cols.append("sequence")
        elite_df = pd.DataFrame(columns=elite_cols)
        return pair_df, elite_df, {"overlap_rate_median": None, "overlap_total_bp_median": None}

    if hits_df is None or hits_df.empty:
        raise ValueError("elites_hits.parquet is required to compute overlap metrics.")

    elite_info: dict[str, dict[str, object]] = {}
    for _, row in elites_df.iterrows():
        elite_id = str(row.get("id") or "")
        if not elite_id:
            raise ValueError("Elites parquet missing required 'id' column.")
        elite_info[elite_id] = {
            "rank": row.get("rank"),
            "sequence": row.get("sequence"),
        }

    hits_by_elite: dict[str, dict[str, tuple[int, int, str | None]]] = {}
    for _, row in hits_df.iterrows():
        elite_id = str(row.get("elite_id") or "")
        tf_name = row.get("tf")
        if not elite_id or tf_name not in tf_list:
            continue
        start = row.get("best_start")
        width = row.get("pwm_width")
        strand = row.get("best_strand")
        if not isinstance(start, (int, float)) or not isinstance(width, (int, float)) or not isinstance(strand, str):
            raise ValueError(f"Invalid hit metadata for elite '{elite_id}' TF '{tf_name}'.")
        start_i = int(start)
        end_i = start_i + int(width)
        hits_by_elite.setdefault(elite_id, {})[str(tf_name)] = (start_i, end_i, strand)

    for elite_id, hits in hits_by_elite.items():
        elite_meta = elite_info.get(elite_id, {})

        overlap_total = 0
        overlap_pairs = 0
        for tf_i, tf_j in pair_keys:
            hit_i = hits.get(tf_i)
            hit_j = hits.get(tf_j)
            if hit_i is None or hit_j is None:
                continue
            start_i, end_i, strand_i = hit_i
            start_j, end_j, strand_j = hit_j
            stats = pair_stats[(tf_i, tf_j)]
            stats["pair_count"] += 1
            stats["delta"].append(start_j - start_i)
            if strand_i and strand_j:
                stats["strand_counts"][f"{strand_i}{strand_j}"] += 1
            overlap_bp = max(0, min(end_i, end_j) - max(start_i, start_j))
            if overlap_bp > 0:
                stats["overlap_count"] += 1
                stats["overlap_bp"].append(overlap_bp)
                overlap_total += overlap_bp
                overlap_pairs += 1

        seq_val = elite_meta.get("sequence")
        seq_hash = None
        if isinstance(seq_val, str):
            seq_hash = hashlib.sha256(seq_val.encode("utf-8")).hexdigest()[:12]
        row_payload = {
            "id": elite_id,
            "rank": elite_meta.get("rank"),
            "sequence_hash": seq_hash,
            "overlap_total_bp": overlap_total,
            "overlap_pair_count": overlap_pairs,
        }
        if include_sequences:
            row_payload["sequence"] = seq_val
        elite_rows.append(row_payload)

    pair_rows: list[dict[str, object]] = []
    for tf_i, tf_j in pair_keys:
        stats = pair_stats[(tf_i, tf_j)]
        pair_count = int(stats["pair_count"])
        overlap_count = int(stats["overlap_count"])
        overlap_rate = overlap_count / float(pair_count) if pair_count else None
        overlap_bp = np.asarray(stats["overlap_bp"], dtype=float)
        delta = np.asarray(stats["delta"], dtype=float)
        hist_payload = None
        if overlap_bp.size:
            bins = _resolve_hist_bins(hist_bins, float(overlap_bp.max()))
            counts, edges = np.histogram(overlap_bp, bins=bins)
            hist_payload = json.dumps({"bins": edges.tolist(), "counts": counts.tolist()})
        strand_counts: Counter = stats["strand_counts"]
        pair_rows.append(
            {
                "tf_i": tf_i,
                "tf_j": tf_j,
                "pair_count": pair_count,
                "overlap_count": overlap_count,
                "overlap_rate": overlap_rate,
                "overlap_bp_mean": float(np.mean(overlap_bp)) if overlap_bp.size else None,
                "overlap_bp_median": float(np.median(overlap_bp)) if overlap_bp.size else None,
                "overlap_bp_hist": hist_payload,
                "offset_delta_mean": float(np.mean(delta)) if delta.size else None,
                "offset_delta_median": float(np.median(delta)) if delta.size else None,
                "strand_pp": int(strand_counts.get("++", 0)),
                "strand_pm": int(strand_counts.get("+-", 0)),
                "strand_mp": int(strand_counts.get("-+", 0)),
                "strand_mm": int(strand_counts.get("--", 0)),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    elite_df = pd.DataFrame(elite_rows)
    overlap_rate_median = None
    if not pair_df.empty and pair_df["overlap_rate"].notna().any():
        overlap_rate_median = float(pair_df["overlap_rate"].median())
    overlap_total_bp_median = None
    if not elite_df.empty and elite_df["overlap_total_bp"].notna().any():
        overlap_total_bp_median = float(elite_df["overlap_total_bp"].median())

    return (
        pair_df,
        elite_df,
        {
            "overlap_rate_median": overlap_rate_median,
            "overlap_total_bp_median": overlap_total_bp_median,
        },
    )


def extract_elite_hits(
    hits_df: pd.DataFrame,
    tf_names: Iterable[str],
) -> pd.DataFrame:
    tf_list = list(tf_names)
    if hits_df is None or hits_df.empty:
        return pd.DataFrame(columns=["tf", "offset", "end", "strand", "elite_id"])
    rows: list[dict[str, object]] = []
    for _, row in hits_df.iterrows():
        tf_name = row.get("tf")
        if tf_name not in tf_list:
            continue
        start = row.get("best_start")
        width = row.get("pwm_width")
        strand = row.get("best_strand")
        if not isinstance(start, (int, float)) or not isinstance(width, (int, float)) or not isinstance(strand, str):
            continue
        rows.append(
            {
                "tf": tf_name,
                "offset": int(start),
                "end": int(start) + int(width),
                "strand": strand,
                "elite_id": row.get("elite_id"),
            }
        )
    return pd.DataFrame(rows)
