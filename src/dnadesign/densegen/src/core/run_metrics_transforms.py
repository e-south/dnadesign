"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/run_metrics_transforms.py

Pure transforms for run metrics derivation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .record_values import require_list_of_dicts as _ensure_list_of_dicts


def _shannon_entropy(values: Iterable[int]) -> float:
    counts = [float(v) for v in values if v is not None and float(v) > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def _plan_fixed_bp_min(plan) -> int:
    fixed = getattr(plan, "fixed_elements", None)
    if fixed is None:
        return 0
    total = 0
    promoter_constraints = getattr(fixed, "promoter_constraints", None) or []
    for item in promoter_constraints:
        upstream = str(getattr(item, "upstream", ""))
        downstream = str(getattr(item, "downstream", ""))
        spacer = getattr(item, "spacer_length", None)
        if spacer is None:
            spacer_min = 0
        elif isinstance(spacer, (list, tuple)):
            spacer_min = int(min(spacer))
        else:
            spacer_min = int(spacer)
        total += len(upstream) + len(downstream) + spacer_min
    return int(total)


def _required_min_length(
    members: pd.DataFrame,
    *,
    groups: list,
    min_count_by_regulator: dict[str, int],
    min_count_per_tf: int,
) -> int | None:
    if members.empty:
        return None
    tf_lengths = (
        members.assign(tfbs_len=members["tfbs"].astype(str).map(len))
        .groupby(members["tf"].astype(str))["tfbs_len"]
        .apply(list)
        .to_dict()
    )
    if not tf_lengths:
        return None
    for tf in tf_lengths:
        tf_lengths[tf] = sorted(int(v) for v in tf_lengths[tf] if v is not None)

    per_tf_required: dict[str, int] = {}
    if min_count_per_tf > 0:
        for tf in tf_lengths:
            per_tf_required[tf] = max(per_tf_required.get(tf, 0), int(min_count_per_tf))
    for tf, count in min_count_by_regulator.items():
        per_tf_required[tf] = max(per_tf_required.get(tf, 0), int(count))

    per_tf_total = 0
    for tf, count in per_tf_required.items():
        lengths = tf_lengths.get(tf) or []
        if len(lengths) < int(count):
            return None
        per_tf_total += int(sum(lengths[: int(count)]))

    group_required_extra = 0
    for group in groups:
        members_min = []
        for tf in group.members:
            lengths = tf_lengths.get(tf) or []
            if lengths:
                members_min.append(int(lengths[0]))
        if len(members_min) < int(group.min_required):
            return None
        group_required_extra += int(sum(sorted(members_min)[: int(group.min_required)]))

    return int(per_tf_total + group_required_extra)


def _assign_score_quantiles(df: pd.DataFrame, *, quantiles: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_rank"] = df.sort_values(["best_hit_score", "tfbs"], ascending=[False, True]).groupby("tf").cumcount()
    df["_rank_max"] = df.groupby("tf")["_rank"].transform("max") + 1
    q = int(quantiles)
    if q <= 0:
        df["score_quantile"] = 1
        return df
    df["score_quantile"] = 1
    for tf, sub in df.groupby("tf"):
        total = int(sub["_rank_max"].iloc[0])
        bins = min(q, total)
        idx = list(range(total))
        splits = np.array_split(idx, bins)
        mapping = {}
        for q_idx, indices in enumerate(splits, start=1):
            for item in indices:
                mapping[item] = q_idx
        df.loc[sub.index, "score_quantile"] = [mapping[int(r)] for r in sub["_rank"].tolist()]
    df.drop(columns=["_rank", "_rank_max"], inplace=True)
    return df


def _placements_from_dense_arrays(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "densegen__used_tfbs_detail",
        "densegen__sampling_library_index",
        "densegen__sampling_library_hash",
        "densegen__input_name",
        "densegen__plan",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"records.parquet missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for _, row in df.iterrows():
        input_name = str(row.get("densegen__input_name") or "")
        plan_name = str(row.get("densegen__plan") or "")
        if not input_name or not plan_name:
            raise ValueError("records.parquet missing input_name/plan_name metadata.")
        library_index = int(row.get("densegen__sampling_library_index") or 0)
        library_hash = str(row.get("densegen__sampling_library_hash") or "").strip()
        if not library_hash:
            raise ValueError(
                "records.parquet missing densegen__sampling_library_hash metadata for "
                f"{input_name}/{plan_name} library_index={library_index}."
            )
        for item in _ensure_list_of_dicts(row.get("densegen__used_tfbs_detail")):
            part_kind = str(item.get("part_kind") or "tfbs").strip().lower()
            if part_kind != "tfbs":
                continue
            tf = str(item.get("regulator") or "").strip()
            tfbs = str(item.get("sequence") or "").strip()
            if not tf or not tfbs:
                continue
            rows.append(
                {
                    "input_name": input_name,
                    "plan_name": plan_name,
                    "library_index": library_index,
                    "library_hash": library_hash,
                    "tf": tf,
                    "tfbs": tfbs,
                }
            )
    return pd.DataFrame(rows)


def _placements_from_composition(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "input_name",
        "plan_name",
        "library_index",
        "library_hash",
        "regulator",
        "sequence",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"composition.parquet missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for _, row in df.iterrows():
        input_name = str(row.get("input_name") or "")
        plan_name = str(row.get("plan_name") or "")
        if not input_name or not plan_name:
            raise ValueError("composition.parquet missing input_name/plan_name metadata.")
        library_index = int(row.get("library_index") or 0)
        library_hash = str(row.get("library_hash") or "")
        part_kind = str(row.get("part_kind") or "tfbs").strip().lower()
        if part_kind != "tfbs":
            continue
        tf = str(row.get("regulator") or "").strip()
        tfbs = str(row.get("sequence") or "").strip()
        if not tf or not tfbs:
            continue
        rows.append(
            {
                "input_name": input_name,
                "plan_name": plan_name,
                "library_index": library_index,
                "library_hash": library_hash,
                "tf": tf,
                "tfbs": tfbs,
            }
        )
    return pd.DataFrame(rows)
