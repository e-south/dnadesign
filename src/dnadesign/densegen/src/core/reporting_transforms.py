"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/reporting_transforms.py

Pure data transforms for DenseGen report tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .record_values import coerce_list as _ensure_list
from .record_values import coerce_list_of_dicts as _ensure_list_of_dicts


def _dg(col: str) -> str:
    return col if col.startswith("densegen__") else f"densegen__{col}"


def _solution_id(row: pd.Series) -> str:
    if "solution_id" in row and isinstance(row["solution_id"], str) and row["solution_id"]:
        return row["solution_id"]
    if "id" in row and isinstance(row["id"], str) and row["id"]:
        return row["id"]
    raise ValueError("Output records missing solution_id/id; regenerate outputs before reporting.")


def _explode_used(df: pd.DataFrame) -> pd.DataFrame:
    used_col = _dg("used_tfbs_detail")
    lib_hash_col = _dg("sampling_library_hash")
    lib_index_col = _dg("sampling_library_index")
    plan_col = _dg("plan")
    input_col = _dg("input_name")
    required_col = _dg("required_regulators")

    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        used_detail = _ensure_list_of_dicts(row.get(used_col))
        if not used_detail:
            continue
        seq_id = _solution_id(row)
        input_name = str(row.get(input_col) or "")
        plan_name = str(row.get(plan_col) or "")
        library_index = int(row.get(lib_index_col) or 0)
        library_hash = str(row.get(lib_hash_col) or "").strip()
        if not library_hash:
            raise ValueError(
                f"Output record '{seq_id}' is missing {lib_hash_col}; "
                "regenerate run outputs with the current DenseGen schema."
            )
        for entry in used_detail:
            tf = str(entry.get("tf") or "").strip()
            tfbs = str(entry.get("tfbs") or "").strip()
            if not tf and not tfbs:
                continue
            records.append(
                {
                    "solution_id": seq_id,
                    "library_hash": library_hash,
                    "library_index": library_index,
                    "plan": plan_name,
                    "input_name": input_name,
                    "tf": tf,
                    "tfbs": tfbs,
                    "motif_id": entry.get("motif_id"),
                    "tfbs_id": entry.get("tfbs_id"),
                    "orientation": entry.get("orientation"),
                    "offset": entry.get("offset"),
                    "length": entry.get("length"),
                    "end": entry.get("end"),
                    "site_id": entry.get("site_id"),
                    "source": entry.get("source"),
                    "required_regulators": row.get(required_col),
                }
            )
    return pd.DataFrame(records)


def _explode_library_from_attempts(attempts_df: pd.DataFrame) -> pd.DataFrame:
    if attempts_df is None or attempts_df.empty:
        return pd.DataFrame(
            columns=[
                "library_hash",
                "library_index",
                "input_name",
                "plan_name",
                "tf",
                "tfbs",
                "site_id",
                "source",
                "tfbs_length",
            ]
        )
    records: list[dict[str, Any]] = []
    for _, row in attempts_df.iterrows():
        library_hash = str(row.get("sampling_library_hash") or "")
        library_index = int(row.get("sampling_library_index") or 0)
        input_name = str(row.get("input_name") or "")
        plan_name = str(row.get("plan_name") or "")
        library_tfbs = _ensure_list(row.get("library_tfbs"))
        library_tfs = _ensure_list(row.get("library_tfs"))
        library_site_ids = _ensure_list(row.get("library_site_ids"))
        library_sources = _ensure_list(row.get("library_sources"))
        if not library_tfbs:
            continue
        for idx, tfbs in enumerate(library_tfbs):
            tf = str(library_tfs[idx]) if idx < len(library_tfs) else ""
            site_id = str(library_site_ids[idx]) if idx < len(library_site_ids) else ""
            source = str(library_sources[idx]) if idx < len(library_sources) else ""
            records.append(
                {
                    "library_hash": library_hash,
                    "library_index": library_index,
                    "input_name": input_name,
                    "plan_name": plan_name,
                    "tf": tf,
                    "tfbs": str(tfbs),
                    "site_id": site_id if site_id not in ("", "None") else None,
                    "source": source if source not in ("", "None") else None,
                    "tfbs_length": len(str(tfbs)),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "library_hash",
                "library_index",
                "input_name",
                "plan_name",
                "tf",
                "tfbs",
                "site_id",
                "source",
                "tfbs_length",
            ]
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(
            ["library_hash", "library_index", "input_name", "plan_name", "tf", "tfbs", "site_id", "source"]
        )
    return df


def _normalized_entropy_from_counts(counts: dict[str, int]) -> float | None:
    if not counts:
        return None
    total = float(sum(counts.values()))
    if total <= 0:
        return None
    if len(counts) <= 1:
        return 0.0
    ent = 0.0
    for val in counts.values():
        p = float(val) / total
        if p <= 0:
            continue
        ent -= p * np.log(p)
    return float(ent / np.log(len(counts)))


def _usage_stats_from_counts(counts: dict[str, int]) -> dict[str, float | int | None]:
    if not counts:
        return {"min": None, "median": None, "max": None, "unique": 0}
    values = np.asarray(list(counts.values()), dtype=float)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "unique": int(len(values)),
    }


def _compute_cooccurrence(used_df: pd.DataFrame) -> pd.DataFrame:
    if used_df.empty:
        return pd.DataFrame(columns=["library_hash", "plan", "tf_left", "tf_right", "count"])
    rows = []
    grouped = used_df.groupby(["library_hash", "plan", "solution_id"])
    for (library_hash, plan, seq_id), group in grouped:
        tfs = sorted({tf for tf in group["tf"].tolist() if tf})
        for i in range(len(tfs)):
            for j in range(i + 1, len(tfs)):
                rows.append(
                    {
                        "library_hash": library_hash,
                        "plan": plan,
                        "solution_id": seq_id,
                        "tf_left": tfs[i],
                        "tf_right": tfs[j],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["library_hash", "plan", "tf_left", "tf_right", "count"])
    pairs = pd.DataFrame(rows)
    agg = (
        pairs.groupby(["library_hash", "plan", "tf_left", "tf_right"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return agg


def _compute_adjacency(used_df: pd.DataFrame) -> pd.DataFrame:
    if used_df.empty:
        return pd.DataFrame(columns=["library_hash", "plan", "tf_left", "tf_right", "count", "mean_distance"])
    rows = []
    for (library_hash, plan, seq_id), group in used_df.groupby(["library_hash", "plan", "solution_id"]):
        sub = group.dropna(subset=["offset"]).sort_values("offset")
        if sub.empty or len(sub) < 2:
            continue
        pairs = zip(sub.iloc[:-1].itertuples(index=False), sub.iloc[1:].itertuples(index=False))
        for left, right in pairs:
            if not left.tf or not right.tf:
                continue
            dist = None
            if left.offset is not None and right.offset is not None:
                dist = int(right.offset) - int(left.offset)
            rows.append(
                {
                    "library_hash": library_hash,
                    "plan": plan,
                    "solution_id": seq_id,
                    "tf_left": left.tf,
                    "tf_right": right.tf,
                    "distance": dist,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["library_hash", "plan", "tf_left", "tf_right", "count", "mean_distance"])
    pairs = pd.DataFrame(rows)
    agg = (
        pairs.groupby(["library_hash", "plan", "tf_left", "tf_right"])
        .agg(count=("distance", "size"), mean_distance=("distance", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return agg
