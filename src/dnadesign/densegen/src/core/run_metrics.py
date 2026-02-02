"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/run_metrics.py

Run-level diagnostics metrics for Stage-A/Stage-B sampling and solver outcomes.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .artifacts.pool import load_pool_artifact
from .run_paths import run_outputs_root, run_tables_root

RUN_METRICS_VERSION = "1.0"
DEFAULT_SCORE_QUANTILES = 5


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


def _ensure_list_of_dicts(value) -> list[dict]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError("used_tfbs_detail must be a list of dicts or JSON.") from exc
        if isinstance(parsed, list):
            if any(not isinstance(item, dict) for item in parsed):
                raise ValueError("used_tfbs_detail JSON list must contain dicts.")
            return list(parsed)
        raise ValueError("used_tfbs_detail JSON must decode to a list.")
    if isinstance(value, (list, np.ndarray)):
        items = list(value)
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("used_tfbs_detail list must contain dicts.")
        return items
    raise ValueError(f"used_tfbs_detail must be list[dict], got {type(value).__name__}.")


def _ensure_list(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError("Expected list data or JSON-encoded list.") from exc
        if isinstance(parsed, list):
            return list(parsed)
        raise ValueError("Expected JSON list data.")
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    raise ValueError(f"Expected list data, got {type(value).__name__}.")


def _load_events(events_path: Path) -> pd.DataFrame:
    rows = []
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return pd.DataFrame(rows)


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
        raise ValueError(f"dense_arrays.parquet missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for _, row in df.iterrows():
        input_name = str(row.get("densegen__input_name") or "")
        plan_name = str(row.get("densegen__plan") or "")
        if not input_name or not plan_name:
            raise ValueError("dense_arrays.parquet missing input_name/plan_name metadata.")
        library_index = int(row.get("densegen__sampling_library_index") or 0)
        library_hash = str(row.get("densegen__sampling_library_hash") or "")
        for item in _ensure_list_of_dicts(row.get("densegen__used_tfbs_detail")):
            tf = str(item.get("tf") or "").strip()
            tfbs = str(item.get("tfbs") or "").strip()
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


def _load_pool_frames(run_root: Path) -> tuple[pd.DataFrame | None, str | None]:
    pool_dir = run_outputs_root(run_root) / "pools"
    if not pool_dir.exists():
        return None, "missing_pool_dir"
    try:
        manifest = load_pool_artifact(pool_dir)
    except FileNotFoundError:
        return None, "missing_pool_manifest"
    frames = []
    for name, entry in manifest.inputs.items():
        pool_path = entry.pool_path
        if not pool_path.is_absolute():
            pool_path = pool_dir / pool_path
        if not pool_path.exists():
            return None, f"missing_pool:{pool_path.name}"
        df = pd.read_parquet(pool_path)
        if "input_name" not in df.columns:
            df.insert(0, "input_name", name)
        if "tfbs" not in df.columns and "tfbs_sequence" in df.columns:
            df["tfbs"] = df["tfbs_sequence"].astype(str)
        frames.append(df)
    if not frames:
        return None, "empty_pool"
    return pd.concat(frames, ignore_index=True), None


def build_run_metrics(*, cfg, run_root: Path) -> pd.DataFrame:
    outputs_root = run_outputs_root(run_root)
    tables_root = run_tables_root(run_root)
    attempts_path = tables_root / "attempts.parquet"
    if not attempts_path.exists():
        raise RuntimeError(f"attempts.parquet not found at {attempts_path}")
    attempts_df = pd.read_parquet(attempts_path)

    libraries_dir = outputs_root / "libraries"
    builds_path = libraries_dir / "library_builds.parquet"
    members_path = libraries_dir / "library_members.parquet"
    if not builds_path.exists() or not members_path.exists():
        raise RuntimeError("library_builds.parquet and library_members.parquet are required to build run metrics.")
    builds_df = pd.read_parquet(builds_path)
    members_df = pd.read_parquet(members_path)

    composition_path = tables_root / "composition.parquet"
    composition_df = pd.read_parquet(composition_path) if composition_path.exists() else pd.DataFrame()

    dense_arrays_path = tables_root / "dense_arrays.parquet"
    dense_arrays_df = pd.DataFrame()
    if composition_df.empty and dense_arrays_path.exists():
        dense_arrays_df = pd.read_parquet(
            dense_arrays_path,
            columns=[
                "densegen__used_tfbs_detail",
                "densegen__sampling_library_index",
                "densegen__sampling_library_hash",
                "densegen__input_name",
                "densegen__plan",
            ],
        )

    placement_source = "none"
    placements_df = pd.DataFrame()
    if not composition_df.empty:
        placements_df = composition_df.copy()
        placement_source = "composition"
    elif not dense_arrays_df.empty:
        placements_df = _placements_from_dense_arrays(dense_arrays_df)
        placement_source = "dense_arrays"

    events_path = outputs_root / "meta" / "events.jsonl"
    events_df = pd.DataFrame()
    if events_path.exists():
        events_df = _load_events(events_path)

    pool_df, pool_status = _load_pool_frames(run_root)
    has_scores = False
    has_tiers = False
    if pool_df is not None:
        has_scores = "best_hit_score" in pool_df.columns
        has_tiers = "tier" in pool_df.columns

    records: list[dict] = []
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = str(cfg.run.id)

    records.append(
        {
            "metric_group": "run_inputs",
            "run_id": run_id,
            "created_at": created_at,
            "has_attempts": True,
            "attempt_rows": int(len(attempts_df)),
            "has_libraries": True,
            "library_build_rows": int(len(builds_df)),
            "library_member_rows": int(len(members_df)),
            "has_composition": bool(not composition_df.empty),
            "composition_rows": int(len(composition_df)) if not composition_df.empty else 0,
            "has_dense_arrays": bool(not dense_arrays_df.empty),
            "dense_arrays_rows": int(len(dense_arrays_df)) if not dense_arrays_df.empty else 0,
            "placement_source": placement_source,
            "has_pools": pool_df is not None,
            "pool_rows": int(len(pool_df)) if pool_df is not None else 0,
            "pool_status": pool_status,
            "has_best_hit_score": has_scores,
            "has_tier": has_tiers,
            "has_events": bool(not events_df.empty),
            "events_rows": int(len(events_df)) if not events_df.empty else 0,
            "has_sampling_pressure": bool(
                not events_df.empty
                and "event" in events_df.columns
                and (events_df["event"] == "LIBRARY_SAMPLING_PRESSURE").any()
            ),
            "metrics_version": RUN_METRICS_VERSION,
        }
    )

    plan_meta = {}
    min_count_per_tf = int(getattr(cfg.runtime, "min_count_per_tf", 0) or 0)
    for plan in cfg.generation.plan or []:
        constraints = plan.regulator_constraints
        plan_meta[str(plan.name)] = {
            "groups": list(constraints.groups or []),
            "min_count_by_regulator": dict(constraints.min_count_by_regulator or {}),
            "fixed_bp_min": _plan_fixed_bp_min(plan),
        }

    if placements_df.empty:
        used_by_library = {}
    else:
        used_by_library = {
            key: sub
            for key, sub in placements_df.groupby(
                ["input_name", "plan_name", "library_index", "library_hash"], dropna=False
            )
        }

    pool_lookup = None
    if pool_df is not None:
        pool_lookup = pool_df.copy()
        if "tf" not in pool_lookup.columns or "tfbs" not in pool_lookup.columns:
            raise RuntimeError("Pool data missing required tf/tfbs columns for run_metrics.")

    required_attempt_cols = {
        "input_name",
        "plan_name",
        "sampling_library_index",
        "sampling_library_hash",
        "library_tfs",
        "library_tfbs",
    }
    missing_attempt_cols = required_attempt_cols - set(attempts_df.columns)
    if missing_attempt_cols:
        raise RuntimeError(f"attempts.parquet missing required columns: {sorted(missing_attempt_cols)}")

    offered_by_library: dict[tuple[str, str, int, str], dict[str, int]] = {}
    offered_tfbs_by_library: dict[tuple[str, str, int, str], set[str]] = {}
    seen_offers: set[tuple[str, str, int, str]] = set()
    for _, row in attempts_df.iterrows():
        input_name = str(row.get("input_name") or "")
        plan_name = str(row.get("plan_name") or "")
        library_index = int(row.get("sampling_library_index") or 0)
        library_hash = str(row.get("sampling_library_hash") or "")
        key = (input_name, plan_name, library_index, library_hash)
        if key in seen_offers:
            continue
        seen_offers.add(key)
        tf_list = _ensure_list(row.get("library_tfs"))
        tfbs_list = _ensure_list(row.get("library_tfbs"))
        if len(tf_list) != len(tfbs_list):
            raise RuntimeError(
                f"attempts.parquet library_tfs/library_tfbs length mismatch for {input_name}/{plan_name}."
            )
        tf_counts: dict[str, int] = {}
        tfbs_set: set[str] = set()
        for tf, tfbs in zip(tf_list, tfbs_list):
            tf = str(tf).strip()
            tfbs = str(tfbs).strip()
            if not tf or not tfbs:
                continue
            tf_counts[tf] = tf_counts.get(tf, 0) + 1
            tfbs_set.add(tfbs)
        offered_by_library[key] = tf_counts
        offered_tfbs_by_library[key] = tfbs_set

    for _, build in builds_df.iterrows():
        input_name = str(build.get("input_name"))
        plan_name = str(build.get("plan_name"))
        library_index = int(build.get("library_index") or 0)
        library_hash = str(build.get("library_hash") or "")
        members = members_df[
            (members_df["input_name"].astype(str) == input_name)
            & (members_df["plan_name"].astype(str) == plan_name)
            & (members_df["library_index"].astype(int) == library_index)
        ]
        if library_hash:
            members = members[members["library_hash"].astype(str) == library_hash]
        if members.empty:
            raise RuntimeError(
                f"library_members.parquet missing rows for {input_name}/{plan_name} library_index={library_index}."
            )
        tf_counts = members["tf"].astype(str).value_counts()
        library_size = int(build.get("library_size") or len(members))
        unique_tf_count = int(tf_counts.size)
        unique_tfbs_count = int(members["tfbs"].astype(str).nunique())
        max_tf_dominance = float(tf_counts.max() / library_size) if library_size > 0 else 0.0
        entropy = _shannon_entropy(tf_counts.values.tolist())
        tfbs_lengths = members["tfbs"].astype(str).map(len)
        tfbs_length_sum = int(tfbs_lengths.sum())

        score_mean = None
        score_median = None
        if pool_lookup is not None and has_scores:
            merged = members.merge(
                pool_lookup[["input_name", "tf", "tfbs", "best_hit_score"]],
                on=["input_name", "tf", "tfbs"],
                how="left",
            )
            scores = pd.to_numeric(merged["best_hit_score"], errors="coerce").dropna()
            if not scores.empty:
                score_mean = float(scores.mean())
                score_median = float(scores.median())

        plan_info = plan_meta.get(plan_name, {})
        required_min = _required_min_length(
            members,
            groups=list(plan_info.get("groups") or []),
            min_count_by_regulator=dict(plan_info.get("min_count_by_regulator") or {}),
            min_count_per_tf=min_count_per_tf,
        )
        fixed_bp_min = int(plan_info.get("fixed_bp_min") or 0)
        sequence_length = int(cfg.generation.sequence_length)
        slack = None if required_min is None else int(sequence_length - fixed_bp_min - int(required_min))

        used = used_by_library.get((input_name, plan_name, library_index, library_hash))
        used_tfbs_unique = None
        used_tf_count = None
        used_fraction = None
        never_used_fraction = None
        if used is not None and not used.empty:
            used_tfbs_unique = int(used["tfbs"].astype(str).nunique())
            used_tf_count = int(used["tf"].astype(str).nunique())
            if unique_tfbs_count > 0:
                used_fraction = float(used_tfbs_unique / unique_tfbs_count)
                never_used_fraction = float(1.0 - used_fraction)

        records.append(
            {
                "metric_group": "library_health",
                "run_id": run_id,
                "created_at": created_at,
                "input_name": input_name,
                "plan_name": plan_name,
                "library_index": library_index,
                "library_hash": library_hash,
                "library_size": library_size,
                "unique_tf_count": unique_tf_count,
                "unique_tfbs_count": unique_tfbs_count,
                "tf_entropy": entropy,
                "max_tf_dominance": max_tf_dominance,
                "score_mean": score_mean,
                "score_median": score_median,
                "tfbs_length_sum": tfbs_length_sum,
                "required_min_length": required_min,
                "fixed_bp_min": fixed_bp_min,
                "slack_bp": slack,
                "sequence_length": sequence_length,
                "target_length": build.get("target_length"),
                "achieved_length": build.get("achieved_length"),
                "used_tfbs_unique": used_tfbs_unique,
                "used_tf_count": used_tf_count,
                "used_fraction": used_fraction,
                "never_used_tfbs_fraction": never_used_fraction,
            }
        )

        if used is not None and not used.empty:
            used_tf_counts = used["tf"].astype(str).value_counts().to_dict()
        else:
            used_tf_counts = {}
        offered_tf_counts = offered_by_library.get((input_name, plan_name, library_index, library_hash), {})
        for tf, offered_count in offered_tf_counts.items():
            used_count = int(used_tf_counts.get(tf, 0))
            used_fraction = float(used_count / offered_count) if offered_count else 0.0
            records.append(
                {
                    "metric_group": "offered_vs_used_tf",
                    "run_id": run_id,
                    "created_at": created_at,
                    "input_name": input_name,
                    "plan_name": plan_name,
                    "library_index": library_index,
                    "library_hash": library_hash,
                    "tf": str(tf),
                    "offered_count": int(offered_count),
                    "used_count": int(used_count),
                    "used_fraction": used_fraction,
                }
            )

    if pool_lookup is not None and has_tiers:
        pool_unique = pool_lookup[["input_name", "tf", "tfbs", "tier"]].drop_duplicates()
        if placements_df.empty:
            used_unique = pd.DataFrame(columns=["input_name", "tf", "tfbs"])
        else:
            used_unique = placements_df[["input_name", "tf", "tfbs"]].drop_duplicates()
        used_with_tier = used_unique.merge(pool_unique, on=["input_name", "tf", "tfbs"], how="inner")
        pool_counts = pool_unique.groupby(["input_name", "tf", "tier"], dropna=False).size()
        used_counts = used_with_tier.groupby(["input_name", "tf", "tier"], dropna=False).size()
        for key, pool_count in pool_counts.items():
            used_count = int(used_counts.get(key, 0))
            usage_rate = float(used_count / pool_count) if pool_count else 0.0
            records.append(
                {
                    "metric_group": "tier_enrichment",
                    "run_id": run_id,
                    "created_at": created_at,
                    "input_name": key[0],
                    "tf": key[1],
                    "tier": int(key[2]),
                    "pool_tfbs_count": int(pool_count),
                    "used_tfbs_count": used_count,
                    "usage_rate": usage_rate,
                }
            )

    if pool_lookup is not None and has_scores:
        pool_unique = pool_lookup[["input_name", "tf", "tfbs", "best_hit_score"]].drop_duplicates()
        pool_unique = _assign_score_quantiles(pool_unique, quantiles=DEFAULT_SCORE_QUANTILES)
        if placements_df.empty:
            used_unique = pd.DataFrame(columns=["input_name", "tf", "tfbs"])
        else:
            used_unique = placements_df[["input_name", "tf", "tfbs"]].drop_duplicates()
        used_with_quantile = used_unique.merge(
            pool_unique[["input_name", "tf", "tfbs", "score_quantile"]],
            on=["input_name", "tf", "tfbs"],
            how="inner",
        )
        pool_counts = pool_unique.groupby(["input_name", "tf", "score_quantile"], dropna=False).size()
        used_counts = used_with_quantile.groupby(["input_name", "tf", "score_quantile"], dropna=False).size()
        for (input_name, tf, quantile), pool_count in pool_counts.items():
            used_count = int(used_counts.get((input_name, tf, quantile), 0))
            total_pool = int(pool_counts.loc[(input_name, tf)].sum())
            total_used = int(used_counts.loc[(input_name, tf)].sum()) if (input_name, tf) in used_counts.index else 0
            pool_fraction = float(pool_count / total_pool) if total_pool else 0.0
            used_fraction = float(used_count / total_used) if total_used else 0.0
            enrichment = float(used_fraction / pool_fraction) if pool_fraction > 0 else 0.0
            records.append(
                {
                    "metric_group": "quantile_enrichment",
                    "run_id": run_id,
                    "created_at": created_at,
                    "input_name": input_name,
                    "tf": tf,
                    "quantile": int(quantile),
                    "pool_tfbs_count": int(pool_count),
                    "used_tfbs_count": int(used_count),
                    "pool_fraction": pool_fraction,
                    "used_fraction": used_fraction,
                    "enrichment": enrichment,
                }
            )

    if not events_df.empty and "event" in events_df.columns:
        pressure_events = events_df[events_df["event"] == "LIBRARY_SAMPLING_PRESSURE"]
        for _, row in pressure_events.iterrows():
            input_name = str(row.get("input_name") or "")
            plan_name = str(row.get("plan_name") or "")
            library_index = int(row.get("library_index") or 0)
            library_hash = str(row.get("library_hash") or "")
            weights = row.get("weight_by_tf") or {}
            weight_fracs = row.get("weight_fraction_by_tf") or {}
            usage_counts = row.get("usage_count_by_tf") or {}
            failure_counts = row.get("failure_count_by_tf") or {}
            if not isinstance(weights, dict):
                raise RuntimeError("events.jsonl weight_by_tf must be a dict.")
            if weight_fracs and not isinstance(weight_fracs, dict):
                raise RuntimeError("events.jsonl weight_fraction_by_tf must be a dict.")
            if usage_counts and not isinstance(usage_counts, dict):
                raise RuntimeError("events.jsonl usage_count_by_tf must be a dict.")
            if failure_counts and not isinstance(failure_counts, dict):
                raise RuntimeError("events.jsonl failure_count_by_tf must be a dict.")
            if not weight_fracs:
                total = float(sum(float(v) for v in weights.values())) if weights else 0.0
                weight_fracs = {k: (float(v) / total if total > 0 else 0.0) for k, v in weights.items()}
            for tf, weight in weights.items():
                records.append(
                    {
                        "metric_group": "sampling_pressure",
                        "run_id": run_id,
                        "created_at": created_at,
                        "input_name": input_name,
                        "plan_name": plan_name,
                        "library_index": library_index,
                        "library_hash": library_hash,
                        "tf": str(tf),
                        "weight": float(weight),
                        "weight_fraction": float(weight_fracs.get(tf, 0.0)),
                        "usage_count": int(usage_counts.get(tf, 0)) if usage_counts else 0,
                        "failure_count": int(failure_counts.get(tf, 0)) if failure_counts else 0,
                        "sampling_strategy": str(row.get("sampling_strategy") or ""),
                    }
                )

    return pd.DataFrame.from_records(records)


def write_run_metrics(*, cfg, run_root: Path) -> Path:
    tables_root = run_tables_root(run_root)
    tables_root.mkdir(parents=True, exist_ok=True)
    metrics_df = build_run_metrics(cfg=cfg, run_root=run_root)
    out_path = tables_root / "run_metrics.parquet"
    metrics_df.to_parquet(out_path, index=False)
    return out_path
