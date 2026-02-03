"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/reporting.py

Audit-grade reporting helpers for DenseGen runs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..adapters.outputs import load_records_from_config
from ..config import RootConfig, resolve_outputs_scoped_path, resolve_run_root
from .artifacts.pool import POOL_MODE_TFBS, load_pool_data
from .motif_labels import input_motifs, motif_display_name
from .pipeline.plan_pools import build_plan_pools, plan_pool_label
from .run_manifest import load_run_manifest
from .run_paths import (
    candidates_root,
    dense_arrays_path,
    run_manifest_path,
    run_outputs_root,
    run_tables_root,
)

log = logging.getLogger(__name__)


def _dg(col: str) -> str:
    return col if col.startswith("densegen__") else f"densegen__{col}"


def _as_py(val):
    if hasattr(val, "as_py"):
        return val.as_py()
    return val


def _ensure_list_of_dicts(val) -> list[dict]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    val = _as_py(val)
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                val = json.loads(s)
            except Exception as exc:
                raise ValueError(f"Failed to parse JSON list field: {s[:80]}") from exc
    if isinstance(val, (list, tuple, np.ndarray)):
        out: list[dict] = []
        for item in list(val):
            item = _as_py(item)
            if not isinstance(item, dict):
                raise ValueError("Expected list of dicts; found non-dict entries.")
            out.append(item)
        return out
    raise ValueError(f"Expected list of dicts; got {type(val).__name__}.")


def _solution_id(row: pd.Series) -> str:
    if "solution_id" in row and isinstance(row["solution_id"], str) and row["solution_id"]:
        return row["solution_id"]
    if "id" in row and isinstance(row["id"], str) and row["id"]:
        return row["id"]
    raise ValueError("Output records missing solution_id/id; regenerate outputs before reporting.")


def _ensure_list(val: Any) -> list:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    val = _as_py(val)
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return list(parsed)
            except Exception:
                return []
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    return []


def _load_events(events_path: Path) -> pd.DataFrame:
    if not events_path.exists():
        return pd.DataFrame(columns=["event", "created_at", "input_name", "plan_name", "library_index", "library_hash"])
    rows = []
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        rows.append(payload)
    return pd.DataFrame(rows)


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
        for entry in used_detail:
            tf = str(entry.get("tf") or "").strip()
            tfbs = str(entry.get("tfbs") or "").strip()
            if not tf and not tfbs:
                continue
            records.append(
                {
                    "solution_id": seq_id,
                    "library_hash": str(row.get(lib_hash_col) or ""),
                    "library_index": int(row.get(lib_index_col) or 0),
                    "plan": str(row.get(plan_col) or ""),
                    "input_name": str(row.get(input_col) or ""),
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


@dataclass
class ReportBundle:
    run_report: dict
    tables: Dict[str, pd.DataFrame]
    plots: dict[str, list[str]] | None = None


def collect_report_data(
    root_cfg: RootConfig,
    cfg_path: Path,
    *,
    include_combinatorics: bool = False,
    strict: bool = False,
) -> ReportBundle:
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    outputs_root = run_outputs_root(run_root)
    tables_root = run_tables_root(run_root)
    warnings: list[str] = []
    cols = [
        "id",
        "sequence",
        _dg("plan"),
        _dg("input_name"),
        _dg("sampling_library_hash"),
        _dg("sampling_library_index"),
        _dg("used_tfbs_detail"),
        _dg("required_regulators"),
        _dg("covers_required_regulators"),
        _dg("covers_all_tfs_in_solution"),
        _dg("min_required_regulators"),
        _dg("used_tf_list"),
        _dg("min_count_per_tf"),
    ]
    try:
        df, source_label = load_records_from_config(root_cfg, cfg_path, columns=cols)
    except Exception as exc:
        message = f"No output records available; report will focus on Stage-A/Stage-B diagnostics. ({exc})"
        if strict:
            raise ValueError(message) from exc
        warnings.append(message)
        df = pd.DataFrame(columns=cols)
        source_label = "missing"
    if df.empty:
        warnings.append("Output records are empty; solution-focused sections will be blank.")

    used_df = _explode_used(df)
    attempts_path = tables_root / "attempts.parquet"
    if not attempts_path.exists():
        message = "outputs/tables/attempts.parquet is missing; library utilization summaries may be incomplete."
        if strict:
            raise ValueError(message)
        warnings.append(message)
        attempts_df = pd.DataFrame()
    else:
        attempts_df = pd.read_parquet(attempts_path)
    library_df = _explode_library_from_attempts(attempts_df)
    solutions_path = tables_root / "solutions.parquet"
    if not solutions_path.exists():
        message = (
            "outputs/tables/solutions.parquet is missing; solution previews and composition summaries will be skipped."
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)
        solutions_df = pd.DataFrame()
    else:
        try:
            solutions_df = pd.read_parquet(solutions_path)
        except Exception as exc:
            message = f"Failed to load solutions.parquet; skipping solution tables. ({exc})"
            if strict:
                raise ValueError(message) from exc
            warnings.append(message)
            solutions_df = pd.DataFrame()
    tables: Dict[str, pd.DataFrame] = {}
    tables["solutions"] = solutions_df
    display_map_by_input: dict[str, dict[str, str]] = {}
    inputs_by_name = {inp.name: inp for inp in root_cfg.densegen.inputs}
    plan_items = list(root_cfg.densegen.generation.resolve_plan())
    for plan in plan_items:
        include_inputs = list(getattr(plan, "include_inputs", []) or [])
        if not include_inputs:
            raise ValueError(f"plan '{plan.name}' is missing include_inputs for plan-scoped pooling")
        mapping: dict[str, str] = {}
        for input_name in include_inputs:
            inp = inputs_by_name.get(input_name)
            if inp is None:
                continue
            for motif_id, name in input_motifs(inp, cfg_path):
                if motif_id and name and motif_id not in mapping:
                    mapping[motif_id] = name
        if mapping:
            pool_name = plan_pool_label(str(plan.name))
            display_map_by_input[pool_name] = mapping

    def _display_tf_label(input_name: str, tf: str) -> str:
        mapping = display_map_by_input.get(input_name, {})
        return mapping.get(tf, motif_display_name(tf, None))

    stage_a_tiers = pd.DataFrame(columns=["input_name", "tf", "tier", "count", "total"])
    stage_a_score_summary = pd.DataFrame(
        columns=[
            "input_name",
            "tf",
            "metric",
            "count",
            "min",
            "p10",
            "p50",
            "p90",
            "max",
        ]
    )
    pool_dir = outputs_root / "pools"
    if pool_dir.exists():
        try:
            _artifact, pool_data = load_pool_data(pool_dir)
            plan_pools = build_plan_pools(plan_items=plan_items, pool_data=pool_data)
            rows: list[dict[str, Any]] = []
            score_rows: list[dict[str, Any]] = []
            for plan in plan_items:
                spec = plan_pools.get(str(plan.name))
                if spec is None or spec.pool.pool_mode != POOL_MODE_TFBS or spec.pool.df is None:
                    continue
                df_pool = spec.pool.df
                if "tf" not in df_pool.columns:
                    continue
                if "tier" in df_pool.columns:
                    total_counts = df_pool.groupby("tf").size().to_dict()
                    grouped = df_pool.groupby(["tf", "tier"])
                    for (tf, tier), group in grouped:
                        rows.append(
                            {
                                "input_name": spec.pool.name,
                                "tf": tf,
                                "tier": int(tier),
                                "count": int(len(group)),
                                "total": int(total_counts.get(tf, len(group))),
                            }
                        )
                for tf, sub in df_pool.groupby("tf"):
                    if sub.empty:
                        continue
                    if "best_hit_score" in sub.columns:
                        vals = pd.to_numeric(sub["best_hit_score"], errors="coerce").dropna()
                        if not vals.empty:
                            score_rows.append(
                                {
                                    "input_name": spec.pool.name,
                                    "tf": tf,
                                    "metric": "best_hit_score",
                                    "count": int(len(vals)),
                                    "min": float(vals.min()),
                                    "p10": float(vals.quantile(0.1)),
                                    "p50": float(vals.quantile(0.5)),
                                    "p90": float(vals.quantile(0.9)),
                                    "max": float(vals.max()),
                                }
                            )
                        if "score" in sub.columns:
                            vals = pd.to_numeric(sub["score"], errors="coerce").dropna()
                            if not vals.empty:
                                score_rows.append(
                                    {
                                        "input_name": spec.pool.name,
                                        "tf": tf,
                                        "metric": "densegen_score",
                                        "count": int(len(vals)),
                                        "min": float(vals.min()),
                                        "p10": float(vals.quantile(0.1)),
                                        "p50": float(vals.quantile(0.5)),
                                        "p90": float(vals.quantile(0.9)),
                                        "max": float(vals.max()),
                                    }
                                )
            if rows:
                stage_a_tiers = pd.DataFrame(rows)
            if score_rows:
                stage_a_score_summary = pd.DataFrame(score_rows)
        except Exception:
            log.warning("Failed to load Stage-A pool tiers for report.", exc_info=True)

    if not stage_a_tiers.empty and "input_name" in stage_a_tiers.columns and "tf" in stage_a_tiers.columns:
        stage_a_tiers["tf"] = stage_a_tiers.apply(
            lambda row: _display_tf_label(str(row.get("input_name") or ""), str(row.get("tf") or "")),
            axis=1,
        )
    if (
        not stage_a_score_summary.empty
        and "input_name" in stage_a_score_summary.columns
        and "tf" in stage_a_score_summary.columns
    ):
        stage_a_score_summary["tf"] = stage_a_score_summary.apply(
            lambda row: _display_tf_label(str(row.get("input_name") or ""), str(row.get("tf") or "")),
            axis=1,
        )

    tables["stage_a_tiers"] = stage_a_tiers
    tables["stage_a_score_summary"] = stage_a_score_summary

    def _candidate_logging_enabled() -> bool:
        for inp in root_cfg.densegen.inputs:
            sampling = getattr(inp, "sampling", None)
            if sampling is None:
                continue
            if getattr(sampling, "keep_all_candidates_debug", False):
                return True
        return False

    candidate_logging = _candidate_logging_enabled()
    candidates_summary = pd.DataFrame(
        columns=[
            "input_name",
            "motif_id",
            "motif_label",
            "scoring_backend",
            "total_candidates",
            "accepted",
            "selected",
            "rejected",
        ]
    )
    candidates_dir = candidates_root(outputs_root, root_cfg.densegen.run.id)
    cand_summary_path = candidates_dir / "candidates_summary.parquet"
    if cand_summary_path.exists():
        if candidate_logging:
            try:
                candidates_summary = pd.read_parquet(cand_summary_path)
            except Exception:
                log.warning("Failed to load candidates_summary.parquet for report.", exc_info=True)
        else:
            warnings.append(
                "Candidate summary exists but keep_all_candidates_debug is false; "
                "candidate artifacts may be stale. Enable keep_all_candidates_debug to refresh."
            )
    tables["candidates_summary"] = candidates_summary

    library_summary = pd.DataFrame(
        columns=[
            "input_name",
            "plan_name",
            "libraries",
            "library_size_min",
            "library_size_median",
            "library_size_max",
            "total_bp_min",
            "total_bp_median",
            "total_bp_max",
        ]
    )
    if not library_df.empty:
        per_library = (
            library_df.groupby(["library_hash", "library_index", "input_name", "plan_name"])
            .agg(
                size=("tfbs", "size"),
                total_bp=("tfbs_length", "sum"),
            )
            .reset_index()
        )
        library_summary = (
            per_library.groupby(["input_name", "plan_name"])
            .agg(
                libraries=("library_index", "size"),
                library_size_min=("size", "min"),
                library_size_median=("size", "median"),
                library_size_max=("size", "max"),
                total_bp_min=("total_bp", "min"),
                total_bp_median=("total_bp", "median"),
                total_bp_max=("total_bp", "max"),
            )
            .reset_index()
        )
        library_summary["libraries"] = (
            pd.to_numeric(library_summary["libraries"], errors="coerce").fillna(0).astype(int)
        )

    tables["library_summary"] = library_summary

    plan_summary = pd.DataFrame(
        columns=[
            "input_name",
            "plan_name",
            "outputs",
            "unique_solutions",
            "coverage_required_rate",
            "coverage_all_tfs_rate",
            "avg_used_tf_count",
            "min_required_regulators",
            "min_count_per_tf",
        ]
    )
    if not df.empty:
        df_plan = df.copy()
        df_plan["_used_tf_count"] = df_plan[_dg("used_tf_list")].apply(lambda x: len(_ensure_list(x)))
        df_plan["_covers_required"] = df_plan[_dg("covers_required_regulators")].fillna(False).astype(bool)
        df_plan["_covers_all"] = df_plan[_dg("covers_all_tfs_in_solution")].fillna(False).astype(bool)
        plan_summary = (
            df_plan.groupby([_dg("input_name"), _dg("plan")])
            .agg(
                outputs=("sequence", "size"),
                unique_solutions=("id", pd.Series.nunique),
                coverage_required_rate=("_covers_required", "mean"),
                coverage_all_tfs_rate=("_covers_all", "mean"),
                avg_used_tf_count=("_used_tf_count", "mean"),
                min_required_regulators=(_dg("min_required_regulators"), "max"),
                min_count_per_tf=(_dg("min_count_per_tf"), "max"),
            )
            .reset_index()
            .rename(columns={_dg("input_name"): "input_name", _dg("plan"): "plan_name"})
        )
    tables["plan_summary"] = plan_summary

    offered_tf = pd.DataFrame(columns=["library_hash", "tf", "offered_instances", "offered_unique_tfbs"])
    offered_tfbs = pd.DataFrame(columns=["library_hash", "tf", "tfbs", "offered_instances"])
    if not library_df.empty:
        offered_tf = (
            library_df.groupby(["library_hash", "tf"])
            .agg(offered_instances=("tfbs", "size"), offered_unique_tfbs=("tfbs", pd.Series.nunique))
            .reset_index()
        )
        offered_tfbs = (
            library_df.groupby(["library_hash", "tf", "tfbs"]).agg(offered_instances=("tfbs", "size")).reset_index()
        )

    if not used_df.empty:
        used_tf = (
            used_df.groupby(["library_hash", "plan", "tf"])
            .agg(
                used_placements=("tf", "size"),
                used_unique_tfbs=("tfbs", pd.Series.nunique),
                used_sequences=("solution_id", pd.Series.nunique),
            )
            .reset_index()
        )
        used_tfbs = (
            used_df.groupby(["library_hash", "plan", "tf", "tfbs"])
            .agg(used_placements=("tfbs", "size"), used_sequences=("solution_id", pd.Series.nunique))
            .reset_index()
        )
    else:
        used_tf = pd.DataFrame(
            columns=["library_hash", "plan", "tf", "used_placements", "used_unique_tfbs", "used_sequences"]
        )
        used_tfbs = pd.DataFrame(columns=["library_hash", "plan", "tf", "tfbs", "used_placements", "used_sequences"])

    total_sequences = (
        used_df.groupby("library_hash")["solution_id"].nunique().reset_index(name="total_sequences")
        if not used_df.empty
        else pd.DataFrame(columns=["library_hash", "total_sequences"])
    )
    used_tf_any = (
        used_tf.groupby(["library_hash", "tf"])
        .agg(
            used_placements=("used_placements", "sum"),
            used_unique_tfbs=("used_unique_tfbs", "sum"),
            used_sequences=("used_sequences", "sum"),
        )
        .reset_index()
    )
    offered_vs_used_tf = offered_tf.merge(used_tf_any, on=["library_hash", "tf"], how="left").merge(
        total_sequences, on="library_hash", how="left"
    )
    for col in ["used_placements", "used_unique_tfbs", "used_sequences", "total_sequences"]:
        if col in offered_vs_used_tf.columns:
            offered_vs_used_tf[col] = pd.to_numeric(offered_vs_used_tf[col], errors="coerce").fillna(0).astype(int)
    offered_vs_used_tf["utilization_any"] = offered_vs_used_tf.apply(
        lambda r: (r["used_sequences"] / r["total_sequences"]) if r["total_sequences"] else 0.0, axis=1
    )
    offered_vs_used_tf["utilization_placements_per_offered"] = offered_vs_used_tf.apply(
        lambda r: (r["used_placements"] / r["offered_instances"]) if r["offered_instances"] else 0.0, axis=1
    )

    used_tfbs_any = (
        used_tfbs.groupby(["library_hash", "tf", "tfbs"])
        .agg(used_placements=("used_placements", "sum"), used_sequences=("used_sequences", "sum"))
        .reset_index()
    )
    offered_vs_used_tfbs = offered_tfbs.merge(used_tfbs_any, on=["library_hash", "tf", "tfbs"], how="left")
    for col in ["used_placements", "used_sequences"]:
        if col in offered_vs_used_tfbs.columns:
            offered_vs_used_tfbs[col] = pd.to_numeric(offered_vs_used_tfbs[col], errors="coerce").fillna(0).astype(int)

    tables["offered_vs_used_tf"] = offered_vs_used_tf
    tables["offered_vs_used_tfbs"] = offered_vs_used_tfbs
    tables["attempts"] = attempts_df

    events_path = outputs_root / "meta" / "events.jsonl"
    events_df = _load_events(events_path)
    tables["events"] = events_df

    if include_combinatorics:
        tables["tf_cooccurrence"] = _compute_cooccurrence(used_df)
        tables["tf_adjacency"] = _compute_adjacency(used_df)

    composition_path = tables_root / "composition.parquet"
    if composition_path.exists():
        try:
            composition = pd.read_parquet(composition_path)
            if not composition.empty and "tf" in composition.columns and "input_name" in composition.columns:
                composition["tf"] = composition.apply(
                    lambda row: _display_tf_label(str(row.get("input_name") or ""), str(row.get("tf") or "")),
                    axis=1,
                )
            tables["composition"] = composition
        except Exception:
            log.warning("Failed to load composition.parquet for report tables.", exc_info=True)

    library_hashes = df[_dg("sampling_library_hash")].dropna().unique().tolist()
    tf_counts = used_df["tf"].value_counts().to_dict() if not used_df.empty else {}
    tfbs_counts = used_df["tfbs"].value_counts().to_dict() if not used_df.empty else {}
    diversity_entropy_tfbs = _normalized_entropy_from_counts({str(k): int(v) for k, v in tfbs_counts.items()})
    tf_usage_stats = _usage_stats_from_counts({str(k): int(v) for k, v in tf_counts.items()})
    tfbs_usage_stats = _usage_stats_from_counts({str(k): int(v) for k, v in tfbs_counts.items()})
    attempts_total = int(len(attempts_df)) if attempts_df is not None else 0
    attempts_success = int((attempts_df["status"] == "success").sum()) if "status" in attempts_df else 0
    attempts_failed = max(0, attempts_total - attempts_success)

    lib_tf_total = int(offered_vs_used_tf["tf"].nunique()) if not offered_vs_used_tf.empty else 0
    used_tf_total = (
        int(offered_vs_used_tf[offered_vs_used_tf["used_sequences"] > 0]["tf"].nunique())
        if not offered_vs_used_tf.empty
        else 0
    )
    lib_tfbs_total = (
        int(offered_vs_used_tfbs.drop_duplicates(["tf", "tfbs"]).shape[0]) if not offered_vs_used_tfbs.empty else 0
    )
    used_tfbs_total = (
        int(offered_vs_used_tfbs[offered_vs_used_tfbs["used_sequences"] > 0].drop_duplicates(["tf", "tfbs"]).shape[0])
        if not offered_vs_used_tfbs.empty
        else 0
    )
    tf_coverage = used_tf_total / max(1, lib_tf_total) if lib_tf_total else None
    tfbs_coverage = used_tfbs_total / max(1, lib_tfbs_total) if lib_tfbs_total else None

    run_report = {
        "run_root": os.path.relpath(run_root, run_root),
        "schema_version": root_cfg.densegen.schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_source": source_label,
        "warnings": warnings,
        "output_rows": int(len(df)),
        "output_unique_sequences": int(df["id"].nunique()) if "id" in df.columns else int(len(df)),
        "libraries_in_outputs": int(len(set(library_hashes))),
        "plans": sorted({str(x) for x in df[_dg("plan")].dropna().tolist()}),
        "diversity_unique_tfs": int(len(tf_counts)),
        "diversity_unique_tfbs": int(len(tfbs_counts)),
        "diversity_entropy_tfbs": diversity_entropy_tfbs,
        "coverage": {
            "tf_coverage": tf_coverage,
            "tfbs_coverage": tfbs_coverage,
            "library_tf_count": int(lib_tf_total),
            "library_tfbs_count": int(lib_tfbs_total),
            "used_tf_count": int(len(tf_counts)),
            "used_tfbs_count": int(len(tfbs_counts)),
        },
        "usage_stats": {
            "tf": tf_usage_stats,
            "tfbs": tfbs_usage_stats,
        },
        "attempts_total": attempts_total,
        "attempts_success": attempts_success,
        "attempts_failed": attempts_failed,
        "attempts_path": os.path.relpath(attempts_path, run_root) if attempts_path.exists() else None,
        "solutions_path": os.path.relpath(solutions_path, run_root) if solutions_path.exists() else None,
        "events_path": os.path.relpath(events_path, run_root) if events_path.exists() else None,
        "candidates_path": os.path.relpath(candidates_dir / "candidates.parquet", run_root)
        if candidate_logging and (candidates_dir / "candidates.parquet").exists()
        else None,
        "candidates_summary_path": os.path.relpath(cand_summary_path, run_root)
        if candidate_logging and cand_summary_path.exists()
        else None,
        "outputs_path": os.path.relpath(dense_arrays_path(run_root), run_root),
        "effective_config_path": os.path.relpath(outputs_root / "meta" / "effective_config.json", run_root)
        if (outputs_root / "meta" / "effective_config.json").exists()
        else None,
    }
    manifest_path = run_manifest_path(run_root)
    if manifest_path.exists():
        try:
            manifest = load_run_manifest(manifest_path)
            run_report.update(
                {
                    "run_id": manifest.run_id,
                    "dense_arrays_version": manifest.dense_arrays_version,
                    "dense_arrays_version_source": manifest.dense_arrays_version_source,
                }
            )
        except Exception:
            log.warning("Failed to read run_manifest.json for report metadata.", exc_info=True)

    return ReportBundle(run_report=run_report, tables=tables, plots={})


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "densegen"


def _markdown_table(df: pd.DataFrame, *, columns: list[str] | None = None, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return ""
    cols = columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return ""
    sub = df[cols].head(max_rows)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _load_plot_manifest(manifest_path: Path, *, report_root: Path) -> dict[str, list[str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"plot_manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    plots: dict[str, list[str]] = {}
    for entry in payload.get("plots", []):
        name = str(entry.get("name") or "").strip()
        rel = str(entry.get("path") or "").strip()
        if not name or not rel:
            raise ValueError("plot_manifest entries must include name and path.")
        plot_path = manifest_path.parent / rel
        if not plot_path.exists():
            raise FileNotFoundError(f"Plot listed in plot_manifest is missing: {plot_path}")
        rel_to_report = os.path.relpath(plot_path, report_root)
        plots.setdefault(name, []).append(rel_to_report)
    return plots


def write_report(
    root_cfg: RootConfig,
    cfg_path: Path,
    *,
    out_dir: str | Path = "outputs/report",
    include_combinatorics: bool = False,
    include_plots: bool = False,
    strict: bool = False,
    formats: set[str] | None = None,
) -> ReportBundle:
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    out_path = resolve_outputs_scoped_path(cfg_path, run_root, str(out_dir), label="report.out")
    out_path.mkdir(parents=True, exist_ok=True)
    bundle = collect_report_data(
        root_cfg,
        cfg_path,
        include_combinatorics=include_combinatorics,
        strict=strict,
    )
    composition = bundle.tables.get("composition")
    if composition is not None and not composition.empty:
        bundle.run_report["composition_rows"] = int(len(composition))
    if include_plots:
        plot_manifest = run_root / "outputs" / "plots" / "plot_manifest.json"
        plots = _load_plot_manifest(plot_manifest, report_root=out_path)
        bundle.plots = plots
        bundle.run_report["plot_manifest"] = str(plot_manifest.relative_to(run_root))
    formats = {f.lower() for f in (formats or {"json", "md"})}
    if "all" in formats:
        formats = {"json", "md", "html"}
    if "json" in formats:
        report_path = out_path / "report.json"
        report_path.write_text(json.dumps(bundle.run_report, indent=2, sort_keys=True))
    if "md" in formats:
        report_md = out_path / "report.md"
        _write_report_md(report_md, bundle)
    if "html" in formats:
        report_html = out_path / "report.html"
        _write_report_html(report_html, bundle)
    return bundle


def _render_report_md(bundle: ReportBundle) -> str:
    report = bundle.run_report
    coverage = report.get("coverage") or {}
    usage_stats = report.get("usage_stats") or {}
    tf_cov = coverage.get("tf_coverage")
    tfbs_cov = coverage.get("tfbs_coverage")
    tf_cov_label = "-"
    tfbs_cov_label = "-"
    if tf_cov is not None:
        tf_cov_label = f"{float(tf_cov):.1%} ({coverage.get('used_tf_count', 0)}/{coverage.get('library_tf_count', 0)})"
    if tfbs_cov is not None:
        tfbs_cov_label = (
            f"{float(tfbs_cov):.1%} ({coverage.get('used_tfbs_count', 0)}/{coverage.get('library_tfbs_count', 0)})"
        )
    tf_usage = usage_stats.get("tf") or {}
    tfbs_usage = usage_stats.get("tfbs") or {}
    tf_usage_label = "-"
    tfbs_usage_label = "-"
    if tf_usage.get("min") is not None:
        tf_usage_label = f"{tf_usage.get('min'):.0f}/{tf_usage.get('median'):.0f}/{tf_usage.get('max'):.0f}"
    if tfbs_usage.get("min") is not None:
        tfbs_usage_label = f"{tfbs_usage.get('min'):.0f}/{tfbs_usage.get('median'):.0f}/{tfbs_usage.get('max'):.0f}"
    lines = [
        "# DenseGen Report",
        "",
        f"- Run root: {report.get('run_root')}",
        f"- Schema: {report.get('schema_version')}",
        f"- Output rows: {report.get('output_rows')}",
        f"- Unique sequences: {report.get('output_unique_sequences')}",
        f"- Libraries in outputs: {report.get('libraries_in_outputs')}",
        f"- Diversity (unique TFs): {report.get('diversity_unique_tfs')}",
        f"- Diversity (unique TFBS): {report.get('diversity_unique_tfbs')}",
        f"- Diversity entropy (TFBS): {report.get('diversity_entropy_tfbs')}",
        f"- Coverage (TFs used / offered): {tf_cov_label}",
        f"- Coverage (TFBS used / offered): {tfbs_cov_label}",
        f"- TF usage (min/median/max): {tf_usage_label}",
        f"- TFBS usage (min/median/max): {tfbs_usage_label}",
        f"- Warnings: {len(report.get('warnings') or [])}",
        "",
        "## Outputs",
        "- outputs/tables/dense_arrays.parquet",
        "- outputs/tables/attempts.parquet",
        "- outputs/tables/solutions.parquet",
        "- outputs/tables/composition.parquet",
        "- outputs/libraries/library_builds.parquet",
        "- outputs/libraries/library_members.parquet",
        "- outputs/pools/pool_manifest.json",
        "- outputs/meta/effective_config.json",
        "- outputs/meta/events.jsonl",
        "- outputs/pools/candidates/candidates.parquet (when candidate logging is enabled)",
        "- outputs/pools/candidates/candidates_summary.parquet (when candidate logging is enabled)",
        "- outputs/plots/ (visual artifacts; run `dense plot` to populate)",
        "- outputs/plots/plot_manifest.json (plot index for reports)",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Notes"])
        for warning in warnings:
            lines.append(f"- {warning}")
    stage_a_tiers = bundle.tables.get("stage_a_tiers")
    if stage_a_tiers is not None and not stage_a_tiers.empty:
        lines.extend(["", "## Stage-A tiers"])
        for (input_name, tf), sub in stage_a_tiers.groupby(["input_name", "tf"]):
            sub = sub.sort_values("tier")
            parts = []
            for _, row in sub.iterrows():
                tier = int(row.get("tier") or 0)
                count = int(row.get("count") or 0)
                parts.append(f"tier{tier}:{count}")
            lines.append(f"- {input_name}/{tf}: " + " ".join(parts))
    stage_a_score_summary = bundle.tables.get("stage_a_score_summary")
    if stage_a_score_summary is not None and not stage_a_score_summary.empty:
        lines.extend(["", "## Stage-A score summary (per TF)"])
        lines.append(
            _markdown_table(
                stage_a_score_summary,
                columns=["input_name", "tf", "metric", "count", "min", "p10", "p50", "p90", "max"],
                max_rows=20,
            )
        )
    candidates_summary = bundle.tables.get("candidates_summary")
    if candidates_summary is not None and not candidates_summary.empty:
        lines.extend(["", "## Candidate mining summary"])
        lines.append(
            _markdown_table(
                candidates_summary,
                columns=[
                    "input_name",
                    "motif_label",
                    "motif_id",
                    "scoring_backend",
                    "total_candidates",
                    "accepted",
                    "selected",
                    "rejected",
                ],
                max_rows=20,
            )
        )
    plan_summary = bundle.tables.get("plan_summary")
    if plan_summary is not None and not plan_summary.empty:
        summary = plan_summary.copy()
        for col in ("coverage_required_rate", "coverage_all_tfs_rate"):
            if col in summary.columns:
                summary[col] = summary[col].apply(
                    lambda v: f"{float(v):.1%}" if v is not None and not pd.isna(v) else "-"
                )
        lines.extend(["", "## Plan coverage summary"])
        lines.append(
            _markdown_table(
                summary,
                columns=[
                    "input_name",
                    "plan_name",
                    "outputs",
                    "unique_solutions",
                    "coverage_required_rate",
                    "coverage_all_tfs_rate",
                    "avg_used_tf_count",
                    "min_required_regulators",
                ],
                max_rows=20,
            )
        )
    library_summary = bundle.tables.get("library_summary")
    if library_summary is not None and not library_summary.empty:
        lines.extend(["", "## Library summary (aggregate)"])
        lines.append(
            _markdown_table(
                library_summary,
                columns=[
                    "input_name",
                    "plan_name",
                    "libraries",
                    "library_size_min",
                    "library_size_median",
                    "library_size_max",
                    "total_bp_min",
                    "total_bp_median",
                    "total_bp_max",
                ],
                max_rows=20,
            )
        )
    events = bundle.tables.get("events")
    if events is not None and not events.empty and "event" in events.columns:
        event_summary = (
            events.groupby("event")
            .agg(count=("event", "size"), last_created_at=("created_at", "max"))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        lines.extend(["", "## Events summary"])
        lines.append(_markdown_table(event_summary, columns=["event", "count", "last_created_at"], max_rows=10))
    solutions = bundle.tables.get("solutions")
    if solutions is not None and not solutions.empty:
        preview = solutions.copy()
        if "sequence" in preview.columns:
            preview["sequence_len"] = preview["sequence"].apply(lambda s: len(s) if isinstance(s, str) else 0)
            preview["sequence_preview"] = preview["sequence"].apply(
                lambda s: (s[:24] + "…") if isinstance(s, str) and len(s) > 25 else s
            )
        lines.extend(["", "## Solutions (sample)"])
        lines.append(
            _markdown_table(
                preview,
                columns=[
                    "solution_id",
                    "attempt_id",
                    "input_name",
                    "plan_name",
                    "sampling_library_hash",
                    "sequence_len",
                    "sequence_preview",
                ],
                max_rows=10,
            )
        )
    else:
        lines.extend(["", "## Solutions"])
        lines.append("- No solutions found yet. Review attempts/events and adjust constraints or runtime settings.")
    composition = bundle.tables.get("composition")
    if composition is not None and not composition.empty:
        comp = composition.copy()
        if "library_hash" in comp.columns:
            comp["library_hash"] = comp["library_hash"].apply(lambda v: str(v)[:8])
        lines.extend(["", "## Composition (sample)"])
        lines.append(
            _markdown_table(
                comp,
                columns=[
                    "solution_id",
                    "attempt_id",
                    "placement_index",
                    "tf",
                    "tfbs",
                    "motif_id",
                    "tfbs_id",
                    "orientation",
                    "offset",
                    "library_hash",
                ],
                max_rows=12,
            )
        )
        comp_rows = report.get("composition_rows")
        if comp_rows is not None:
            lines.append(f"- Full composition rows: {comp_rows}")
    if bundle.plots:
        lines.extend(["", "## Report plots"])
        for plot_name, paths in bundle.plots.items():
            lines.append(f"- {plot_name}:")
            for rel_path in paths:
                lines.append(f"  - {rel_path}")
    return "\n".join(lines) + "\n"


def _write_report_md(path: Path, bundle: ReportBundle) -> None:
    path.write_text(_render_report_md(bundle))


def _write_report_html(path: Path, bundle: ReportBundle) -> None:
    md = _render_report_md(bundle)
    body = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    img_sections: list[str] = []
    if bundle.plots:
        for plot_name, paths in bundle.plots.items():
            for rel_path in paths:
                img_sections.append(
                    f'<div><h3>{plot_name}</h3><img src="{rel_path}" '
                    'style="max-width:100%;height:auto;border:1px solid #ddd;"/></div>'
                )
    img_html = "\n".join(img_sections)
    html = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8"/>',
            "<title>DenseGen Report</title>",
            "<style>body{font-family:ui-monospace,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;"
            "padding:24px;background:#fafafa;color:#111;}pre{white-space:pre-wrap;}</style>",
            "</head>",
            "<body>",
            "<pre>",
            body,
            "</pre>",
            img_html,
            "</body>",
            "</html>",
        ]
    )
    path.write_text(html)
