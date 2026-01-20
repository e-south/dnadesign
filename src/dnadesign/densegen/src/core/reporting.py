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

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..adapters.outputs import load_records_from_config
from ..config import RootConfig, resolve_run_root, resolve_run_scoped_path
from ..utils.mpl_utils import ensure_mpl_cache_dir
from .artifacts.pool import POOL_MODE_TFBS, load_pool_artifact
from .run_manifest import load_run_manifest
from .run_paths import run_manifest_path, run_outputs_root

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


def _sequence_id(row: pd.Series, fallback: str) -> str:
    if "id" in row and isinstance(row["id"], str) and row["id"]:
        return row["id"]
    seq = row.get("sequence")
    if isinstance(seq, str) and seq:
        return hashlib.sha256(seq.encode("utf-8")).hexdigest()
    return fallback


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
        seq_id = _sequence_id(row, fallback=f"row-{idx}")
        for entry in used_detail:
            tf = str(entry.get("tf") or "").strip()
            tfbs = str(entry.get("tfbs") or "").strip()
            if not tf and not tfbs:
                continue
            records.append(
                {
                    "sequence_id": seq_id,
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


def _summarize_failure_top_tfbs(attempts_df: pd.DataFrame, *, top: int = 5) -> list[dict]:
    if attempts_df.empty or "status" not in attempts_df.columns:
        return []
    failed = attempts_df[attempts_df["status"] != "success"]
    if failed.empty:
        return []
    counts: dict[tuple[str, str], int] = {}
    reason_counts: dict[tuple[str, str], dict[str, int]] = {}
    for _, row in failed.iterrows():
        reason = str(row.get("reason") or "unknown")
        tfbs_list = _ensure_list(row.get("library_tfbs"))
        tf_list = _ensure_list(row.get("library_tfs"))
        for idx, tfbs in enumerate(tfbs_list):
            tf = str(tf_list[idx]) if idx < len(tf_list) else ""
            if not tf and not tfbs:
                continue
            key = (tf, str(tfbs))
            counts[key] = counts.get(key, 0) + 1
            reasons = reason_counts.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top))]
    summary: list[dict] = []
    for (tf, tfbs), count in ordered:
        reasons = reason_counts.get((tf, tfbs), {})
        top_reason = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else ""
        summary.append(
            {
                "tf": tf,
                "tfbs": tfbs,
                "failures": int(count),
                "top_reason": top_reason,
            }
        )
    return summary


def _summarize_top_tfs(tf_counts: dict[str, int], *, top: int = 5) -> list[dict]:
    if not tf_counts:
        return []
    items = sorted(tf_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top))]
    return [{"tf": tf, "count": int(count)} for tf, count in items]


def _summarize_top_tfbs(used_df: pd.DataFrame, *, top: int = 5) -> list[dict]:
    if used_df.empty or "tf" not in used_df.columns or "tfbs" not in used_df.columns:
        return []
    counts = used_df.groupby(["tf", "tfbs"]).size().reset_index(name="count")
    if counts.empty:
        return []
    counts = counts.sort_values("count", ascending=False).head(max(1, int(top)))
    return [
        {"tf": str(row["tf"]), "tfbs": str(row["tfbs"]), "count": int(row["count"])} for _, row in counts.iterrows()
    ]


def _compute_cooccurrence(used_df: pd.DataFrame) -> pd.DataFrame:
    if used_df.empty:
        return pd.DataFrame(columns=["library_hash", "plan", "tf_left", "tf_right", "count"])
    rows = []
    grouped = used_df.groupby(["library_hash", "plan", "sequence_id"])
    for (library_hash, plan, seq_id), group in grouped:
        tfs = sorted({tf for tf in group["tf"].tolist() if tf})
        for i in range(len(tfs)):
            for j in range(i + 1, len(tfs)):
                rows.append(
                    {
                        "library_hash": library_hash,
                        "plan": plan,
                        "sequence_id": seq_id,
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
    for (library_hash, plan, seq_id), group in used_df.groupby(["library_hash", "plan", "sequence_id"]):
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
                    "sequence_id": seq_id,
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
) -> ReportBundle:
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    outputs_root = run_outputs_root(run_root)
    cols = [
        "id",
        "sequence",
        _dg("plan"),
        _dg("input_name"),
        _dg("sampling_library_hash"),
        _dg("sampling_library_index"),
        _dg("used_tfbs_detail"),
        _dg("required_regulators"),
    ]
    df, source_label = load_records_from_config(root_cfg, cfg_path, columns=cols)
    if df.empty:
        raise ValueError("No output records found; cannot build report.")

    used_df = _explode_used(df)
    attempts_path = outputs_root / "attempts.parquet"
    if not attempts_path.exists():
        raise ValueError(
            "outputs/attempts.parquet is required for report/summarize. "
            "Re-run `dense run -c <config.yaml>` to regenerate attempts."
        )
    attempts_df = pd.read_parquet(attempts_path)
    library_df = _explode_library_from_attempts(attempts_df)

    tables: Dict[str, pd.DataFrame] = {}

    stage_a_bins = pd.DataFrame(columns=["input_name", "tf", "bin_id", "bin_low", "bin_high", "count", "total"])
    pool_dir = outputs_root / "pools"
    if pool_dir.exists():
        try:
            pool_artifact = load_pool_artifact(pool_dir)
            rows: list[dict[str, Any]] = []
            for entry in pool_artifact.inputs.values():
                if entry.pool_mode != POOL_MODE_TFBS:
                    continue
                pool_path = pool_dir / entry.pool_path
                if not pool_path.exists():
                    continue
                df_pool = pd.read_parquet(pool_path)
                if "fimo_bin_id" not in df_pool.columns or "tf" not in df_pool.columns:
                    continue
                total_counts = df_pool.groupby("tf").size().to_dict()
                grouped = df_pool.groupby(["tf", "fimo_bin_id"])
                for (tf, bin_id), group in grouped:
                    bin_low = None
                    bin_high = None
                    if "fimo_bin_low" in group.columns and not group["fimo_bin_low"].empty:
                        bin_low = float(group["fimo_bin_low"].iloc[0])
                    if "fimo_bin_high" in group.columns and not group["fimo_bin_high"].empty:
                        bin_high = float(group["fimo_bin_high"].iloc[0])
                    rows.append(
                        {
                            "input_name": entry.name,
                            "tf": tf,
                            "bin_id": int(bin_id),
                            "bin_low": bin_low,
                            "bin_high": bin_high,
                            "count": int(len(group)),
                            "total": int(total_counts.get(tf, len(group))),
                        }
                    )
            if rows:
                stage_a_bins = pd.DataFrame(rows)
        except Exception:
            log.warning("Failed to load Stage-A pool bins for report.", exc_info=True)

    tables["stage_a_bins"] = stage_a_bins

    library_summary = pd.DataFrame(
        columns=["library_hash", "library_index", "input_name", "plan_name", "size", "total_bp", "outputs"]
    )
    if not library_df.empty:
        library_summary = (
            library_df.groupby(["library_hash", "library_index", "input_name", "plan_name"])
            .agg(
                size=("tfbs", "size"),
                total_bp=("tfbs_length", "sum"),
                unique_tf_count=("tf", pd.Series.nunique),
                unique_tfbs_count=("tfbs", pd.Series.nunique),
            )
            .reset_index()
        )
    outputs_by_lib = pd.DataFrame()
    if _dg("sampling_library_index") in df.columns:
        outputs_by_lib = (
            df.groupby(_dg("sampling_library_index"))
            .size()
            .reset_index(name="outputs")
            .rename(columns={_dg("sampling_library_index"): "library_index"})
        )
    if not library_summary.empty and not outputs_by_lib.empty:
        library_summary = library_summary.merge(outputs_by_lib, on="library_index", how="left")
    elif not library_summary.empty:
        library_summary["outputs"] = 0

    tables["library_summary"] = library_summary

    library_usage = pd.DataFrame(
        columns=[
            "library_hash",
            "library_index",
            "input_name",
            "plan_name",
            "attempts",
            "successes",
            "outputs",
        ]
    )
    if not attempts_df.empty:
        attempts_by_lib = (
            attempts_df.groupby(["sampling_library_hash", "sampling_library_index", "input_name", "plan_name"])
            .agg(
                attempts=("status", "size"),
                successes=("status", lambda x: int((x == "success").sum())),
            )
            .reset_index()
            .rename(
                columns={
                    "sampling_library_hash": "library_hash",
                    "sampling_library_index": "library_index",
                }
            )
        )
        library_usage = attempts_by_lib
        if not outputs_by_lib.empty:
            library_usage = library_usage.merge(outputs_by_lib, on="library_index", how="left")
        if "outputs" not in library_usage.columns:
            library_usage["outputs"] = 0
    tables["library_usage"] = library_usage

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
                used_sequences=("sequence_id", pd.Series.nunique),
            )
            .reset_index()
        )
        used_tfbs = (
            used_df.groupby(["library_hash", "plan", "tf", "tfbs"])
            .agg(used_placements=("tfbs", "size"), used_sequences=("sequence_id", pd.Series.nunique))
            .reset_index()
        )
    else:
        used_tf = pd.DataFrame(columns=["library_hash", "plan", "tf", "used_placements", "used_unique_tfbs"])
        used_tfbs = pd.DataFrame(columns=["library_hash", "plan", "tf", "tfbs", "used_placements", "used_sequences"])

    total_sequences = (
        used_df.groupby("library_hash")["sequence_id"].nunique().reset_index(name="total_sequences")
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
    offered_vs_used_tf["used_placements"] = offered_vs_used_tf["used_placements"].fillna(0).astype(int)
    offered_vs_used_tf["used_unique_tfbs"] = offered_vs_used_tf["used_unique_tfbs"].fillna(0).astype(int)
    offered_vs_used_tf["used_sequences"] = offered_vs_used_tf["used_sequences"].fillna(0).astype(int)
    offered_vs_used_tf["total_sequences"] = offered_vs_used_tf["total_sequences"].fillna(0).astype(int)
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
    offered_vs_used_tfbs["used_placements"] = offered_vs_used_tfbs["used_placements"].fillna(0).astype(int)
    offered_vs_used_tfbs["used_sequences"] = offered_vs_used_tfbs["used_sequences"].fillna(0).astype(int)

    tables["offered_vs_used_tf"] = offered_vs_used_tf
    tables["offered_vs_used_tfbs"] = offered_vs_used_tfbs
    tables["attempts"] = attempts_df

    if include_combinatorics:
        tables["tf_cooccurrence"] = _compute_cooccurrence(used_df)
        tables["tf_adjacency"] = _compute_adjacency(used_df)

    composition_path = outputs_root / "composition.parquet"
    if composition_path.exists():
        try:
            tables["composition"] = pd.read_parquet(composition_path)
        except Exception:
            log.warning("Failed to load composition.parquet for report tables.", exc_info=True)

    library_hashes = df[_dg("sampling_library_hash")].dropna().unique().tolist()
    tf_counts = used_df["tf"].value_counts().to_dict() if not used_df.empty else {}
    tfbs_counts = used_df["tfbs"].value_counts().to_dict() if not used_df.empty else {}
    diversity_entropy_tfbs = _normalized_entropy_from_counts({str(k): int(v) for k, v in tfbs_counts.items()})
    leaderboard_tf = _summarize_top_tfs({str(k): int(v) for k, v in tf_counts.items()}, top=5)
    leaderboard_tfbs = _summarize_top_tfbs(used_df, top=5)
    failure_top_tfbs = _summarize_failure_top_tfbs(attempts_df, top=5)
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
        "run_root": str(run_root),
        "schema_version": root_cfg.densegen.schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_source": source_label,
        "output_rows": int(len(df)),
        "output_unique_sequences": int(df["id"].nunique()) if "id" in df.columns else int(len(df)),
        "libraries_in_outputs": int(len(set(library_hashes))),
        "plans": sorted({str(x) for x in df[_dg("plan")].dropna().tolist()}),
        "diversity_unique_tfs": int(len(tf_counts)),
        "diversity_unique_tfbs": int(len(tfbs_counts)),
        "diversity_entropy_tfbs": diversity_entropy_tfbs,
        "failure_top_tfbs": failure_top_tfbs,
        "leaderboard_latest": {
            "tf": leaderboard_tf,
            "tfbs": leaderboard_tfbs,
            "failed_tfbs": failure_top_tfbs,
            "diversity": {
                "tf_coverage": tf_coverage,
                "tfbs_coverage": tfbs_coverage,
                "tfbs_entropy": diversity_entropy_tfbs,
                "used_tf_count": int(len(tf_counts)),
                "library_tf_count": int(lib_tf_total),
                "used_tfbs_count": int(len(tfbs_counts)),
                "library_tfbs_count": int(lib_tfbs_total),
            },
        },
        "attempts_total": attempts_total,
        "attempts_success": attempts_success,
        "attempts_failed": attempts_failed,
        "attempts_path": str(attempts_path) if attempts_path.exists() else None,
        "outputs_path": str(outputs_root / "dense_arrays.parquet"),
        "effective_config_path": str(outputs_root / "meta" / "effective_config.json")
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


def _plot_available() -> bool:
    try:
        ensure_mpl_cache_dir()
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "densegen"


def _generate_report_plots(bundle: ReportBundle, *, cfg_path: Path, out_dir: Path) -> dict[str, list[str]]:
    if not _plot_available():
        log.info("matplotlib not available; skipping report plots.")
        return {}
    import matplotlib.pyplot as plt

    plots: dict[str, list[str]] = {}
    assets_dir = out_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    run_root = resolve_run_root(cfg_path, bundle.run_report.get("run_root", ""))
    outputs_root = run_outputs_root(run_root)

    # Stage-A p-value histograms per input/TF (FIMO)
    pool_dir = outputs_root / "pools"
    if pool_dir.exists():
        try:
            pool_artifact = load_pool_artifact(pool_dir)
            for entry in pool_artifact.inputs.values():
                if entry.pool_mode != POOL_MODE_TFBS:
                    continue
                pool_path = pool_dir / entry.pool_path
                if not pool_path.exists():
                    continue
                df_pool = pd.read_parquet(pool_path)
                if "fimo_pvalue" not in df_pool.columns or "tf" not in df_pool.columns:
                    continue
                for tf, sub in df_pool.groupby("tf"):
                    if sub.empty:
                        continue
                    pvals = sub["fimo_pvalue"].astype(float).replace(0, np.nan).dropna()
                    if pvals.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(np.log10(pvals), bins=30, color="#4c78a8", edgecolor="white")
                    ax.set_title(f"Stage-A p-value histogram: {entry.name}/{tf}")
                    ax.set_xlabel("log10(p-value)")
                    ax.set_ylabel("count")
                    fname = f"stage_a_pvalue_hist__{_safe_filename(entry.name)}__{_safe_filename(str(tf))}.png"
                    path = assets_dir / fname
                    fig.tight_layout()
                    fig.savefig(path)
                    plt.close(fig)
                    plots.setdefault("stage_a_pvalue_hist", []).append(str(path.relative_to(out_dir)))
        except Exception:
            log.warning("Failed to generate Stage-A p-value histograms.", exc_info=True)

    # Stage-A bin occupancy bar charts (per input)
    stage_a_bins = bundle.tables.get("stage_a_bins")
    if stage_a_bins is not None and not stage_a_bins.empty:
        try:
            for input_name, sub in stage_a_bins.groupby("input_name"):
                fig, ax = plt.subplots(figsize=(6, 4))
                sub = sub.sort_values(["tf", "bin_id"])
                labels = [f"{row['tf']}:{int(row['bin_id'])}" for _, row in sub.iterrows()]
                counts = sub["count"].astype(int).tolist()
                ax.bar(labels, counts, color="#f58518")
                ax.set_title(f"Stage-A bin occupancy: {input_name}")
                ax.set_ylabel("count")
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
                fname = f"stage_a_bin_counts__{_safe_filename(str(input_name))}.png"
                path = assets_dir / fname
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)
                plots.setdefault("stage_a_bin_counts", []).append(str(path.relative_to(out_dir)))
        except Exception:
            log.warning("Failed to generate Stage-A bin occupancy plots.", exc_info=True)

    # Stage-B TF utilization (offered vs used)
    offered_vs_used = bundle.tables.get("offered_vs_used_tf")
    if offered_vs_used is not None and not offered_vs_used.empty:
        try:
            for lib_hash, sub in offered_vs_used.groupby("library_hash"):
                sub = sub.sort_values("tf")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(sub["tf"], sub["used_sequences"], color="#54a24b", label="used sequences")
                ax.set_title(f"Stage-B TF utilization: {lib_hash[:8]}")
                ax.set_ylabel("used sequences")
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
                fname = f"stage_b_tf_util__{_safe_filename(str(lib_hash))}.png"
                path = assets_dir / fname
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)
                plots.setdefault("stage_b_tf_utilization", []).append(str(path.relative_to(out_dir)))
        except Exception:
            log.warning("Failed to generate Stage-B utilization plots.", exc_info=True)

    return plots


def write_report(
    root_cfg: RootConfig,
    cfg_path: Path,
    *,
    out_dir: str | Path = "outputs",
    include_combinatorics: bool = False,
    formats: set[str] | None = None,
) -> ReportBundle:
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    out_path = resolve_run_scoped_path(cfg_path, run_root, str(out_dir), label="report.out")
    out_path.mkdir(parents=True, exist_ok=True)

    bundle = collect_report_data(root_cfg, cfg_path, include_combinatorics=include_combinatorics)
    try:
        plots = _generate_report_plots(bundle, cfg_path=cfg_path, out_dir=out_path)
        bundle.plots = plots
        if plots:
            bundle.run_report["report_plots"] = plots
    except Exception:
        log.debug("Failed to generate report plots.", exc_info=True)
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
        "",
        "## Outputs",
        "- outputs/dense_arrays.parquet",
        "- outputs/attempts.parquet",
        "- outputs/composition.parquet",
        "- outputs/libraries/library_builds.parquet",
        "- outputs/libraries/library_members.parquet",
        "- outputs/pools/pool_manifest.json",
        "- outputs/meta/effective_config.json",
    ]
    stage_a_bins = bundle.tables.get("stage_a_bins")
    if stage_a_bins is not None and not stage_a_bins.empty:
        lines.extend(["", "## Stage-A p-value bins"])
        for (input_name, tf), sub in stage_a_bins.groupby(["input_name", "tf"]):
            sub = sub.sort_values("bin_id")
            parts = []
            for _, row in sub.iterrows():
                bin_id = int(row.get("bin_id") or 0)
                count = int(row.get("count") or 0)
                low = row.get("bin_low")
                high = row.get("bin_high")
                if low is not None and high is not None:
                    label = f"({float(low):.0e},{float(high):.0e}]"
                else:
                    label = f"bin{bin_id}"
                parts.append(f"{label}:{count}")
            lines.append(f"- {input_name}/{tf}: " + " ".join(parts))
    library_usage = bundle.tables.get("library_usage")
    if library_usage is not None and not library_usage.empty:
        lines.extend(["", "## Library usage (top 5)"])
        top_usage = library_usage.sort_values(["attempts", "outputs"], ascending=False).head(5)
        for _, row in top_usage.iterrows():
            lib_hash = str(row.get("library_hash") or "")[:8]
            attempts = int(row.get("attempts") or 0)
            outputs = int(row.get("outputs") or 0)
            plan_name = str(row.get("plan_name") or "")
            lines.append(f"- {plan_name}/{lib_hash}: attempts={attempts} outputs={outputs}")
    leaderboard = report.get("leaderboard_latest") or {}
    leader_tf = leaderboard.get("tf") or []
    leader_tfbs = leaderboard.get("tfbs") or []
    if leader_tf or leader_tfbs:
        lines.extend(["", "## Leaderboards"])
        if leader_tf:
            lines.append("")
            lines.append("Top TFs:")
            for row in leader_tf:
                lines.append(f"- {row.get('tf')}: {row.get('count')}")
        if leader_tfbs:
            lines.append("")
            lines.append("Top TFBS:")
            for row in leader_tfbs:
                lines.append(f"- {row.get('tf')}:{row.get('tfbs')} ({row.get('count')})")
    failure_top = report.get("failure_top_tfbs") or []
    if failure_top:
        lines.extend(["", "## Failure hotspots (top TFBS)"])
        for entry in failure_top:
            tf = entry.get("tf") or ""
            tfbs = entry.get("tfbs") or ""
            failures = entry.get("failures") or 0
            reason = entry.get("top_reason") or ""
            label = f"{tf}:{tfbs}" if tf else tfbs
            reason_suffix = f" (top reason: {reason})" if reason else ""
            lines.append(f"- {label} — failures={failures}{reason_suffix}")
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
