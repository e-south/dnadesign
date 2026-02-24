"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze_support.py

Shared helpers for analysis artifact loading, persistence, and elite summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from dnadesign.cruncher.analysis.diversity import (
    compute_elite_distance_matrix,
    compute_elites_full_sequence_nn_table,
    compute_elites_nn_distance_table,
    representative_elite_ids,
    summarize_elite_distances,
)
from dnadesign.cruncher.analysis.hits import load_baseline_hits, load_elites_hits
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_yaml
from dnadesign.cruncher.artifacts.layout import (
    elites_hits_path,
    elites_path,
    elites_yaml_path,
    random_baseline_hits_path,
    random_baseline_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.core.sequence import identity_key


@dataclass(frozen=True)
class _AnalyzeRunArtifacts:
    sequences_df: pd.DataFrame
    elites_df: pd.DataFrame
    hits_df: pd.DataFrame
    baseline_df: pd.DataFrame
    baseline_hits_df: pd.DataFrame
    trace_idata: object | None
    elites_meta: dict[str, object]


def _load_elites_meta(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing elites metadata YAML: {path}")
    try:
        payload = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid elites metadata YAML at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Elites metadata must contain a YAML mapping: {path}")
    return payload


def _load_run_artifacts_for_analysis(run_dir: Path, *, require_random_baseline: bool) -> _AnalyzeRunArtifacts:
    sequences_file = sequences_path(run_dir)
    elites_file = elites_path(run_dir)
    hits_file = elites_hits_path(run_dir)
    baseline_file = random_baseline_path(run_dir)
    baseline_hits_file = random_baseline_hits_path(run_dir)
    if not sequences_file.exists():
        raise FileNotFoundError(f"Missing sequences parquet: {sequences_file}")
    if not elites_file.exists():
        raise FileNotFoundError(f"Missing elites parquet: {elites_file}")
    if not hits_file.exists():
        raise FileNotFoundError(f"Missing elites hits parquet: {hits_file}")
    sequences_df = read_parquet(sequences_file)
    elites_df = read_parquet(elites_file)
    hits_df = load_elites_hits(hits_file)
    if baseline_file.exists() and baseline_hits_file.exists():
        baseline_df = read_parquet(baseline_file)
        baseline_hits_df = load_baseline_hits(baseline_hits_file)
    else:
        if require_random_baseline:
            if not baseline_file.exists():
                raise FileNotFoundError(f"Missing random baseline parquet: {baseline_file}")
            raise FileNotFoundError(f"Missing random baseline hits parquet: {baseline_hits_file}")
        if baseline_file.exists() != baseline_hits_file.exists():
            if baseline_file.exists():
                raise FileNotFoundError(
                    f"Missing random baseline hits parquet: {baseline_hits_file}. "
                    "Baseline artifacts must be written together."
                )
            raise FileNotFoundError(
                f"Missing random baseline parquet: {baseline_file}. Baseline artifacts must be written together."
            )
        baseline_df = pd.DataFrame()
        baseline_hits_df = pd.DataFrame()

    trace_file = trace_path(run_dir)
    trace_idata = None
    if trace_file.exists():
        import arviz as az

        trace_idata = az.from_netcdf(trace_file)

    elites_meta = _load_elites_meta(elites_yaml_path(run_dir))
    return _AnalyzeRunArtifacts(
        sequences_df=sequences_df,
        elites_df=elites_df,
        hits_df=hits_df,
        baseline_df=baseline_df,
        baseline_hits_df=baseline_hits_df,
        trace_idata=trace_idata,
        elites_meta=elites_meta,
    )


def _resolve_tf_names(used_cfg: dict, pwms: dict[str, object]) -> list[str]:
    active = used_cfg.get("active_regulator_set") if isinstance(used_cfg, dict) else None
    if isinstance(active, dict):
        tfs = active.get("tfs")
        if isinstance(tfs, list) and tfs:
            return [str(tf) for tf in tfs if str(tf)]
    return sorted(pwms.keys())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        write_parquet(df, path)
    else:
        df.to_csv(path, index=False)


def _write_analysis_used(
    path: Path,
    analysis_cfg: dict[str, object],
    analysis_id: str,
    run_name: str,
    *,
    extras: dict[str, object] | None = None,
) -> None:
    payload = {
        "analysis": analysis_cfg,
        "analysis_id": analysis_id,
        "run": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(path, payload, sort_keys=False, default_flow_style=False)


def _objective_axis_label(objective_cfg: dict[str, object]) -> str:
    score_scale = str(objective_cfg.get("score_scale") or "normalized-llr").strip().lower()
    if score_scale in {"llr", "raw-llr", "raw_llr"}:
        scale_label = "raw-LLR"
    else:
        scale_label = "norm-LLR"
    combine = str(objective_cfg.get("combine") or "min").strip().lower()
    softmin_cfg = objective_cfg.get("softmin")
    softmin_enabled = isinstance(softmin_cfg, dict) and bool(softmin_cfg.get("enabled"))
    if combine == "sum":
        return f"Cruncher sum-TF best-window {scale_label}"
    if softmin_enabled:
        return f"Cruncher soft-min TF best-window {scale_label}"
    return f"Cruncher min-TF best-window {scale_label}"


def _resolve_baseline_seed(baseline_df: pd.DataFrame) -> int | None:
    if "baseline_seed" not in baseline_df.columns or baseline_df.empty:
        return None
    raw_seed = baseline_df["baseline_seed"].iloc[0]
    if pd.isna(raw_seed):
        raise ValueError("random baseline metadata baseline_seed is missing in random_baseline.parquet")
    try:
        return int(raw_seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"random baseline metadata baseline_seed is not an integer: {raw_seed!r}") from exc


def _resolve_identity_maps(
    *,
    elites_df: pd.DataFrame,
    bidirectional: bool,
) -> tuple[dict[str, str], dict[str, int]]:
    identity_by_elite_id: dict[str, str] = {}
    rank_by_elite_id: dict[str, int] = {}
    if elites_df is None or elites_df.empty or "id" not in elites_df.columns:
        return identity_by_elite_id, rank_by_elite_id
    if bidirectional and "canonical_sequence" in elites_df.columns:
        seq_series = elites_df["canonical_sequence"].astype(str).str.strip().str.upper()
        identity_by_elite_id = {str(elite_id): seq for elite_id, seq in zip(elites_df["id"].astype(str), seq_series)}
    else:
        seq_series = elites_df["sequence"].astype(str)
        identity_by_elite_id = {
            str(elite_id): identity_key(seq, bidirectional=bidirectional)
            for elite_id, seq in zip(elites_df["id"].astype(str), seq_series)
        }
    if "rank" in elites_df.columns:
        rank_by_elite_id = {
            str(elite_id): int(rank)
            for elite_id, rank in zip(elites_df["id"].astype(str), elites_df["rank"])
            if rank is not None
        }
    else:
        rank_by_elite_id = {str(elite_id): idx for idx, elite_id in enumerate(elites_df["id"].astype(str))}
    return identity_by_elite_id, rank_by_elite_id


def _resolve_median_relevance_raw(
    *,
    elites_df: pd.DataFrame,
    elites_meta: dict[str, object],
) -> float | None:
    min_norm = pd.to_numeric(elites_df.get("min_norm"), errors="coerce") if "min_norm" in elites_df.columns else None
    median_relevance_raw = float(min_norm.median()) if min_norm is not None and not min_norm.empty else None
    mmr_summary_meta = elites_meta.get("mmr_summary")
    if isinstance(mmr_summary_meta, dict):
        mmr_relevance_meta = mmr_summary_meta.get("median_relevance_raw")
        if mmr_relevance_meta is not None:
            try:
                median_relevance_raw = float(mmr_relevance_meta)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("elites_mmr_meta.mmr_summary.median_relevance_raw must be numeric.") from exc
    return median_relevance_raw


def _resolve_unique_fraction(
    *,
    df: pd.DataFrame,
    bidirectional: bool,
) -> float | None:
    if "sequence" not in df.columns or len(df) == 0:
        return None
    if bidirectional and "canonical_sequence" in df.columns:
        unique = int(df["canonical_sequence"].astype(str).str.strip().str.upper().nunique())
    else:
        source = df["sequence"].astype(str)
        unique = int(source.map(lambda seq: identity_key(seq, bidirectional=bidirectional)).nunique())
    return unique / float(len(df))


def _summarize_elites_mmr(
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    elites_meta: dict[str, object],
    tf_names: list[str],
    pwms: dict[str, object],
    *,
    bidirectional: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity_mode = "canonical" if bidirectional else "raw"
    identity_by_elite_id, rank_by_elite_id = _resolve_identity_maps(
        elites_df=elites_df,
        bidirectional=bidirectional,
    )

    nn_df = compute_elites_nn_distance_table(
        hits_df,
        tf_names,
        pwms,
        identity_mode=identity_mode,
        identity_by_elite_id=identity_by_elite_id or None,
        rank_by_elite_id=rank_by_elite_id or None,
    )
    full_nn_df, full_summary = compute_elites_full_sequence_nn_table(
        elites_df,
        identity_mode=identity_mode,
        identity_by_elite_id=identity_by_elite_id or None,
        rank_by_elite_id=rank_by_elite_id or None,
    )
    if not full_nn_df.empty:
        nn_df = nn_df.merge(
            full_nn_df,
            on=["elite_id", "identity_mode"],
            how="left",
        )
    rep_by_identity = representative_elite_ids(identity_by_elite_id, rank_by_elite_id) if identity_by_elite_id else {}
    hits_for_dist = hits_df
    if rep_by_identity:
        keep_ids = set(rep_by_identity.values())
        hits_for_dist = hits_df[hits_df["elite_id"].isin(keep_ids)].copy()
    _, dist = compute_elite_distance_matrix(hits_for_dist, tf_names, pwms)
    dist_summary = summarize_elite_distances(dist)
    median_relevance_raw = _resolve_median_relevance_raw(
        elites_df=elites_df,
        elites_meta=elites_meta,
    )
    draw_df = sequences_df[sequences_df.get("phase") == "draw"] if "phase" in sequences_df.columns else sequences_df
    unique_draw_fraction = _resolve_unique_fraction(df=draw_df, bidirectional=bidirectional)
    unique_elites_fraction = _resolve_unique_fraction(df=elites_df, bidirectional=bidirectional)
    summary = {
        "k": int(elites_meta.get("n_elites") or len(elites_df)),
        "score_weight": elites_meta.get("selection_score_weight"),
        "diversity_weight": elites_meta.get("selection_diversity_weight"),
        "pool_size": elites_meta.get("pool_size"),
        "relevance": elites_meta.get("selection_relevance"),
        "median_relevance_raw": median_relevance_raw,
        "mean_pairwise_distance": dist_summary.get("mean_pairwise_distance"),
        "min_pairwise_distance": dist_summary.get("min_pairwise_distance"),
        "sequence_length_bp": full_summary.get("sequence_length_bp"),
        "mean_pairwise_full_bp": full_summary.get("mean_pairwise_full_bp"),
        "min_pairwise_full_bp": full_summary.get("min_pairwise_full_bp"),
        "median_nn_full_bp": full_summary.get("median_nn_full_bp"),
        "mean_pairwise_full_distance": full_summary.get("mean_pairwise_full_distance"),
        "min_pairwise_full_distance": full_summary.get("min_pairwise_full_distance"),
        "median_nn_full_distance": full_summary.get("median_nn_full_distance"),
        "unique_fraction_canonical_draws": unique_draw_fraction,
        "unique_fraction_canonical_elites": unique_elites_fraction,
    }
    return pd.DataFrame([summary]), nn_df
