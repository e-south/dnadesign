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
from dnadesign.cruncher.analysis.hits import (
    load_baseline_hits,
    load_baseline_occurrences,
    load_elite_occurrences,
    load_elites_hits,
    representative_hit_contract_enabled,
)
from dnadesign.cruncher.analysis.objective_labels import objective_scale_label
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.app.sample.preflight import _core_def_hash
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_yaml
from dnadesign.cruncher.artifacts.layout import (
    elites_hits_path,
    elites_occurrences_path,
    elites_path,
    elites_yaml_path,
    random_baseline_hits_path,
    random_baseline_occurrences_path,
    random_baseline_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.sequence import identity_key

_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


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


def _revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _representative_hit_contract(manifest: dict[str, object]) -> bool:
    objective_payload = manifest.get("objective")
    if not isinstance(objective_payload, dict):
        return True
    return representative_hit_contract_enabled(objective_payload)


def _motif_provenance_by_tf(
    *,
    manifest: dict[str, object],
    pwms: dict[str, PWM],
) -> tuple[dict[str, str | None], dict[str, str | None], dict[str, str], dict[str, int]]:
    pwm_ref_by_tf: dict[str, str | None] = {}
    pwm_hash_by_tf: dict[str, str | None] = {}
    motifs_payload = manifest.get("motifs")
    if isinstance(motifs_payload, list):
        for item in motifs_payload:
            if not isinstance(item, dict):
                continue
            tf_name = str(item.get("tf_name") or "").strip()
            if not tf_name:
                continue
            source = str(item.get("source") or "").strip()
            motif_id = str(item.get("motif_id") or "").strip()
            pwm_ref_by_tf[tf_name] = f"{source}:{motif_id}" if source and motif_id else None
            sha256 = item.get("sha256")
            pwm_hash_by_tf[tf_name] = str(sha256) if isinstance(sha256, str) and sha256.strip() else None
    core_def_by_tf: dict[str, str] = {}
    pwm_width_by_tf: dict[str, int] = {}
    for tf_name, pwm in pwms.items():
        pwm_width_by_tf[tf_name] = int(pwm.length)
        core_def_by_tf[tf_name] = _core_def_hash(pwm, pwm_hash_by_tf.get(tf_name))
    return pwm_ref_by_tf, pwm_hash_by_tf, core_def_by_tf, pwm_width_by_tf


def _sequence_by_id(frame: pd.DataFrame, *, id_column: str) -> dict[object, str]:
    if frame is None or frame.empty:
        return {}
    if id_column not in frame.columns or "sequence" not in frame.columns:
        raise ValueError(f"Cannot build sequence map without columns '{id_column}' and 'sequence'.")
    mapping: dict[object, str] = {}
    for item_id, sequence in frame[[id_column, "sequence"]].itertuples(index=False, name=None):
        if item_id is None:
            continue
        mapping[item_id] = str(sequence).upper()
    return mapping


def _slot_label(tf_name: str, occurrence_rank: int, *, duplicate_tfs: set[str]) -> str:
    if tf_name not in duplicate_tfs:
        return tf_name
    return f"{tf_name}#{occurrence_rank}"


def _normalize_occurrence_hits(
    *,
    occurrences_df: pd.DataFrame,
    sequence_map: dict[object, str],
    id_column: str,
    pwms: dict[str, PWM],
    manifest: dict[str, object],
) -> pd.DataFrame:
    if occurrences_df is None or occurrences_df.empty:
        columns = [
            id_column,
            "tf",
            "tf_slot",
            "occurrence_rank",
            "best_start",
            "best_core_offset",
            "best_strand",
            "best_window_seq",
            "best_core_seq",
            "best_score_raw",
            "best_score_scaled",
            "best_score_norm",
            "tiebreak_rule",
            "pwm_ref",
            "pwm_hash",
            "pwm_width",
            "core_width",
            "core_def_hash",
        ]
        return pd.DataFrame(columns=columns)

    pwm_ref_by_tf, pwm_hash_by_tf, core_def_by_tf, pwm_width_by_tf = _motif_provenance_by_tf(
        manifest=manifest,
        pwms=pwms,
    )
    selected_df = occurrences_df.copy()
    if "selected" in selected_df.columns:
        selected_df = selected_df[selected_df["selected"].fillna(False)].copy()
    if selected_df.empty:
        return pd.DataFrame()
    occurrence_counts = (
        selected_df.groupby("tf")["occurrence_rank"].nunique().to_dict()
        if "occurrence_rank" in selected_df.columns
        else {}
    )
    duplicate_tfs = {str(tf_name) for tf_name, count in occurrence_counts.items() if int(count) > 1}
    rows: list[dict[str, object]] = []
    for row in selected_df.to_dict(orient="records"):
        item_id = row.get(id_column)
        if item_id not in sequence_map:
            raise ValueError(f"Missing sequence for {id_column}={item_id!r} while normalizing occurrences.")
        tf_name = str(row.get("tf") or "").strip()
        if not tf_name:
            raise ValueError(f"Occurrence row missing TF for {id_column}={item_id!r}.")
        pwm = pwms.get(tf_name)
        if pwm is None:
            raise ValueError(f"Missing PWM for TF '{tf_name}' while normalizing occurrences.")
        occurrence_rank = int(row.get("occurrence_rank") or 0)
        if occurrence_rank < 1:
            raise ValueError(f"Occurrence row has invalid occurrence_rank for TF '{tf_name}': {occurrence_rank}")
        start = int(row.get("start"))
        end = int(row.get("end"))
        if end <= start:
            raise ValueError(f"Occurrence row has invalid interval for TF '{tf_name}': [{start}, {end})")
        sequence = sequence_map[item_id]
        if start < 0 or end > len(sequence):
            raise ValueError(
                f"Occurrence row span out of bounds for {id_column}={item_id!r}, TF '{tf_name}': "
                f"[{start}, {end}) with sequence length {len(sequence)}"
            )
        window_seq = sequence[start:end]
        strand = str(row.get("strand") or "").strip()
        if strand not in {"+", "-"}:
            raise ValueError(f"Occurrence row has invalid strand for TF '{tf_name}': {strand!r}")
        oriented_core = window_seq if strand == "+" else _revcomp(window_seq)
        pwm_width = pwm_width_by_tf.get(tf_name, int(end - start))
        rows.append(
            {
                id_column: item_id,
                "tf": tf_name,
                "tf_slot": _slot_label(tf_name, occurrence_rank, duplicate_tfs=duplicate_tfs),
                "occurrence_rank": occurrence_rank,
                "best_start": start,
                "best_core_offset": start,
                "best_strand": strand,
                "best_window_seq": window_seq,
                "best_core_seq": oriented_core,
                "best_score_raw": float(row.get("raw_score")),
                "best_score_scaled": float(row.get("scaled_score")),
                "best_score_norm": float(row.get("normalized_score")),
                "tiebreak_rule": "occurrence_rank",
                "pwm_ref": pwm_ref_by_tf.get(tf_name),
                "pwm_hash": pwm_hash_by_tf.get(tf_name),
                "pwm_width": pwm_width,
                "core_width": int(end - start),
                "core_def_hash": core_def_by_tf.get(tf_name),
            }
        )
    normalized = pd.DataFrame(rows)
    sort_columns = [id_column, "tf", "occurrence_rank", "best_start"]
    return normalized.sort_values(sort_columns).reset_index(drop=True)


def _load_run_artifacts_for_analysis(
    run_dir: Path,
    *,
    require_random_baseline: bool,
    pwms: dict[str, PWM],
    manifest: dict[str, object],
) -> _AnalyzeRunArtifacts:
    sequences_file = sequences_path(run_dir)
    elites_file = elites_path(run_dir)
    hits_file = elites_hits_path(run_dir)
    elite_occurrences_file = elites_occurrences_path(run_dir)
    baseline_file = random_baseline_path(run_dir)
    baseline_hits_file = random_baseline_hits_path(run_dir)
    baseline_occurrences_file = random_baseline_occurrences_path(run_dir)
    if not sequences_file.exists():
        raise FileNotFoundError(f"Missing sequences parquet: {sequences_file}")
    if not elites_file.exists():
        raise FileNotFoundError(f"Missing elites parquet: {elites_file}")
    sequences_df = read_parquet(sequences_file)
    elites_df = read_parquet(elites_file)
    representative_hits = _representative_hit_contract(manifest)
    if representative_hits:
        if not hits_file.exists():
            raise FileNotFoundError(f"Missing elites hits parquet: {hits_file}")
        hits_df = load_elites_hits(hits_file)
    else:
        if not elite_occurrences_file.exists():
            raise FileNotFoundError(f"Missing elites occurrences parquet: {elite_occurrences_file}")
        hits_df = _normalize_occurrence_hits(
            occurrences_df=load_elite_occurrences(elite_occurrences_file),
            sequence_map=_sequence_by_id(elites_df, id_column="id"),
            id_column="elite_id",
            pwms=pwms,
            manifest=manifest,
        )
    if representative_hits and baseline_file.exists() and baseline_hits_file.exists():
        baseline_df = read_parquet(baseline_file)
        baseline_hits_df = load_baseline_hits(baseline_hits_file)
    elif (not representative_hits) and baseline_file.exists() and baseline_occurrences_file.exists():
        baseline_df = read_parquet(baseline_file)
        baseline_hits_df = _normalize_occurrence_hits(
            occurrences_df=load_baseline_occurrences(baseline_occurrences_file),
            sequence_map=_sequence_by_id(baseline_df, id_column="baseline_id"),
            id_column="baseline_id",
            pwms=pwms,
            manifest=manifest,
        )
    else:
        if require_random_baseline:
            if not baseline_file.exists():
                raise FileNotFoundError(f"Missing random baseline parquet: {baseline_file}")
            missing_path = baseline_hits_file if representative_hits else baseline_occurrences_file
            label = "hits" if representative_hits else "occurrences"
            raise FileNotFoundError(f"Missing random baseline {label} parquet: {missing_path}")
        expected_companion = baseline_hits_file if representative_hits else baseline_occurrences_file
        if baseline_file.exists() != expected_companion.exists():
            if baseline_file.exists():
                raise FileNotFoundError(
                    f"Missing random baseline {'hits' if representative_hits else 'occurrences'} parquet: "
                    f"{expected_companion}. "
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
    scale_label = objective_scale_label(objective_cfg, unknown_fallback="norm-LLR")
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
