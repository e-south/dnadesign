"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze_workflow.py

Analyze sampling runs and produce summary reports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from dnadesign.cruncher.analysis.consensus import compute_consensus_anchors
from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.analysis.diversity import (
    compute_baseline_nn_distances,
    compute_elite_distance_matrix,
    compute_elites_nn_distance_table,
    representative_elite_ids,
    summarize_elite_distances,
)
from dnadesign.cruncher.analysis.hits import load_baseline_hits, load_elites_hits
from dnadesign.cruncher.analysis.layout import (
    analysis_manifest_path,
    analysis_plot_path,
    analysis_root,
    analysis_table_path,
    analysis_used_path,
    plot_manifest_path,
    report_json_path,
    report_md_path,
    summary_path,
    table_manifest_path,
)
from dnadesign.cruncher.analysis.objective import compute_objective_components
from dnadesign.cruncher.analysis.overlap import compute_overlap_tables
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.analysis.plot_registry import PLOT_SPECS
from dnadesign.cruncher.analysis.report import build_report_payload, write_report_json, write_report_md
from dnadesign.cruncher.analysis.trajectory import (
    add_raw_llr_objective,
    build_particle_trajectory_points,
    build_trajectory_points,
)
from dnadesign.cruncher.app.analyze.archive import _prune_latest_analysis_artifacts
from dnadesign.cruncher.app.analyze.manifests import build_analysis_manifests
from dnadesign.cruncher.app.analyze.metadata import (
    _analysis_id,
    _get_version,
    _load_pwms_from_config,
    _resolve_sample_meta,
)
from dnadesign.cruncher.app.analyze.plan import resolve_analysis_plan
from dnadesign.cruncher.app.run_service import get_run, list_runs, load_run_status
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_yaml
from dnadesign.cruncher.artifacts.entries import append_artifacts, artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    elites_hits_path,
    elites_path,
    elites_yaml_path,
    manifest_path,
    random_baseline_hits_path,
    random_baseline_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest, write_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.core.sequence import identity_key
from dnadesign.cruncher.utils.arviz_cache import ensure_arviz_data_dir
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)

__all__ = ["run_analyze"]


def _status_text(status: object) -> str:
    return str(status or "").strip().lower()


def _run_status_detail(run_dir: Path) -> str | None:
    try:
        payload = load_run_status(run_dir)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    status_message = payload.get("status_message")
    if isinstance(status_message, str) and status_message.strip():
        return status_message.strip()
    return None


def _is_analyzable_sample_run(run: object) -> bool:
    status = _status_text(getattr(run, "status", None))
    if status in {"failed", "aborted", "running"}:
        return False
    run_dir = getattr(run, "run_dir", None)
    if not isinstance(run_dir, Path):
        return False
    return manifest_path(run_dir).exists()


def _latest_unavailable_reason(run: object) -> str:
    run_name = str(getattr(run, "name", "<unknown>"))
    status = _status_text(getattr(run, "status", None)) or "unknown"
    run_dir = getattr(run, "run_dir", None)
    has_manifest = isinstance(run_dir, Path) and manifest_path(run_dir).exists()
    detail = _run_status_detail(run_dir) if isinstance(run_dir, Path) else None
    if not has_manifest:
        if status in {"failed", "aborted"} and detail:
            return f"run '{run_name}' {status}: {detail}"
        return f"run '{run_name}' status={status} is missing run_manifest.json"
    if status in {"failed", "aborted"}:
        if detail:
            return f"run '{run_name}' {status}: {detail}"
        return f"run '{run_name}' status={status}"
    if status == "running":
        return f"run '{run_name}' is still running"
    return f"run '{run_name}' is not ready for analysis (status={status})"


def _resolve_run_names(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    runs_override: list[str] | None,
    use_latest: bool,
) -> list[str]:
    analysis_cfg = cfg.analysis
    if analysis_cfg is None:
        raise ValueError("analysis section is required for analyze")
    if runs_override:
        return runs_override
    if analysis_cfg.run_selector == "explicit":
        if not analysis_cfg.runs:
            raise ValueError("analysis.run_selector=explicit requires analysis.runs to be non-empty")
        return list(analysis_cfg.runs)
    if use_latest or analysis_cfg.run_selector == "latest":
        runs = list_runs(cfg, config_path, stage="sample")
        if not runs:
            raise ValueError("No sample runs found for analysis.")
        for run in runs:
            if _is_analyzable_sample_run(run):
                return [run.name]
        latest_reason = _latest_unavailable_reason(runs[0])
        raise ValueError(
            "No completed sample runs found for analysis. "
            f"Latest sample run unavailable: {latest_reason}. "
            "Re-run sampling with `cruncher sample -c <CONFIG>`."
        )
    raise ValueError("analysis.run_selector must be 'latest' or 'explicit'")


def _resolve_run_dir(cfg: CruncherConfig, config_path: Path, run_name: str) -> Path:
    run = get_run(cfg, config_path, run_name)
    if run.stage != "sample":
        raise ValueError(f"Run '{run_name}' is not a sample run (stage={run.stage}).")
    return run.run_dir


def _resolve_optimizer_stats(manifest: dict[str, object], run_dir: Path) -> dict[str, object] | None:
    stats_payload = manifest.get("optimizer_stats")
    if stats_payload is None:
        return None
    if not isinstance(stats_payload, dict):
        raise ValueError("Run manifest field 'optimizer_stats' must be a dictionary.")
    optimizer_stats = dict(stats_payload)
    move_stats_path = optimizer_stats.get("move_stats_path") or manifest.get("optimizer_stats_path")
    if move_stats_path is None:
        return optimizer_stats
    if not isinstance(move_stats_path, str) or not move_stats_path:
        raise ValueError("Run manifest field 'optimizer_stats_path' must be a non-empty string when provided.")
    relative = Path(move_stats_path)
    if relative.is_absolute():
        raise ValueError("Run manifest field 'optimizer_stats_path' must be relative to the run directory.")
    sidecar = run_dir / relative
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing optimizer stats sidecar: {sidecar}")
    if sidecar.suffix == ".gz":
        with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(sidecar.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Optimizer stats sidecar must contain an object: {sidecar}")
    move_stats = payload.get("move_stats")
    if move_stats is not None and not isinstance(move_stats, list):
        raise ValueError(f"Optimizer stats sidecar field 'move_stats' must be a list when present: {sidecar}")
    swap_events = payload.get("swap_events")
    if swap_events is not None and not isinstance(swap_events, list):
        raise ValueError(f"Optimizer stats sidecar field 'swap_events' must be a list when present: {sidecar}")
    if move_stats is None and swap_events is None:
        raise ValueError(f"Optimizer stats sidecar missing both 'move_stats' and 'swap_events' lists: {sidecar}")
    if move_stats is not None:
        optimizer_stats["move_stats"] = move_stats
    if swap_events is not None:
        optimizer_stats["swap_events"] = swap_events
    return optimizer_stats


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


def _resolve_tf_names(used_cfg: dict, pwms: dict[str, object]) -> list[str]:
    active = used_cfg.get("active_regulator_set") if isinstance(used_cfg, dict) else None
    if isinstance(active, dict):
        tfs = active.get("tfs")
        if isinstance(tfs, list) and tfs:
            return [str(tf) for tf in tfs if str(tf)]
    return sorted(pwms.keys())


def _select_tf_pair(tf_names: list[str], pairwise: object) -> tuple[str, str] | None:
    if pairwise == "off":
        return None
    if isinstance(pairwise, list) and len(pairwise) == 2:
        a, b = str(pairwise[0]), str(pairwise[1])
        if a not in tf_names or b not in tf_names:
            raise ValueError("analysis.pairwise TFs must be present in the run.")
        return (a, b)
    if len(tf_names) >= 2:
        return (tf_names[0], tf_names[1])
    return None


def _resolve_trajectory_tf_pair(tf_names: list[str], pairwise: object) -> tuple[str, str]:
    selected = _select_tf_pair(tf_names, pairwise)
    if selected is not None:
        return selected
    if len(tf_names) == 1:
        tf_name = str(tf_names[0])
        return (tf_name, tf_name)
    raise ValueError("Trajectory scatter plot requires at least one TF.")


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
    identity_by_elite_id: dict[str, str] = {}
    rank_by_elite_id: dict[str, int] = {}
    if elites_df is not None and not elites_df.empty and "id" in elites_df.columns:
        seq_series = elites_df["sequence"].astype(str)
        if bidirectional and "canonical_sequence" in elites_df.columns:
            seq_series = elites_df["canonical_sequence"].astype(str)
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

    nn_df = compute_elites_nn_distance_table(
        hits_df,
        tf_names,
        pwms,
        identity_mode=identity_mode,
        identity_by_elite_id=identity_by_elite_id or None,
        rank_by_elite_id=rank_by_elite_id or None,
    )
    rep_by_identity = representative_elite_ids(identity_by_elite_id, rank_by_elite_id) if identity_by_elite_id else {}
    hits_for_dist = hits_df
    if rep_by_identity:
        keep_ids = set(rep_by_identity.values())
        hits_for_dist = hits_df[hits_df["elite_id"].isin(keep_ids)].copy()
    elite_ids, dist = compute_elite_distance_matrix(hits_for_dist, tf_names, pwms)
    dist_summary = summarize_elite_distances(dist)
    min_norm = pd.to_numeric(elites_df.get("min_norm"), errors="coerce") if "min_norm" in elites_df.columns else None
    median_relevance_raw = float(min_norm.median()) if min_norm is not None and not min_norm.empty else None
    draw_df = sequences_df[sequences_df.get("phase") == "draw"] if "phase" in sequences_df.columns else sequences_df
    unique_draws = None
    if "sequence" in draw_df.columns:
        source = draw_df["sequence"].astype(str)
        if bidirectional and "canonical_sequence" in draw_df.columns:
            source = draw_df["canonical_sequence"].astype(str)
        unique_draws = source.map(lambda seq: identity_key(seq, bidirectional=bidirectional)).nunique()
    unique_elites = None
    if "sequence" in elites_df.columns:
        source = elites_df["sequence"].astype(str)
        if bidirectional and "canonical_sequence" in elites_df.columns:
            source = elites_df["canonical_sequence"].astype(str)
        unique_elites = source.map(lambda seq: identity_key(seq, bidirectional=bidirectional)).nunique()
    summary = {
        "k": int(elites_meta.get("n_elites") or len(elites_df)),
        "alpha": elites_meta.get("mmr_alpha"),
        "pool_size": elites_meta.get("pool_size"),
        "median_relevance_raw": median_relevance_raw,
        "mean_pairwise_distance": dist_summary.get("mean_pairwise_distance"),
        "min_pairwise_distance": dist_summary.get("min_pairwise_distance"),
        "unique_fraction_canonical_draws": (
            unique_draws / float(len(draw_df)) if unique_draws is not None and len(draw_df) else None
        ),
        "unique_fraction_canonical_elites": (
            unique_elites / float(len(elites_df)) if unique_elites is not None and len(elites_df) else None
        ),
    }
    return pd.DataFrame([summary]), nn_df


def _prepare_analysis_root(
    analysis_root_path: Path,
    *,
    analysis_id: str,
) -> Path | None:
    if not analysis_root_path.exists():
        return None
    managed_paths = _analysis_managed_paths(analysis_root_path)
    if not managed_paths:
        return None
    prev_root = analysis_root_path / f".analysis_prev_{analysis_id}"
    prev_root.mkdir(parents=True, exist_ok=False)
    for path in managed_paths:
        shutil.move(str(path), prev_root / path.name)
    return prev_root


def _finalize_analysis_root(
    analysis_root_path: Path,
    tmp_root: Path,
    *,
    archive: bool,
    prev_root: Path | None,
) -> None:
    for path in sorted(tmp_root.iterdir()):
        shutil.move(str(path), analysis_root_path / path.name)
    shutil.rmtree(tmp_root, ignore_errors=True)
    if prev_root is None:
        return
    if not archive:
        shutil.rmtree(prev_root, ignore_errors=True)
        return
    prev_id = None
    summary_file = summary_path(prev_root)
    if summary_file.exists():
        try:
            payload = json.loads(summary_file.read_text())
            if isinstance(payload, dict):
                prev_id = payload.get("analysis_id")
        except Exception:
            prev_id = None
    if not prev_id:
        prev_id = prev_root.name.replace(".analysis_prev_", "")
    archive_root = analysis_root_path / "_archive" / str(prev_id)
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    if archive_root.exists():
        shutil.rmtree(archive_root)
    shutil.move(str(prev_root), archive_root)


def _analysis_managed_paths(analysis_root_path: Path) -> list[Path]:
    managed = [
        analysis_root_path / "analysis",
        analysis_root_path / "plots",
        analysis_root_path / "notebook__run_overview.py",
        analysis_root_path / "tables",
        analysis_root_path / "analysis_used.yaml",
        analysis_root_path / "summary.json",
        analysis_root_path / "report.json",
        analysis_root_path / "report.md",
        analysis_root_path / "plot_manifest.json",
        analysis_root_path / "table_manifest.json",
        analysis_root_path / "manifest.json",
    ]
    managed.extend(sorted(analysis_root_path.glob("plot__*")))
    managed.extend(sorted(analysis_root_path.glob("table__*")))
    return [path for path in managed if path.exists()]


def _delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def run_analyze(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    runs_override: list[str] | None = None,
    use_latest: bool = False,
) -> list[Path]:
    if cfg.analysis is None:
        raise ValueError("analysis section is required for analyze")
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")

    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    ensure_mpl_cache(catalog_root)
    ensure_arviz_data_dir(catalog_root)

    from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance
    from dnadesign.cruncher.analysis.plots.health_panel import plot_health_panel
    from dnadesign.cruncher.analysis.plots.opt_trajectory import plot_opt_trajectory, plot_opt_trajectory_sweep
    from dnadesign.cruncher.analysis.plots.overlap import plot_overlap_panel

    plan = resolve_analysis_plan(cfg)
    analysis_cfg = plan.analysis_cfg
    runs = _resolve_run_names(cfg, config_path, runs_override=runs_override, use_latest=use_latest)
    results: list[Path] = []

    for run_name in runs:
        run_dir = _resolve_run_dir(cfg, config_path, run_name)
        manifest = load_manifest(run_dir)
        optimizer_stats = _resolve_optimizer_stats(manifest, run_dir)
        lockfile_path = manifest.get("lockfile_path")
        lockfile_sha = manifest.get("lockfile_sha256")
        if lockfile_path and lockfile_sha:
            lock_path = Path(str(lockfile_path))
            if not lock_path.exists():
                raise FileNotFoundError(f"Lockfile referenced by run manifest missing: {lock_path}")
            current_sha = sha256_path(lock_path)
            if str(current_sha) != str(lockfile_sha):
                raise ValueError("Lockfile checksum mismatch (run manifest does not match current lockfile).")
        pwms, used_cfg = _load_pwms_from_config(run_dir)
        tf_names = _resolve_tf_names(used_cfg, pwms)
        sample_meta = _resolve_sample_meta(cfg, used_cfg, manifest)
        analysis_id = _analysis_id()
        created_at = datetime.now(timezone.utc).isoformat()

        analysis_root_path = analysis_root(run_dir)
        tmp_root = analysis_root_path / ".analysis_tmp"
        if tmp_root.exists():
            raise RuntimeError(
                f"Analyze already in progress for run '{run_name}' (lock: {tmp_root}). "
                "If no analyze is running, remove the stale analysis temp directory."
            )
        try:
            tmp_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Analyze already in progress for run '{run_name}' (lock: {tmp_root}). "
                "If no analyze is running, remove the stale analysis temp directory."
            ) from exc

        prev_root = None

        analysis_used_file = analysis_used_path(tmp_root)

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
        if not baseline_file.exists():
            raise FileNotFoundError(f"Missing random baseline parquet: {baseline_file}")
        if not baseline_hits_file.exists():
            raise FileNotFoundError(f"Missing random baseline hits parquet: {baseline_hits_file}")

        sequences_df = read_parquet(sequences_file)
        elites_df = read_parquet(elites_file)
        hits_df = load_elites_hits(hits_file)
        baseline_df = read_parquet(baseline_file)
        baseline_hits_df = load_baseline_hits(baseline_hits_file)

        trace_file = trace_path(run_dir)
        trace_idata = None
        if trace_file.exists():
            import arviz as az

            trace_idata = az.from_netcdf(trace_file)

        elites_meta = _load_elites_meta(elites_yaml_path(run_dir))

        table_ext = analysis_cfg.table_format
        score_summary_path = analysis_table_path(tmp_root, "scores_summary", table_ext)
        topk_path = analysis_table_path(tmp_root, "elites_topk", table_ext)
        joint_metrics_path = analysis_table_path(tmp_root, "metrics_joint", table_ext)
        overlap_pair_path = analysis_table_path(tmp_root, "overlap_pair_summary", table_ext)
        overlap_elite_path = analysis_table_path(tmp_root, "overlap_per_elite", table_ext)
        diagnostics_path = analysis_table_path(tmp_root, "diagnostics_summary", "json")
        objective_path = analysis_table_path(tmp_root, "objective_components", "json")
        elites_mmr_path = analysis_table_path(tmp_root, "elites_mmr_summary", table_ext)
        nn_distance_path = analysis_table_path(tmp_root, "elites_nn_distance", table_ext)
        trajectory_path = analysis_table_path(tmp_root, "opt_trajectory_points", table_ext)
        trajectory_particles_path = analysis_table_path(tmp_root, "opt_trajectory_particles", table_ext)

        from dnadesign.cruncher.analysis.plots.summary import (
            score_frame_from_df,
            write_elite_topk,
            write_joint_metrics,
            write_score_summary,
        )

        score_df = score_frame_from_df(sequences_df, tf_names)
        write_score_summary(score_df=score_df, tf_names=tf_names, out_path=score_summary_path)

        top_k = sample_meta.top_k if sample_meta.top_k else len(elites_df)
        write_elite_topk(elites_df=elites_df, tf_names=tf_names, out_path=topk_path, top_k=top_k)
        write_joint_metrics(elites_df=elites_df, tf_names=tf_names, out_path=joint_metrics_path)

        objective_from_manifest = manifest.get("objective")
        if objective_from_manifest is None:
            objective_from_manifest = {}
        if not isinstance(objective_from_manifest, dict):
            raise ValueError("Run manifest field 'objective' must be an object when provided.")
        sample_payload = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
        objective_used = sample_payload.get("objective") if isinstance(sample_payload, dict) else None
        scoring_used = objective_used.get("scoring") if isinstance(objective_used, dict) else None
        pseudocounts_raw = 0.10
        if isinstance(scoring_used, dict) and scoring_used.get("pwm_pseudocounts") is not None:
            pseudocounts_raw = float(scoring_used.get("pwm_pseudocounts"))
        log_odds_clip_raw = None
        if isinstance(scoring_used, dict) and scoring_used.get("log_odds_clip") is not None:
            log_odds_clip_raw = float(scoring_used.get("log_odds_clip"))
        bidirectional = bool(objective_from_manifest.get("bidirectional", True))
        beta_ladder = None
        if isinstance(optimizer_stats, dict):
            ladder_payload = optimizer_stats.get("beta_ladder_final")
            if ladder_payload is not None:
                if not isinstance(ladder_payload, list):
                    raise ValueError("optimizer_stats.beta_ladder_final must be a list when provided.")
                beta_ladder = [float(v) for v in ladder_payload]

        try:
            trajectory_df = build_trajectory_points(
                sequences_df,
                tf_names,
                max_points=analysis_cfg.max_points,
                objective_config=objective_from_manifest,
                beta_ladder=beta_ladder,
            )
            trajectory_df = add_raw_llr_objective(
                trajectory_df,
                tf_names,
                pwms=pwms,
                objective_config=objective_from_manifest,
                bidirectional=bidirectional,
                pwm_pseudocounts=pseudocounts_raw,
                log_odds_clip=log_odds_clip_raw,
            )
            baseline_plot_df = add_raw_llr_objective(
                baseline_df,
                tf_names,
                pwms=pwms,
                objective_config=objective_from_manifest,
                bidirectional=bidirectional,
                pwm_pseudocounts=pseudocounts_raw,
                log_odds_clip=log_odds_clip_raw,
            )
        except Exception:
            shutil.rmtree(tmp_root, ignore_errors=True)
            raise
        _write_table(trajectory_df, trajectory_path)
        trajectory_particles_df = build_particle_trajectory_points(
            trajectory_df,
            max_points=analysis_cfg.max_points,
        )
        _write_table(trajectory_particles_df, trajectory_particles_path)

        overlap_summary_df, elite_overlap_df, overlap_summary = compute_overlap_tables(
            elites_df, hits_df, tf_names, include_sequences=False
        )
        _write_table(overlap_summary_df, overlap_pair_path)
        _write_table(elite_overlap_df, overlap_elite_path)

        diagnostics_payload = summarize_sampling_diagnostics(
            trace_idata=trace_idata,
            sequences_df=sequences_df,
            elites_df=elites_df,
            elites_hits_df=hits_df,
            tf_names=tf_names,
            optimizer=manifest.get("optimizer"),
            optimizer_stats=optimizer_stats,
            mode=sample_meta.mode,
            optimizer_kind=sample_meta.optimizer_kind,
            sample_meta={
                "chains": sample_meta.chains,
                "draws": sample_meta.draws,
                "tune": sample_meta.tune,
                "mode": sample_meta.mode,
                "optimizer_kind": sample_meta.optimizer_kind,
                "top_k": sample_meta.top_k,
                "dsdna_canonicalize": sample_meta.bidirectional,
            },
            trace_required=False,
            overlap_summary=overlap_summary,
        )
        _write_json(diagnostics_path, diagnostics_payload)

        objective_components = compute_objective_components(
            sequences_df=sequences_df,
            tf_names=tf_names,
            top_k=sample_meta.top_k,
            overlap_total_bp_median=overlap_summary.get("overlap_total_bp_median"),
        )
        _write_json(objective_path, objective_components)

        elites_mmr_df, nn_df = _summarize_elites_mmr(
            elites_df,
            hits_df,
            sequences_df,
            elites_meta,
            tf_names,
            pwms,
            bidirectional=bool(sample_meta.bidirectional),
        )
        _write_table(elites_mmr_df, elites_mmr_path)
        _write_table(nn_df, nn_distance_path)

        baseline_seed = _resolve_baseline_seed(baseline_df)
        baseline_nn = compute_baseline_nn_distances(
            baseline_hits_df,
            tf_names,
            pwms,
            seed=baseline_seed,
        )

        plot_format = analysis_cfg.plot_format
        plot_dpi = analysis_cfg.plot_dpi
        plot_kwargs = {"dpi": plot_dpi, "png_compress_level": 9}

        focus_pair = _select_tf_pair(tf_names, analysis_cfg.pairwise)
        trajectory_tf_pair = _resolve_trajectory_tf_pair(tf_names, analysis_cfg.pairwise)
        trajectory_scale = str(analysis_cfg.trajectory_scatter_scale)
        sequence_length = manifest.get("sequence_length")
        if sequence_length is None and "sequence" in sequences_df.columns and not sequences_df.empty:
            sequence_length = int(sequences_df["sequence"].astype(str).str.len().iloc[0])
        if sequence_length is None:
            raise ValueError("Run manifest missing sequence_length required for trajectory consensus anchors.")
        anchor_objective_cfg = dict(objective_from_manifest)
        anchor_objective_cfg["score_scale"] = trajectory_scale
        consensus_anchors = compute_consensus_anchors(
            pwms=pwms,
            tf_names=tf_names,
            sequence_length=int(sequence_length),
            objective_config=anchor_objective_cfg,
            x_metric=f"score_{trajectory_tf_pair[0]}",
            y_metric=f"score_{trajectory_tf_pair[1]}",
        )

        _write_analysis_used(
            analysis_used_file,
            analysis_cfg.model_dump(mode="json"),
            analysis_id,
            run_name,
            extras={
                "tf_pair_mode": analysis_cfg.pairwise,
                "tf_pair_selected": list(focus_pair) if focus_pair else None,
                "trajectory_tf_pair": list(trajectory_tf_pair),
                "trajectory_scatter_scale": trajectory_scale,
                "trajectory_sweep_y_column": analysis_cfg.trajectory_sweep_y_column,
            },
        )

        plot_entries: list[dict[str, object]] = []
        plot_artifacts: list[dict[str, object]] = []

        def _record_plot(spec_key: str, output: Path, generated: bool, skip_reason: str | None) -> None:
            spec = next(spec for spec in PLOT_SPECS if spec.key == spec_key)
            rel_output = output.relative_to(tmp_root)
            final_output = analysis_root_path / rel_output
            plot_entries.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "group": spec.group,
                    "description": spec.description,
                    "requires": list(spec.requires),
                    "outputs": [{"path": str(rel_output), "exists": output.exists()}],
                    "generated": generated,
                    "skipped": not generated,
                    "skip_reason": skip_reason,
                }
            )
            if generated and output.exists():
                plot_artifacts.append(
                    artifact_entry(
                        final_output,
                        run_dir,
                        kind="plot",
                        label=spec.label,
                        stage="analysis",
                    )
                )

        plot_trajectory_path = analysis_plot_path(tmp_root, "opt_trajectory", plot_format)
        plot_trajectory_sweep_path = analysis_plot_path(tmp_root, "opt_trajectory_sweep", plot_format)
        if trajectory_particles_df.empty:
            raise ValueError("Particle trajectory points are required for opt trajectory plot.")
        plot_opt_trajectory(
            trajectory_df=trajectory_particles_df,
            baseline_df=baseline_plot_df,
            tf_pair=trajectory_tf_pair,
            scatter_scale=trajectory_scale,
            consensus_anchors=consensus_anchors,
            out_path=plot_trajectory_path,
            stride=analysis_cfg.trajectory_stride,
            alpha_min=analysis_cfg.trajectory_particle_alpha_min,
            alpha_max=analysis_cfg.trajectory_particle_alpha_max,
            slot_overlay=analysis_cfg.trajectory_slot_overlay,
            **plot_kwargs,
        )
        _record_plot("opt_trajectory", plot_trajectory_path, True, None)
        plot_opt_trajectory_sweep(
            trajectory_df=trajectory_particles_df,
            y_column=str(analysis_cfg.trajectory_sweep_y_column),
            out_path=plot_trajectory_sweep_path,
            stride=analysis_cfg.trajectory_stride,
            alpha_min=analysis_cfg.trajectory_particle_alpha_min,
            alpha_max=analysis_cfg.trajectory_particle_alpha_max,
            slot_overlay=analysis_cfg.trajectory_slot_overlay,
            **plot_kwargs,
        )
        _record_plot("opt_trajectory_sweep", plot_trajectory_sweep_path, True, None)

        plot_nn_path = analysis_plot_path(tmp_root, "elites_nn_distance", plot_format)
        plot_elites_nn_distance(nn_df, plot_nn_path, baseline_nn=pd.Series(baseline_nn), **plot_kwargs)
        _record_plot("elites_nn_distance", plot_nn_path, True, None)

        plot_overlap_path = analysis_plot_path(tmp_root, "overlap_panel", plot_format)
        overlap_skip_reason = None
        if len(tf_names) < 2:
            overlap_skip_reason = "n_tf < 2"
        elif len(elites_df) < 2:
            overlap_skip_reason = "elites_count < 2"
        if overlap_skip_reason is not None:
            _record_plot("overlap_panel", plot_overlap_path, False, overlap_skip_reason)
        else:
            plot_overlap_panel(
                overlap_summary_df,
                elite_overlap_df,
                tf_names,
                plot_overlap_path,
                focus_pair=focus_pair,
                **plot_kwargs,
            )
            _record_plot("overlap_panel", plot_overlap_path, True, None)

        plot_health_path = analysis_plot_path(tmp_root, "health_panel", plot_format)
        if trace_idata is None:
            _record_plot("health_panel", plot_health_path, False, "trace disabled")
        else:
            plot_health_panel(optimizer_stats, plot_health_path, **plot_kwargs)
            _record_plot("health_panel", plot_health_path, True, None)

        table_entries = [
            {
                "key": "scores_summary",
                "label": "Per-TF summary",
                "path": score_summary_path.name,
                "exists": True,
            },
            {
                "key": "elites_topk",
                "label": "Elite top-K",
                "path": topk_path.name,
                "exists": True,
            },
            {
                "key": "metrics_joint",
                "label": "Joint score metrics",
                "path": joint_metrics_path.name,
                "exists": True,
            },
            {
                "key": "opt_trajectory_points",
                "label": "Optimization trajectory points",
                "path": trajectory_path.name,
                "purpose": "plot_support",
                "exists": True,
            },
            {
                "key": "opt_trajectory_particles",
                "label": "Optimization particle trajectories",
                "path": trajectory_particles_path.name,
                "purpose": "plot_support",
                "exists": True,
            },
            {
                "key": "overlap_pair_summary",
                "label": "Overlap pair summary",
                "path": overlap_pair_path.name,
                "exists": True,
            },
            {
                "key": "overlap_per_elite",
                "label": "Overlap per elite",
                "path": overlap_elite_path.name,
                "exists": True,
            },
            {
                "key": "diagnostics_summary",
                "label": "Diagnostics summary",
                "path": diagnostics_path.name,
                "exists": True,
            },
            {
                "key": "objective_components",
                "label": "Objective components",
                "path": objective_path.name,
                "exists": True,
            },
            {
                "key": "elites_mmr_summary",
                "label": "Elites MMR summary",
                "path": elites_mmr_path.name,
                "exists": True,
            },
            {
                "key": "elites_nn_distance",
                "label": "Elites NN distance",
                "path": nn_distance_path.name,
                "exists": True,
            },
        ]

        analysis_artifacts = [
            artifact_entry(
                analysis_root_path / score_summary_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / topk_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / joint_metrics_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / trajectory_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / overlap_pair_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / overlap_elite_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / diagnostics_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / objective_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / elites_mmr_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / nn_distance_path.relative_to(tmp_root), run_dir, kind="table", stage="analysis"
            ),
            artifact_entry(
                analysis_root_path / trajectory_particles_path.relative_to(tmp_root),
                run_dir,
                kind="table",
                stage="analysis",
            ),
        ] + plot_artifacts

        build_analysis_manifests(
            analysis_id=analysis_id,
            created_at=created_at,
            analysis_root=tmp_root,
            analysis_used_file=analysis_used_file,
            plot_entries=plot_entries,
            table_entries=table_entries,
            analysis_artifacts=analysis_artifacts,
        )
        report_json = report_json_path(tmp_root)
        report_md = report_md_path(tmp_root)
        report_payload = build_report_payload(
            analysis_root=tmp_root,
            summary_payload={
                "analysis_id": analysis_id,
                "run": run_name,
                "tf_names": tf_names,
                "diagnostics": diagnostics_payload,
                "objective_components": objective_components,
                "overlap_summary": overlap_summary,
            },
            diagnostics_payload=diagnostics_payload,
            objective_components=objective_components,
            overlap_summary=overlap_summary,
            analysis_used_payload={"analysis": analysis_cfg.model_dump(mode="json")},
        )
        write_report_json(report_json, report_payload)
        write_report_md(report_md, report_payload, analysis_root=tmp_root)
        analysis_artifacts.append(
            artifact_entry(report_json_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
        )
        analysis_artifacts.append(
            artifact_entry(report_md_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
        )

        build_analysis_manifests(
            analysis_id=analysis_id,
            created_at=created_at,
            analysis_root=tmp_root,
            analysis_used_file=analysis_used_file,
            plot_entries=plot_entries,
            table_entries=table_entries,
            analysis_artifacts=analysis_artifacts,
        )
        analysis_artifacts.append(
            artifact_entry(plot_manifest_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
        )
        analysis_artifacts.append(
            artifact_entry(table_manifest_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
        )
        analysis_artifacts.append(
            artifact_entry(analysis_manifest_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
        )

        summary_payload = {
            "analysis_id": analysis_id,
            "run": run_name,
            "created_at": created_at,
            "analysis_dir": str(analysis_root_path.resolve()),
            "analysis_used": str(analysis_used_path(analysis_root_path).relative_to(run_dir)),
            "plot_manifest": str(plot_manifest_path(analysis_root_path).relative_to(run_dir)),
            "table_manifest": str(table_manifest_path(analysis_root_path).relative_to(run_dir)),
            "report_json": str(report_json_path(analysis_root_path).relative_to(run_dir)),
            "report_md": str(report_md_path(analysis_root_path).relative_to(run_dir)),
            "tf_names": tf_names,
            "diagnostics": diagnostics_payload,
            "objective_components": objective_components,
            "overlap_summary": overlap_summary,
            "artifacts": analysis_artifacts,
            "version": _get_version(),
        }

        summary_file = summary_path(tmp_root)
        _write_json(summary_file, summary_payload)

        try:
            prev_root = _prepare_analysis_root(
                analysis_root_path,
                analysis_id=analysis_id,
            )
            _finalize_analysis_root(
                analysis_root_path,
                tmp_root,
                archive=analysis_cfg.archive,
                prev_root=prev_root,
            )
        except Exception:
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            if prev_root is not None and prev_root.exists():
                for path in _analysis_managed_paths(analysis_root_path):
                    _delete_path(path)
                for child in prev_root.iterdir():
                    shutil.move(str(child), analysis_root_path / child.name)
                shutil.rmtree(prev_root, ignore_errors=True)
            raise

        _prune_latest_analysis_artifacts(manifest)
        append_artifacts(manifest, analysis_artifacts)
        write_manifest(run_dir, manifest)

        results.append(analysis_root_path)

    return results
