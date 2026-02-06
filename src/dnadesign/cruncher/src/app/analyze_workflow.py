"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze_workflow.py

Analyze sampling runs and produce summary reports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.analysis.diversity import (
    compute_elite_distance_matrix,
    compute_elites_nn_distance_table,
    representative_elite_ids,
    summarize_elite_distances,
)
from dnadesign.cruncher.analysis.hits import load_elites_hits
from dnadesign.cruncher.analysis.layout import (
    analysis_manifest_path,
    analysis_root,
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
from dnadesign.cruncher.analysis.plots.dashboard import plot_dashboard
from dnadesign.cruncher.analysis.plots.diag_panel import plot_diag_panel
from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance
from dnadesign.cruncher.analysis.plots.opt_trajectory import plot_opt_trajectory
from dnadesign.cruncher.analysis.plots.overlap import plot_overlap_panel
from dnadesign.cruncher.analysis.report import build_report_payload, write_report_json, write_report_md
from dnadesign.cruncher.analysis.trajectory import build_trajectory_points
from dnadesign.cruncher.app.analyze.archive import _prune_latest_analysis_artifacts
from dnadesign.cruncher.app.analyze.manifests import build_analysis_manifests
from dnadesign.cruncher.app.analyze.metadata import (
    _analysis_id,
    _get_version,
    _load_pwms_from_config,
    _resolve_sample_meta,
)
from dnadesign.cruncher.app.analyze.plan import resolve_analysis_plan
from dnadesign.cruncher.app.run_service import list_runs
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_yaml
from dnadesign.cruncher.artifacts.entries import append_artifacts, artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    elites_hits_path,
    elites_path,
    elites_yaml_path,
    random_baseline_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest, write_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.core.sequence import identity_key
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)

__all__ = ["run_analyze"]


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
        return [runs[0].name]
    raise ValueError("analysis.run_selector must be 'latest' or 'explicit'")


def _resolve_run_dir(cfg: CruncherConfig, config_path: Path, run_name: str) -> Path:
    runs = list_runs(cfg, config_path, stage="sample")
    for run in runs:
        if run.name == run_name:
            return run.run_dir
    raise FileNotFoundError(f"Run '{run_name}' not found in run index.")


def _load_elites_meta(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        return {}
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
    atomic_write_yaml(path, payload, sort_keys=False, default_flow_style=False)


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
    archive: bool,
    analysis_id: str,
) -> Path | None:
    if not analysis_root_path.exists():
        return None
    prev_root = analysis_root_path.parent / f".analysis_prev_{analysis_id}"
    os.replace(analysis_root_path, prev_root)
    return prev_root if archive else prev_root


def _finalize_analysis_root(
    analysis_root_path: Path,
    tmp_root: Path,
    *,
    archive: bool,
    prev_root: Path | None,
) -> None:
    os.replace(tmp_root, analysis_root_path)
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
    shutil.move(str(prev_root), archive_root)


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

    ensure_mpl_cache(resolve_catalog_root(config_path, cfg.motif_store.catalog_root))
    import arviz as az

    plan = resolve_analysis_plan(cfg)
    analysis_cfg = plan.analysis_cfg
    runs = _resolve_run_names(cfg, config_path, runs_override=runs_override, use_latest=use_latest)
    results: list[Path] = []

    for run_name in runs:
        run_dir = _resolve_run_dir(cfg, config_path, run_name)
        manifest = load_manifest(run_dir)
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
        tmp_root = analysis_root_path.parent / f".analysis_tmp_{analysis_id}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        prev_root = _prepare_analysis_root(
            analysis_root_path,
            archive=analysis_cfg.archive,
            analysis_id=analysis_id,
        )

        analysis_used_file = analysis_used_path(tmp_root)

        sequences_file = sequences_path(run_dir)
        elites_file = elites_path(run_dir)
        hits_file = elites_hits_path(run_dir)
        baseline_file = random_baseline_path(run_dir)
        if not sequences_file.exists():
            raise FileNotFoundError(f"Missing sequences parquet: {sequences_file}")
        if not elites_file.exists():
            raise FileNotFoundError(f"Missing elites parquet: {elites_file}")
        if not hits_file.exists():
            raise FileNotFoundError(f"Missing elites hits parquet: {hits_file}")
        if not baseline_file.exists():
            raise FileNotFoundError(f"Missing random baseline parquet: {baseline_file}")

        sequences_df = read_parquet(sequences_file)
        elites_df = read_parquet(elites_file)
        hits_df = load_elites_hits(hits_file)
        baseline_df = read_parquet(baseline_file)

        trace_file = trace_path(run_dir)
        trace_idata = az.from_netcdf(trace_file) if trace_file.exists() else None

        elites_meta = _load_elites_meta(elites_yaml_path(run_dir))

        table_ext = analysis_cfg.table_format
        score_summary_path = tmp_root / f"table__scores__summary.{table_ext}"
        topk_path = tmp_root / f"table__elites__topk.{table_ext}"
        joint_metrics_path = tmp_root / f"table__metrics__joint.{table_ext}"
        overlap_pair_path = tmp_root / f"table__overlap__pair_summary.{table_ext}"
        overlap_elite_path = tmp_root / f"table__overlap__per_elite.{table_ext}"
        diagnostics_path = tmp_root / "table__diagnostics__summary.json"
        objective_path = tmp_root / "table__objective__components.json"
        elites_mmr_path = tmp_root / f"table__elites__mmr_summary.{table_ext}"
        nn_distance_path = tmp_root / f"table__elites__nn_distance.{table_ext}"
        trajectory_path = tmp_root / f"table__opt__trajectory_points.{table_ext}"

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

        trajectory_df = build_trajectory_points(
            sequences_df,
            tf_names,
            max_points=analysis_cfg.max_points,
        )
        _write_table(trajectory_df, trajectory_path)

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
            optimizer_stats=manifest.get("optimizer_stats"),
            mode=sample_meta.mode,
            optimizer_kind=sample_meta.optimizer_kind,
            sample_meta={
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

        plot_format = analysis_cfg.plot_format
        plot_dpi = analysis_cfg.plot_dpi
        plot_kwargs = {"dpi": plot_dpi, "png_compress_level": 9}

        score_scale_from_run = None
        sample_payload = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
        if isinstance(sample_payload, dict):
            objective_payload = sample_payload.get("objective")
            if isinstance(objective_payload, dict):
                score_scale_from_run = objective_payload.get("score_scale")
        plot_score_scale_used = str(score_scale_from_run) if score_scale_from_run else None
        focus_pair = _select_tf_pair(tf_names, analysis_cfg.pairwise)

        _write_analysis_used(
            analysis_used_file,
            analysis_cfg.model_dump(mode="json"),
            analysis_id,
            run_name,
            extras={
                "plot_score_scale_used": plot_score_scale_used,
                "score_scale_from_run": score_scale_from_run,
                "tf_pair_mode": analysis_cfg.pairwise,
                "tf_pair_selected": list(focus_pair) if focus_pair else None,
            },
        )

        plot_entries: list[dict[str, object]] = []
        plot_artifacts: list[dict[str, object]] = []

        def _record_plot(spec_key: str, output: Path, generated: bool, skip_reason: str | None) -> None:
            spec = next(spec for spec in PLOT_SPECS if spec.key == spec_key)
            final_output = analysis_root_path / output.name
            plot_entries.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "group": spec.group,
                    "description": spec.description,
                    "requires": list(spec.requires),
                    "outputs": [{"path": output.name, "exists": output.exists()}],
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

        plot_dashboard_path = tmp_root / f"plot__run__dashboard.{plot_format}"
        try:
            plot_dashboard(sequences_df, elites_df, tf_names, plot_dashboard_path, **plot_kwargs)
            _record_plot("run_dashboard", plot_dashboard_path, True, None)
        except Exception as exc:
            _record_plot("run_dashboard", plot_dashboard_path, False, str(exc))

        plot_trajectory_path = tmp_root / f"plot__opt__trajectory.{plot_format}"
        try:
            plot_opt_trajectory(
                trajectory_df,
                baseline_df,
                tf_names,
                plot_trajectory_path,
                score_scale=plot_score_scale_used,
                **plot_kwargs,
            )
            _record_plot("opt_trajectory", plot_trajectory_path, True, None)
        except Exception as exc:
            _record_plot("opt_trajectory", plot_trajectory_path, False, str(exc))

        plot_nn_path = tmp_root / f"plot__elites__nn_distance.{plot_format}"
        try:
            plot_elites_nn_distance(nn_df, plot_nn_path, **plot_kwargs)
            _record_plot("elites_nn_distance", plot_nn_path, True, None)
        except Exception as exc:
            _record_plot("elites_nn_distance", plot_nn_path, False, str(exc))

        plot_overlap_path = tmp_root / f"plot__overlap__panel.{plot_format}"
        try:
            plot_overlap_panel(
                overlap_summary_df,
                elite_overlap_df,
                tf_names,
                plot_overlap_path,
                focus_pair=focus_pair,
                **plot_kwargs,
            )
            _record_plot("overlap_panel", plot_overlap_path, True, None)
        except Exception as exc:
            _record_plot("overlap_panel", plot_overlap_path, False, str(exc))

        plot_diag_path = tmp_root / f"plot__diag__panel.{plot_format}"
        if trace_idata is None:
            _record_plot("diag_panel", plot_diag_path, False, "Trace not available.")
        else:
            try:
                plot_diag_panel(trace_idata, manifest.get("optimizer_stats"), plot_diag_path, **plot_kwargs)
                _record_plot("diag_panel", plot_diag_path, True, None)
            except Exception as exc:
                _record_plot("diag_panel", plot_diag_path, False, str(exc))

        table_entries = [
            {"key": "scores_summary", "label": "Per-TF summary", "path": score_summary_path.name, "exists": True},
            {"key": "elites_topk", "label": "Elite top-K", "path": topk_path.name, "exists": True},
            {"key": "metrics_joint", "label": "Joint score metrics", "path": joint_metrics_path.name, "exists": True},
            {
                "key": "opt_trajectory_points",
                "label": "Optimization trajectory points",
                "path": trajectory_path.name,
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
            {"key": "elites_mmr_summary", "label": "Elites MMR summary", "path": elites_mmr_path.name, "exists": True},
            {
                "key": "elites_nn_distance",
                "label": "Elites NN distance",
                "path": nn_distance_path.name,
                "exists": True,
            },
        ]

        analysis_artifacts = [
            artifact_entry(analysis_root_path / score_summary_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / topk_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / joint_metrics_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / trajectory_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / overlap_pair_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / overlap_elite_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / diagnostics_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / objective_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / elites_mmr_path.name, run_dir, kind="table", stage="analysis"),
            artifact_entry(analysis_root_path / nn_distance_path.name, run_dir, kind="table", stage="analysis"),
        ] + plot_artifacts

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
        report_json = report_json_path(tmp_root)
        write_report_json(report_json, report_payload)
        report_md = report_md_path(tmp_root)
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

        _finalize_analysis_root(
            analysis_root_path,
            tmp_root,
            archive=analysis_cfg.archive,
            prev_root=prev_root,
        )

        if not analysis_cfg.archive:
            _prune_latest_analysis_artifacts(manifest)
        append_artifacts(manifest, analysis_artifacts)
        write_manifest(run_dir, manifest)

        results.append(analysis_root_path)

    return results
