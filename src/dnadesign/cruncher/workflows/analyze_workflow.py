"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/workflows/analyze_workflow.py

When run_analyze is called, it iterates through cfg.analysis.runs (a list of
existing “sample” run names), and for each run:
  1) It looks under the configured output folder for that sample's directory.
  2) Creates a new analysis run folder:
       <sample_run>/analysis/<analysis_id>/{plots,tables,notebooks}
  3) Writes analysis_used.yaml + summary.json, then produces plots/tables
     according to cfg.analysis.plots. Pairwise plots are only generated when
     analysis.tf_pair is explicitly provided.
  4) Appends analysis artifacts to run_manifest.json for CLI discovery.
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.services.run_service import list_runs
from dnadesign.cruncher.utils.artifacts import append_artifacts, artifact_entry
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.manifest import load_manifest
from dnadesign.cruncher.utils.mpl import ensure_mpl_cache
from dnadesign.cruncher.utils.parquet import read_parquet
from dnadesign.cruncher.workflows.analyze.plot_registry import PLOT_SPECS

logger = logging.getLogger(__name__)


def _load_pwms_from_config(run_dir: Path) -> tuple[dict[str, PWM], dict]:
    import numpy as np
    import yaml

    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml in {run_dir}")
    payload = yaml.safe_load(config_path.read_text()) or {}
    cruncher_cfg = payload.get("cruncher")
    if not isinstance(cruncher_cfg, dict):
        raise ValueError("config_used.yaml missing top-level 'cruncher' section.")
    pwms_info = cruncher_cfg.get("pwms_info")
    if not isinstance(pwms_info, dict) or not pwms_info:
        raise ValueError("config_used.yaml missing pwms_info; re-run `cruncher sample`.")
    pwms: dict[str, PWM] = {}
    for tf_name, info in pwms_info.items():
        matrix = info.get("pwm_matrix")
        if not matrix:
            raise ValueError(f"config_used.yaml missing pwm_matrix for TF '{tf_name}'.")
        pwms[tf_name] = PWM(name=tf_name, matrix=np.array(matrix, dtype=float))
    return pwms, cruncher_cfg


def _analysis_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{stamp}_{suffix}"


def _get_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("dnadesign")
    except Exception:
        return None


def _get_git_commit(path: Path) -> str | None:
    probe = path.resolve()
    for _ in range(6):
        git_dir = probe / ".git"
        if git_dir.exists():
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref:"):
                ref = head.split(" ", 1)[1].strip()
                ref_path = git_dir / ref
                if ref_path.exists():
                    return ref_path.read_text().strip()
            return head or None
        if probe.parent == probe:
            break
        probe = probe.parent
    return None


def _resolve_tf_pair(
    cfg: CruncherConfig,
    tf_names: list[str],
    tf_pair_override: tuple[str, str] | None = None,
) -> tuple[str, str] | None:
    pair = tf_pair_override
    if pair is None and cfg.analysis is not None:
        pair = cfg.analysis.tf_pair
    if pair is None:
        return None
    if len(pair) != 2:
        raise ValueError("analysis.tf_pair must contain exactly two TF names.")
    x_tf, y_tf = pair
    if x_tf not in tf_names or y_tf not in tf_names:
        raise ValueError(f"analysis.tf_pair must reference TFs in {tf_names}.")
    return x_tf, y_tf


def run_analyze(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    runs_override: list[str] | None = None,
    use_latest: bool = False,
    tf_pair_override: tuple[str, str] | None = None,
) -> list[Path]:
    """Iterate over runs listed in cfg.analysis.runs and generate diagnostics."""
    # Shared PWM store
    if cfg.analysis is None:
        raise ValueError("analysis section is required for analyze")
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")

    ensure_mpl_cache(config_path.parent / cfg.motif_store.catalog_root)
    import arviz as az
    import yaml

    from dnadesign.cruncher.workflows.analyze.per_pwm import gather_per_pwm_scores
    from dnadesign.cruncher.workflows.analyze.plots.diagnostics import (
        make_pair_idata,
        plot_autocorr,
        plot_ess,
        plot_pair_pwm_scores,
        plot_parallel_pwm_scores,
        plot_rank_diagnostic,
        plot_trace,
        report_convergence,
    )
    from dnadesign.cruncher.workflows.analyze.plots.scatter import plot_scatter
    from dnadesign.cruncher.workflows.analyze.plots.summary import (
        load_score_frame,
        plot_correlation_heatmap,
        plot_parallel_coords,
        plot_score_box,
        plot_score_hist,
        write_elite_topk,
        write_score_summary,
    )

    results_dir = config_path.parent / Path(cfg.out_dir)
    runs = runs_override if runs_override else (cfg.analysis.runs or [])
    if not runs:
        if not use_latest:
            raise ValueError("No analysis runs configured. Set analysis.runs, pass --run, or use --latest.")
        latest = list_runs(cfg, config_path, stage="sample")
        if not latest:
            raise ValueError("No sample runs found. Run `cruncher sample <config>` first.")
        runs = [latest[0].name]
        logger.info("Analyzing latest sample run: %s", runs[0])
    analysis_runs: list[Path] = []
    for run_name in runs:
        sample_dir = results_dir / run_name

        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Run '{run_name}' not found under {results_dir}")

        logger.info("Analyzing run: %s", run_name)
        manifest = load_manifest(sample_dir)
        if manifest.get("stage") != "sample":
            raise ValueError(f"Run '{run_name}' is not a sample run (stage={manifest.get('stage')})")
        lock_path_raw = manifest.get("lockfile_path")
        if not lock_path_raw:
            raise ValueError("Run manifest missing lockfile_path; re-run `cruncher sample`.")
        lock_path = Path(lock_path_raw)
        if not lock_path.exists():
            raise FileNotFoundError(f"Lockfile not found: {lock_path}")
        expected_sha = manifest.get("lockfile_sha256")
        if expected_sha:
            actual_sha = sha256_path(lock_path)
            if actual_sha != expected_sha:
                raise ValueError(f"Lockfile checksum mismatch for {lock_path}")

        # ── reconstruct PWMs from config_used.yaml for reproducibility ───────
        pwms, used_cfg = _load_pwms_from_config(sample_dir)
        tf_names = list(pwms.keys())
        used_sample = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
        bidirectional = cfg.sample.bidirectional
        if isinstance(used_sample, dict) and "bidirectional" in used_sample:
            bidirectional = bool(used_sample.get("bidirectional"))

        # ── locate newest elites parquet file ────────────────────────────────
        parquet_candidates = list(sample_dir.glob("cruncher_elites_*/*.parquet"))
        if not parquet_candidates:
            raise FileNotFoundError(f"No elites parquet found in {sample_dir}")
        latest_parquet = max(parquet_candidates, key=lambda p: p.stat().st_mtime)
        elites_df = read_parquet(latest_parquet)
        logger.info("Loaded %d elites from %s", len(elites_df), latest_parquet.relative_to(sample_dir))

        # ── load trace & sequences for downstream plots ───────────────────────
        seq_path, trace_path = sample_dir / "sequences.parquet", sample_dir / "trace.nc"
        if not seq_path.exists():
            raise FileNotFoundError(
                f"Missing sequences.parquet in {sample_dir}. Re-run `cruncher sample` with sample.save_sequences=true."
            )
        plots = cfg.analysis.plots
        needs_trace = plots.trace or plots.autocorr or plots.convergence
        trace_idata = None
        if needs_trace:
            if not trace_path.exists():
                raise FileNotFoundError(
                    f"Missing trace.nc in {sample_dir}. Re-run `cruncher sample` with sample.save_trace=true."
                )
            trace_idata = az.from_netcdf(trace_path)

        analysis_id = _analysis_id()
        analysis_root = sample_dir / "analysis"
        analysis_dir = analysis_root / analysis_id
        plots_dir = analysis_dir / "plots"
        tables_dir = analysis_dir / "tables"
        notebooks_dir = analysis_dir / "notebooks"
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        notebooks_dir.mkdir(parents=True, exist_ok=True)

        analysis_used_path = analysis_dir / "analysis_used.yaml"
        analysis_used_payload = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis": cfg.analysis.model_dump(),
        }
        analysis_used_path.write_text(yaml.safe_dump(analysis_used_payload, sort_keys=False))

        # gather per-PWM scores (for scatter etc.)
        per_pwm_path = tables_dir / "gathered_per_pwm_everyN.csv"
        gather_per_pwm_scores(
            sample_dir,
            cfg.analysis.subsampling_epsilon,
            pwms,
            bidirectional=bidirectional,
            scale=cfg.analysis.scatter_scale,
            out_path=per_pwm_path,
        )
        logger.info("Wrote per-PWM score table → %s", per_pwm_path.relative_to(sample_dir))

        score_df = load_score_frame(seq_path, tf_names)
        summary_path = tables_dir / "score_summary.csv"
        write_score_summary(score_df, tf_names, summary_path)

        topk_path = tables_dir / "elite_topk.csv"
        write_elite_topk(elites_df, tf_names, topk_path, top_k=cfg.sample.top_k if cfg.sample else 10)

        enabled_specs = [spec for spec in PLOT_SPECS if getattr(plots, spec.key, False)]
        if enabled_specs:
            logger.info("Enabled plots: %s", ", ".join(spec.key for spec in enabled_specs))
        tf_pair = _resolve_tf_pair(cfg, tf_names, tf_pair_override=tf_pair_override)
        pairwise_requested = any("tf_pair" in spec.requires for spec in enabled_specs)
        if pairwise_requested and tf_pair is None:
            raise ValueError("analysis.tf_pair is required when pairwise plots are enabled.")

        # ── diagnostics ───────────────────────────────────────────────────────
        if plots.trace:
            plot_trace(trace_idata, plots_dir)
        if plots.autocorr:
            plot_autocorr(trace_idata, plots_dir)
        if plots.convergence:
            report_convergence(trace_idata, plots_dir)
            plot_rank_diagnostic(trace_idata, plots_dir)
            plot_ess(trace_idata, plots_dir)
        if plots.pair_pwm or plots.parallel_pwm:
            idata_pair = make_pair_idata(sample_dir, cfg, tf_pair=tf_pair)
            if plots.pair_pwm:
                plot_pair_pwm_scores(idata_pair, plots_dir, cfg, tf_pair=tf_pair)
            if plots.parallel_pwm:
                plot_parallel_pwm_scores(idata_pair, plots_dir, cfg, tf_pair=tf_pair)

        # ── scatter plot ──────────────────────────────────────────────────────
        if plots.scatter_pwm:
            if not per_pwm_path.exists():
                raise FileNotFoundError(f"Missing gathered_per_pwm_everyN.csv in {tables_dir}")
            plot_scatter(
                sample_dir,
                pwms,
                cfg,
                tf_pair=tf_pair,
                per_pwm_path=per_pwm_path,
                out_dir=plots_dir,
                bidirectional=bidirectional,
            )
            logger.info("Wrote scatter_pwm.pdf")

        if plots.score_hist:
            plot_score_hist(score_df, tf_names, plots_dir / "score_hist.png")
        if plots.score_box:
            plot_score_box(score_df, tf_names, plots_dir / "score_box.png")
        if plots.correlation_heatmap:
            plot_correlation_heatmap(score_df, tf_names, plots_dir / "score_correlation.png")
        if plots.parallel_coords:
            if elites_df.empty:
                logger.warning("Skipping parallel coordinates: no elites available.")
            else:
                plot_parallel_coords(elites_df, tf_names, plots_dir / "parallel_coords.png")

        # ── top-5 tabular summary (unchanged logic – dynamic TF names) ────────
        if not elites_df.empty:
            if "rank" not in elites_df.columns:
                raise ValueError(
                    "Elites parquet missing 'rank' column. Re-run `cruncher sample` to regenerate elites.parquet."
                )
            if tf_pair is not None:
                x_tf, y_tf = tf_pair
                if f"score_{x_tf}" not in elites_df.columns or f"score_{y_tf}" not in elites_df.columns:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )
                logger.info("Top-5 elites (%s & %s):", x_tf, y_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.info(
                        "rank=%d seq=%s %s=%.1f %s=%.1f",
                        int(row["rank"]),
                        row["sequence"],
                        x_tf,
                        row[f"score_{x_tf}"],
                        y_tf,
                        row[f"score_{y_tf}"],
                    )
            elif len(tf_names) == 1:
                x_tf = tf_names[0]
                if f"score_{x_tf}" not in elites_df.columns:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )
                logger.info("Top-5 elites (%s):", x_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.info(
                        "rank=%d seq=%s %s=%.1f",
                        int(row["rank"]),
                        row["sequence"],
                        x_tf,
                        row[f"score_{x_tf}"],
                    )
            elif len(tf_names) > 1:
                missing_scores = [f"score_{tf}" for tf in tf_names if f"score_{tf}" not in elites_df.columns]
                if missing_scores:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )
                logger.info("Top-5 elites (all TFs):")
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    score_blob = " ".join(f"{tf}={row[f'score_{tf}']:.1f}" for tf in tf_names)
                    logger.info("rank=%d seq=%s %s", int(row["rank"]), row["sequence"], score_blob)
            else:
                logger.warning("No regulators configured; skipping elite summary.")

        artifacts: list[dict[str, object]] = [
            artifact_entry(analysis_used_path, sample_dir, kind="config", label="Analysis settings", stage="analysis"),
            artifact_entry(per_pwm_path, sample_dir, kind="table", label="Per-PWM scores (CSV)", stage="analysis"),
            artifact_entry(summary_path, sample_dir, kind="table", label="Per-TF summary (CSV)", stage="analysis"),
            artifact_entry(topk_path, sample_dir, kind="table", label="Elite top-K (CSV)", stage="analysis"),
        ]

        for plot_name in plots_dir.glob("*"):
            if plot_name.is_file():
                kind = "plot" if plot_name.suffix.lower() in {".png", ".pdf"} else "text"
                artifacts.append(
                    artifact_entry(plot_name, sample_dir, kind=kind, label=plot_name.name, stage="analysis")
                )

        inputs_payload = {
            "sequences.parquet": {
                "path": str(seq_path.relative_to(sample_dir)),
                "sha256": sha256_path(seq_path),
            },
            "config_used.yaml": {
                "path": "config_used.yaml",
                "sha256": sha256_path(sample_dir / "config_used.yaml"),
            },
        }
        if trace_path.exists():
            inputs_payload["trace.nc"] = {
                "path": str(trace_path.relative_to(sample_dir)),
                "sha256": sha256_path(trace_path),
            }
        summary_payload = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run": run_name,
            "run_dir": str(sample_dir.resolve()),
            "analysis_dir": str(analysis_dir.resolve()),
            "tf_names": tf_names,
            "analysis_config": cfg.analysis.model_dump(),
            "cruncher_version": _get_version(),
            "git_commit": _get_git_commit(config_path),
            "analysis_used": str(analysis_used_path.relative_to(sample_dir)),
            "config_used": "config_used.yaml",
            "inputs": inputs_payload,
            "artifacts": [item["path"] for item in artifacts],
        }
        summary_path_json = analysis_dir / "summary.json"
        summary_path_json.write_text(json.dumps(summary_payload, indent=2))
        artifacts.append(
            artifact_entry(summary_path_json, sample_dir, kind="json", label="Analysis summary", stage="analysis")
        )

        append_artifacts(manifest, artifacts)
        analysis_root.mkdir(parents=True, exist_ok=True)
        (analysis_root / "latest.txt").write_text(analysis_id)

        from dnadesign.cruncher.services.run_service import update_run_index_from_manifest
        from dnadesign.cruncher.utils.manifest import write_manifest

        write_manifest(sample_dir, manifest)
        update_run_index_from_manifest(config_path, sample_dir, manifest)
        logger.info("Analysis artifacts recorded (%s).", analysis_id)
        analysis_runs.append(analysis_dir)
    return analysis_runs
