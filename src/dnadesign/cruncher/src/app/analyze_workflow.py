"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze_workflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.layout import (
    ANALYSIS_LAYOUT_VERSION,
    analysis_meta_root,
    analysis_used_path,
    plot_manifest_path,
    summary_path,
    table_manifest_path,
)
from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.analysis.plot_registry import PLOT_SPECS
from dnadesign.cruncher.app.run_service import list_runs
from dnadesign.cruncher.artifacts.entries import (
    append_artifacts,
    artifact_entry,
    normalize_artifacts,
)
from dnadesign.cruncher.artifacts.layout import config_used_path, sequences_path, trace_path
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v2 import AnalysisConfig, CruncherConfig, SampleMovesConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)

_ANALYSIS_ITEMS = (
    "meta",
    "plots",
    "tables",
    "notebooks",
)


def _analysis_item_paths(analysis_root: Path) -> list[Path]:
    return [analysis_root / name for name in _ANALYSIS_ITEMS]


def _prune_latest_analysis_artifacts(manifest: dict) -> None:
    artifacts = normalize_artifacts(manifest.get("artifacts"))
    pruned: list[dict[str, object]] = []
    for item in artifacts:
        path = str(item.get("path") or "")
        norm_path = path.replace("\\", "/")
        if norm_path == "analysis" or norm_path.startswith("analysis/"):
            if not norm_path.startswith("analysis/_archive/"):
                continue
        pruned.append(item)
    manifest["artifacts"] = pruned


def _load_summary_id(analysis_root: Path) -> str | None:
    summary_file = summary_path(analysis_root)
    if not summary_file.exists():
        return None
    try:
        payload = json.loads(summary_file.read_text())
    except Exception as exc:
        raise ValueError(f"analysis summary is not valid JSON: {summary_file}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"analysis summary must be a JSON object: {summary_file}")
    analysis_id = payload.get("analysis_id")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise ValueError(f"analysis summary missing analysis_id: {summary_file}")
    return analysis_id


def _load_summary_payload(analysis_root: Path) -> dict | None:
    summary_file = summary_path(analysis_root)
    if not summary_file.exists():
        return None
    try:
        payload = json.loads(summary_file.read_text())
    except Exception as exc:
        raise ValueError(f"analysis summary is not valid JSON: {summary_file}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"analysis summary must be a JSON object: {summary_file}")
    return payload


def _analysis_signature(
    *,
    analysis_cfg: AnalysisConfig,
    override_payload: dict[str, object] | None,
    config_used_file: Path,
    sequences_file: Path,
    elites_file: Path,
    trace_file: Path,
) -> tuple[str, dict[str, object]]:
    inputs: dict[str, object] = {
        "config_used_sha256": sha256_path(config_used_file),
        "sequences_sha256": sha256_path(sequences_file),
        "elites_sha256": sha256_path(elites_file),
        "analysis_layout_version": ANALYSIS_LAYOUT_VERSION,
    }
    if trace_file.exists():
        inputs["trace_sha256"] = sha256_path(trace_file)
    payload = {
        "analysis": analysis_cfg.model_dump(),
        "analysis_overrides": override_payload or {},
        "inputs": inputs,
    }
    signature = sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return signature, payload


def _rewrite_manifest_paths(manifest: dict, analysis_id: str, moved_prefixes: list[str]) -> None:
    artifacts = manifest.get("artifacts") or []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        for prefix in moved_prefixes:
            old_prefix = f"analysis/{prefix}"
            if path == old_prefix or path.startswith(old_prefix):
                suffix = path[len(old_prefix) :]
                item["path"] = f"analysis/_archive/{analysis_id}/{prefix}{suffix}"
                break


def _update_archived_summary(archive_root: Path, analysis_id: str, moved_prefixes: list[str]) -> None:
    summary_file = summary_path(archive_root)
    if not summary_file.exists():
        return
    payload = json.loads(summary_file.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"analysis summary must be a JSON object: {summary_file}")

    def _rewrite_path(value: str) -> str:
        for prefix in moved_prefixes:
            old_prefix = f"analysis/{prefix}"
            if value == old_prefix or value.startswith(old_prefix):
                suffix = value[len(old_prefix) :]
                return f"analysis/_archive/{analysis_id}/{prefix}{suffix}"
        return value

    for key in ("analysis_used", "plot_manifest", "table_manifest"):
        raw = payload.get(key)
        if isinstance(raw, str):
            payload[key] = _rewrite_path(raw)

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        payload["artifacts"] = [_rewrite_path(item) if isinstance(item, str) else item for item in artifacts]

    payload["analysis_dir"] = str(archive_root.resolve())
    payload["archived_at"] = datetime.now(timezone.utc).isoformat()
    summary_file.write_text(json.dumps(payload, indent=2))


def _archive_existing_analysis(analysis_root: Path, manifest: dict, analysis_id: str) -> None:
    archive_root = analysis_root / "_archive" / analysis_id
    archive_root.mkdir(parents=True, exist_ok=True)
    moved_prefixes: list[str] = []
    for path in _analysis_item_paths(analysis_root):
        if not path.exists():
            continue
        moved_prefixes.append(path.name + ("/" if path.is_dir() else ""))
        shutil.move(str(path), archive_root / path.name)
    if moved_prefixes:
        _rewrite_manifest_paths(manifest, analysis_id, moved_prefixes)
        _update_archived_summary(archive_root, analysis_id, moved_prefixes)


def _clear_latest_analysis(analysis_root: Path) -> None:
    for path in _analysis_item_paths(analysis_root):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


@dataclass(frozen=True)
class SampleMeta:
    optimizer_kind: str
    chains: int
    draws: int
    tune: int
    move_probs: dict[str, float]
    cooling_kind: str
    pwm_sum_threshold: float
    bidirectional: bool
    top_k: int
    mode: str


def _load_pwms_from_config(run_dir: Path) -> tuple[dict[str, PWM], dict]:
    import numpy as np
    import yaml

    config_path = config_used_path(run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing meta/config_used.yaml in {run_dir}")
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


def _resolve_sample_meta(cfg: CruncherConfig, used_cfg: dict) -> SampleMeta:
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")
    used_sample = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
    if not isinstance(used_sample, dict):
        raise ValueError("config_used.yaml missing sample section; re-run `cruncher sample`.")

    def _require(path: list[str], label: str) -> object:
        cursor: object = used_sample
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                raise ValueError(f"config_used.yaml missing sample.{label}; re-run `cruncher sample`.")
            cursor = cursor[key]
        return cursor

    def _coerce_int(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config_used.yaml sample.{label} must be an integer.")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"config_used.yaml sample.{label} must be an integer.")
        return int(value)

    def _coerce_float(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config_used.yaml sample.{label} must be a number.")
        return float(value)

    optimizer_kind = str(_require(["optimizer", "name"], "optimizer.name"))
    draws = _coerce_int(_require(["budget", "draws"], "budget.draws"), "budget.draws")
    tune = _coerce_int(_require(["budget", "tune"], "budget.tune"), "budget.tune")
    restarts = _coerce_int(_require(["budget", "restarts"], "budget.restarts"), "budget.restarts")
    mode_val = _require(["mode"], "mode")
    if not isinstance(mode_val, str):
        raise ValueError("config_used.yaml sample.mode must be a string.")
    bidirectional_val = _require(["objective", "bidirectional"], "objective.bidirectional")
    if not isinstance(bidirectional_val, bool):
        raise ValueError("config_used.yaml sample.objective.bidirectional must be a boolean.")
    bidirectional = bidirectional_val
    top_k = _coerce_int(_require(["elites", "k"], "elites.k"), "elites.k")
    pwm_sum_threshold = _coerce_float(
        _require(["elites", "filters", "pwm_sum_min"], "elites.filters.pwm_sum_min"),
        "elites.filters.pwm_sum_min",
    )

    moves_payload = _require(["moves"], "moves")
    moves_cfg = SampleMovesConfig.model_validate(moves_payload)
    move_probs = resolve_move_config(moves_cfg).move_probs

    if optimizer_kind not in {"gibbs", "pt"}:
        raise ValueError("config_used.yaml sample.optimizer.name must be 'gibbs' or 'pt'.")
    if optimizer_kind == "gibbs":
        chains = restarts
        cooling_path = ["optimizers", "gibbs", "beta_schedule", "kind"]
        cooling_kind = str(_require(cooling_path, "optimizers.gibbs.beta_schedule.kind"))
    else:
        ladder = _require(["optimizers", "pt", "beta_ladder"], "optimizers.pt.beta_ladder")
        if not isinstance(ladder, dict):
            raise ValueError("config_used.yaml sample.optimizers.pt.beta_ladder must be a mapping.")
        cooling_kind = str(ladder.get("kind") or "")
        if cooling_kind == "fixed":
            chains = 1
        else:
            betas = ladder.get("betas")
            n_temps = ladder.get("n_temps")
            if isinstance(betas, list) and betas:
                chains = len(betas)
            elif isinstance(n_temps, int):
                chains = int(n_temps)
            else:
                raise ValueError("config_used.yaml sample.optimizers.pt.beta_ladder must define betas or n_temps.")

    return SampleMeta(
        optimizer_kind=optimizer_kind,
        chains=chains,
        draws=draws,
        tune=tune,
        move_probs=move_probs,
        cooling_kind=cooling_kind,
        pwm_sum_threshold=pwm_sum_threshold,
        bidirectional=bidirectional,
        top_k=top_k,
        mode=mode_val,
    )


def _resolve_scoring_params(used_cfg: dict) -> tuple[float, float | None]:
    if not isinstance(used_cfg, dict):
        raise ValueError("config_used.yaml is missing sample config; re-run `cruncher sample`.")
    sample = used_cfg.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("config_used.yaml missing sample section; re-run `cruncher sample`.")
    objective = sample.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("config_used.yaml missing sample.objective; re-run `cruncher sample`.")
    scoring = objective.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("config_used.yaml missing sample.objective.scoring; re-run `cruncher sample`.")
    pseudocounts = scoring.get("pwm_pseudocounts")
    if not isinstance(pseudocounts, (int, float)):
        raise ValueError("config_used.yaml sample.objective.scoring.pwm_pseudocounts must be a number.")
    log_odds_clip = scoring.get("log_odds_clip")
    if log_odds_clip is not None and not isinstance(log_odds_clip, (int, float)):
        raise ValueError("config_used.yaml sample.objective.scoring.log_odds_clip must be a number or null.")
    return float(pseudocounts), float(log_odds_clip) if log_odds_clip is not None else None


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


def _resolve_git_dir(path: Path) -> Path | None:
    git_path = path / ".git"
    if not git_path.exists():
        return None
    if git_path.is_dir():
        return git_path
    if git_path.is_file():
        try:
            payload = git_path.read_text().strip()
        except OSError:
            return None
        if payload.startswith("gitdir:"):
            git_dir = payload.split(":", 1)[1].strip()
            resolved = Path(git_dir)
            if not resolved.is_absolute():
                resolved = (git_path.parent / resolved).resolve()
            if resolved.exists():
                return resolved
    return None


def _get_git_commit(path: Path) -> str | None:
    probe = path.resolve()
    for _ in range(6):
        git_dir = _resolve_git_dir(probe)
        if git_dir is not None:
            try:
                head = (git_dir / "HEAD").read_text().strip()
            except OSError:
                return None
            if head.startswith("ref:"):
                ref = head.split(" ", 1)[1].strip()
                ref_path = git_dir / ref
                if ref_path.exists():
                    try:
                        return ref_path.read_text().strip()
                    except OSError:
                        return None
            return head or None
        if probe.parent == probe:
            break
        probe = probe.parent
    return None


def _resolve_tf_pair(
    analysis_cfg: AnalysisConfig,
    tf_names: list[str],
    tf_pair_override: tuple[str, str] | None = None,
) -> tuple[str, str] | None:
    pair = tf_pair_override
    if pair is None:
        pair = analysis_cfg.tf_pair
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
    plot_keys_override: list[str] | None = None,
    scatter_background_override: bool | None = None,
    scatter_background_samples_override: int | None = None,
    scatter_background_seed_override: int | None = None,
) -> list[Path]:
    """Iterate over runs listed in cfg.analysis.runs and generate diagnostics."""
    # Shared PWM store
    if cfg.analysis is None:
        raise ValueError("analysis section is required for analyze")
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")

    ensure_mpl_cache(resolve_catalog_root(config_path, cfg.motif_store.catalog_root))
    import arviz as az
    import yaml

    from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
    from dnadesign.cruncher.analysis.per_pwm import gather_per_pwm_scores
    from dnadesign.cruncher.analysis.plots.diagnostics import (
        make_pair_idata,
        plot_autocorr,
        plot_ess,
        plot_pair_pwm_scores,
        plot_parallel_pwm_scores,
        plot_rank_diagnostic,
        plot_trace,
        report_convergence,
    )
    from dnadesign.cruncher.analysis.plots.scatter import plot_scatter
    from dnadesign.cruncher.analysis.plots.summary import (
        plot_correlation_heatmap,
        plot_parallel_coords,
        plot_score_box,
        plot_score_hist,
        plot_score_pairgrid,
        score_frame_from_df,
        write_elite_topk,
        write_joint_metrics,
        write_score_summary,
    )

    analysis_cfg = cfg.analysis
    plot_keys = {spec.key for spec in PLOT_SPECS}
    override_payload: dict[str, object] = {}
    if plot_keys_override:
        requested = [key for key in plot_keys_override if key]
        if "all" in requested and len(requested) > 1:
            raise ValueError("Use either --plots all or explicit plot keys, not both.")
        unknown = [key for key in requested if key != "all" and key not in plot_keys]
        if unknown:
            raise ValueError(f"Unknown plot keys: {', '.join(unknown)}")
        analysis_cfg = analysis_cfg.model_copy(deep=True)
        for key in plot_keys:
            setattr(analysis_cfg.plots, key, False)
        if "all" in requested:
            for key in plot_keys:
                setattr(analysis_cfg.plots, key, True)
        else:
            for key in requested:
                setattr(analysis_cfg.plots, key, True)
        override_payload["plots"] = requested

    if scatter_background_override is not None or scatter_background_samples_override is not None:
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        if scatter_background_override is not None:
            analysis_cfg.scatter_background = scatter_background_override
            override_payload["scatter_background"] = scatter_background_override
        if scatter_background_samples_override is not None:
            if scatter_background_samples_override < 0:
                raise ValueError("--scatter-background-samples must be >= 0.")
            analysis_cfg.scatter_background_samples = scatter_background_samples_override
            override_payload["scatter_background_samples"] = scatter_background_samples_override
    if scatter_background_seed_override is not None:
        if scatter_background_seed_override < 0:
            raise ValueError("--scatter-background-seed must be >= 0.")
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        analysis_cfg.scatter_background_seed = scatter_background_seed_override
        override_payload["scatter_background_seed"] = scatter_background_seed_override

    override_payload = override_payload or None

    cfg_effective = cfg
    if analysis_cfg is not cfg.analysis:
        cfg_effective = cfg.model_copy(deep=True)
        cfg_effective.analysis = analysis_cfg

    runs = runs_override if runs_override else (analysis_cfg.runs or [])
    if not runs:
        if not use_latest:
            raise ValueError("No analysis runs configured. Set analysis.runs, pass --run, or use --latest.")
        latest = list_runs(cfg, config_path, stage="sample")
        if not latest:
            raise ValueError("No sample runs found. Run `cruncher sample <config>` first.")
        runs = [latest[0].name]
        logger.info("Analyzing latest sample run: %s", runs[0])
    analysis_runs: list[Path] = []
    from dnadesign.cruncher.app.run_service import get_run

    for run_name in runs:
        run_info = get_run(cfg, config_path, run_name)
        sample_dir = run_info.run_dir

        if run_info.stage != "sample":
            raise ValueError(f"Run '{run_name}' is not a sample run (stage={run_info.stage})")

        logger.info("Analyzing run: %s", run_info.name)
        if run_name != run_info.name:
            logger.debug("Run path override: %s", run_name)
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
        sample_meta = _resolve_sample_meta(cfg, used_cfg)
        scoring_pseudocounts, scoring_log_odds_clip = _resolve_scoring_params(used_cfg)
        bidirectional = sample_meta.bidirectional

        # ── locate elites parquet file ──────────────────────────────────────
        from dnadesign.cruncher.analysis.elites import find_elites_parquet

        elites_path = find_elites_parquet(sample_dir)
        elites_df = read_parquet(elites_path)
        logger.info(
            "Loaded %d elites from %s",
            len(elites_df),
            elites_path.relative_to(sample_dir),
        )

        # ── load trace & sequences for downstream plots ───────────────────────
        seq_path, trace_file = sequences_path(sample_dir), trace_path(sample_dir)
        if not seq_path.exists():
            raise FileNotFoundError(
                f"Missing artifacts/sequences.parquet in {sample_dir}. "
                "Re-run `cruncher sample` with sample.output.save_sequences=true."
            )
        plots = analysis_cfg.plots
        needs_trace = plots.trace or plots.autocorr or plots.convergence
        trace_idata = None
        if trace_file.exists():
            trace_idata = az.from_netcdf(trace_file)
        elif needs_trace:
            msg = (
                f"Missing artifacts/trace.nc in {sample_dir}. "
                "Re-run `cruncher sample` with sample.output.trace.save=true."
            )
            raise FileNotFoundError(msg)

        analysis_root = sample_dir / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        analysis_signature, signature_payload = _analysis_signature(
            analysis_cfg=analysis_cfg,
            override_payload=override_payload,
            config_used_file=config_used_path(sample_dir),
            sequences_file=seq_path,
            elites_file=elites_path,
            trace_file=trace_file,
        )
        existing_summary = _load_summary_payload(analysis_root)
        if existing_summary and existing_summary.get("signature") == analysis_signature:
            logger.info("Analysis already up to date: %s", analysis_root)
            analysis_runs.append(analysis_root)
            continue
        analysis_id = _analysis_id()
        existing_items = [path for path in _analysis_item_paths(analysis_root) if path.exists()]
        existing_id = _load_summary_id(analysis_root)
        if existing_items and existing_id is None:
            raise ValueError(
                "analysis/ contains artifacts but meta/summary.json is missing. "
                "Remove analysis/ or restore meta/summary.json before re-running analyze."
            )
        if existing_id:
            if analysis_cfg.archive:
                _archive_existing_analysis(analysis_root, manifest, existing_id)
            else:
                _clear_latest_analysis(analysis_root)
                _prune_latest_analysis_artifacts(manifest)
        analysis_meta = analysis_meta_root(analysis_root)
        analysis_meta.mkdir(parents=True, exist_ok=True)
        plots_dir = analysis_root / "plots"
        tables_dir = analysis_root / "tables"
        notebooks_dir = analysis_root / "notebooks"
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        notebooks_dir.mkdir(parents=True, exist_ok=True)

        analysis_used_file = analysis_used_path(analysis_root)
        analysis_used_payload = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "layout_version": ANALYSIS_LAYOUT_VERSION,
            "analysis": analysis_cfg.model_dump(),
        }
        if override_payload:
            analysis_used_payload["analysis_overrides"] = override_payload
            analysis_used_payload["analysis_base"] = cfg.analysis.model_dump()
        analysis_used_file.parent.mkdir(parents=True, exist_ok=True)
        analysis_used_file.write_text(yaml.safe_dump(analysis_used_payload, sort_keys=False))

        # gather per-PWM scores (for scatter etc.)
        per_pwm_path = tables_dir / "gathered_per_pwm_everyN.csv"
        gather_per_pwm_scores(
            sample_dir,
            analysis_cfg.subsampling_epsilon,
            pwms,
            bidirectional=bidirectional,
            scale=analysis_cfg.scatter_scale,
            out_path=per_pwm_path,
            pseudocounts=scoring_pseudocounts,
            log_odds_clip=scoring_log_odds_clip,
        )
        logger.info("Wrote per-PWM score table → %s", per_pwm_path.relative_to(sample_dir))

        seq_df = read_parquet(seq_path)
        score_df = score_frame_from_df(seq_df, tf_names)
        score_summary_path = tables_dir / "score_summary.csv"
        write_score_summary(score_df, tf_names, score_summary_path)

        topk_path = tables_dir / "elite_topk.csv"
        write_elite_topk(elites_df, tf_names, topk_path, top_k=sample_meta.top_k)

        joint_metrics_path = tables_dir / "joint_metrics.csv"
        write_joint_metrics(elites_df, tf_names, joint_metrics_path)

        sample_meta_payload = {
            "mode": sample_meta.mode,
            "optimizer_kind": sample_meta.optimizer_kind,
            "chains": sample_meta.chains,
            "draws": sample_meta.draws,
            "tune": sample_meta.tune,
            "top_k": sample_meta.top_k,
            "pwm_sum_threshold": sample_meta.pwm_sum_threshold,
            "bidirectional": sample_meta.bidirectional,
            "move_probs": sample_meta.move_probs,
            "cooling_kind": sample_meta.cooling_kind,
        }
        diagnostics_summary = summarize_sampling_diagnostics(
            trace_idata=trace_idata,
            sequences_df=seq_df,
            elites_df=elites_df,
            tf_names=tf_names,
            optimizer=manifest.get("optimizer", {}),
            optimizer_stats=manifest.get("optimizer_stats", {}),
            mode=sample_meta.mode,
            optimizer_kind=sample_meta.optimizer_kind,
            sample_meta=sample_meta_payload,
        )
        diagnostics_path = tables_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics_summary, indent=2))
        if diagnostics_summary.get("warnings"):
            logger.warning(
                "Diagnostics warnings detected (%d). See %s.",
                len(diagnostics_summary["warnings"]),
                diagnostics_path.relative_to(sample_dir),
            )

        auto_opt_table_path = None
        auto_opt_plot_path = None
        auto_opt_payload = manifest.get("auto_opt")
        if isinstance(auto_opt_payload, dict):
            candidates = auto_opt_payload.get("candidates")
            if isinstance(candidates, list) and candidates:
                auto_opt_table_path = tables_dir / "auto_opt_pilots.csv"
                pd.DataFrame(candidates).to_csv(auto_opt_table_path, index=False)
                required_cols = {"best_score", "balance_median"}
                if required_cols.issubset({col for col in pd.DataFrame(candidates).columns}):
                    df_auto = pd.DataFrame(candidates)
                    df_auto = df_auto.dropna(subset=["best_score", "balance_median"])
                    if not df_auto.empty:
                        import matplotlib.pyplot as plt

                        auto_opt_plot_path = plots_dir / "auto_opt_tradeoffs.png"
                        fig, ax = plt.subplots(figsize=(6, 4))
                        lengths = df_auto.get("length")
                        colors = None
                        if lengths is not None and lengths.notna().any():
                            colors = lengths.astype(float)
                        scatter = ax.scatter(
                            df_auto["balance_median"],
                            df_auto["best_score"],
                            c=colors,
                            cmap="viridis" if colors is not None else None,
                            s=50,
                            alpha=0.85,
                        )
                        ax.set_xlabel("Balance (median min normalized)")
                        ax.set_ylabel("Best score")
                        ax.set_title("Auto-opt tradeoffs")
                        if colors is not None:
                            fig.colorbar(scatter, ax=ax, label="Length")
                        fig.tight_layout()
                        fig.savefig(auto_opt_plot_path, dpi=150)
                        plt.close(fig)

        enabled_specs = [spec for spec in PLOT_SPECS if getattr(plots, spec.key, False)]
        if enabled_specs:
            logger.debug("Enabled plots: %s", ", ".join(spec.key for spec in enabled_specs))
        tf_pair = _resolve_tf_pair(analysis_cfg, tf_names, tf_pair_override=tf_pair_override)
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
            idata_pair = make_pair_idata(sample_dir, tf_pair=tf_pair)
            if plots.pair_pwm:
                plot_pair_pwm_scores(idata_pair, plots_dir, tf_pair=tf_pair)
            if plots.parallel_pwm:
                plot_parallel_pwm_scores(idata_pair, plots_dir, tf_pair=tf_pair)

        # ── scatter plot ──────────────────────────────────────────────────────
        if plots.scatter_pwm:
            if not per_pwm_path.exists():
                raise FileNotFoundError(f"Missing gathered_per_pwm_everyN.csv in {tables_dir}")
            annotation = (
                f"chains = {sample_meta.chains}\n"
                f"iters  = {sample_meta.tune + sample_meta.draws}\n"
                f"S/B/M   = {sample_meta.move_probs['S']:.2f}/"
                f"{sample_meta.move_probs['B']:.2f}/"
                f"{sample_meta.move_probs['M']:.2f}\n"
                f"cooling = {sample_meta.cooling_kind}"
            )
            if any(sample_meta.move_probs.get(k, 0.0) > 0 for k in ("L", "W", "I")):
                annotation = (
                    annotation
                    + "\n"
                    + f"L/W/I   = {sample_meta.move_probs.get('L', 0.0):.2f}/"
                    + f"{sample_meta.move_probs.get('W', 0.0):.2f}/"
                    + f"{sample_meta.move_probs.get('I', 0.0):.2f}"
                )
            plot_scatter(
                sample_dir,
                pwms,
                cfg_effective,
                tf_pair=tf_pair,
                per_pwm_path=per_pwm_path,
                out_dir=plots_dir,
                bidirectional=bidirectional,
                pwm_sum_threshold=sample_meta.pwm_sum_threshold,
                annotation=annotation,
                pseudocounts=scoring_pseudocounts,
                log_odds_clip=scoring_log_odds_clip,
            )
            logger.debug("Wrote pwm__scatter.pdf")

        if plots.score_hist:
            plot_score_hist(score_df, tf_names, plots_dir / "score__hist.png")
        if plots.score_box:
            plot_score_box(score_df, tf_names, plots_dir / "score__box.png")
        if plots.correlation_heatmap:
            plot_correlation_heatmap(score_df, tf_names, plots_dir / "score__correlation.png")
        if plots.pairgrid:
            plot_score_pairgrid(score_df, tf_names, plots_dir / "score__pairgrid.png")
        if plots.parallel_coords:
            if elites_df.empty:
                logger.warning("Skipping parallel coordinates: no elites available.")
            else:
                plot_parallel_coords(elites_df, tf_names, plots_dir / "elites__parallel_coords.png")

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
                logger.debug("Top-5 elites (%s & %s):", x_tf, y_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.debug(
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
                logger.debug("Top-5 elites (%s):", x_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.debug(
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
                logger.debug("Top-5 elites (all TFs):")
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    score_blob = " ".join(f"{tf}={row[f'score_{tf}']:.1f}" for tf in tf_names)
                    logger.debug(
                        "rank=%d seq=%s %s",
                        int(row["rank"]),
                        row["sequence"],
                        score_blob,
                    )
            else:
                logger.warning("No regulators configured; skipping elite summary.")

        plot_manifest = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plots": [],
        }
        plot_artifacts: list[dict[str, object]] = []
        for spec in PLOT_SPECS:
            enabled_flag = getattr(plots, spec.key, False)
            outputs = []
            missing = []
            for output in spec.outputs:
                out_path = analysis_root / output
                exists = out_path.exists()
                outputs.append({"path": output, "exists": exists})
                if enabled_flag and not exists:
                    missing.append(output)
                elif enabled_flag:
                    kind = "plot" if out_path.suffix.lower() in {".png", ".pdf"} else "text"
                    plot_artifacts.append(
                        artifact_entry(
                            out_path,
                            sample_dir,
                            kind=kind,
                            label=f"{spec.label} ({out_path.name})",
                            stage="analysis",
                        )
                    )
            plot_manifest["plots"].append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "group": spec.group,
                    "description": spec.description,
                    "requires": list(spec.requires),
                    "enabled": enabled_flag,
                    "outputs": outputs,
                    "missing_outputs": missing,
                    "generated": enabled_flag and any(out["exists"] for out in outputs),
                }
            )
        if auto_opt_plot_path is not None:
            plot_manifest["plots"].append(
                {
                    "key": "auto_opt_tradeoffs",
                    "label": "Auto-opt tradeoffs (PNG)",
                    "group": "auto_opt",
                    "description": "Balance vs best score across auto-opt pilots.",
                    "requires": [],
                    "enabled": True,
                    "outputs": [
                        {"path": str(auto_opt_plot_path.relative_to(sample_dir)), "exists": auto_opt_plot_path.exists()}
                    ],
                    "missing_outputs": [] if auto_opt_plot_path.exists() else ["plots/auto_opt_tradeoffs.png"],
                    "generated": auto_opt_plot_path.exists(),
                }
            )
            plot_artifacts.append(
                artifact_entry(
                    auto_opt_plot_path,
                    sample_dir,
                    kind="plot",
                    label="Auto-opt tradeoffs (PNG)",
                    stage="analysis",
                )
            )

        plot_manifest_file = plot_manifest_path(analysis_root)
        plot_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        plot_manifest_file.write_text(json.dumps(plot_manifest, indent=2))

        table_manifest = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tables": [
                {
                    "key": "per_pwm",
                    "label": "Per-PWM scores (CSV)",
                    "path": str(per_pwm_path.relative_to(sample_dir)),
                    "exists": per_pwm_path.exists(),
                },
                {
                    "key": "score_summary",
                    "label": "Per-TF summary (CSV)",
                    "path": str(score_summary_path.relative_to(sample_dir)),
                    "exists": score_summary_path.exists(),
                },
                {
                    "key": "elite_topk",
                    "label": "Elite top-K (CSV)",
                    "path": str(topk_path.relative_to(sample_dir)),
                    "exists": topk_path.exists(),
                },
                {
                    "key": "joint_metrics",
                    "label": "Joint score metrics (CSV)",
                    "path": str(joint_metrics_path.relative_to(sample_dir)),
                    "exists": joint_metrics_path.exists(),
                },
                {
                    "key": "diagnostics",
                    "label": "Diagnostics summary (JSON)",
                    "path": str(diagnostics_path.relative_to(sample_dir)),
                    "exists": diagnostics_path.exists(),
                },
            ],
        }
        if auto_opt_table_path is not None:
            table_manifest["tables"].append(
                {
                    "key": "auto_opt_pilots",
                    "label": "Auto-opt pilot scorecard (CSV)",
                    "path": str(auto_opt_table_path.relative_to(sample_dir)),
                    "exists": auto_opt_table_path.exists(),
                }
            )
        table_manifest_file = table_manifest_path(analysis_root)
        table_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        table_manifest_file.write_text(json.dumps(table_manifest, indent=2))

        artifacts: list[dict[str, object]] = [
            artifact_entry(
                analysis_used_file,
                sample_dir,
                kind="config",
                label="Analysis settings",
                stage="analysis",
            ),
            artifact_entry(
                per_pwm_path,
                sample_dir,
                kind="table",
                label="Per-PWM scores (CSV)",
                stage="analysis",
            ),
            artifact_entry(
                score_summary_path,
                sample_dir,
                kind="table",
                label="Per-TF summary (CSV)",
                stage="analysis",
            ),
            artifact_entry(
                topk_path,
                sample_dir,
                kind="table",
                label="Elite top-K (CSV)",
                stage="analysis",
            ),
            artifact_entry(
                joint_metrics_path,
                sample_dir,
                kind="table",
                label="Joint score metrics (CSV)",
                stage="analysis",
            ),
            artifact_entry(
                diagnostics_path,
                sample_dir,
                kind="json",
                label="Diagnostics summary (JSON)",
                stage="analysis",
            ),
            *(
                [
                    artifact_entry(
                        auto_opt_table_path,
                        sample_dir,
                        kind="table",
                        label="Auto-opt pilot scorecard (CSV)",
                        stage="analysis",
                    )
                ]
                if auto_opt_table_path is not None
                else []
            ),
            *(
                [
                    artifact_entry(
                        auto_opt_plot_path,
                        sample_dir,
                        kind="plot",
                        label="Auto-opt tradeoffs (PNG)",
                        stage="analysis",
                    )
                ]
                if auto_opt_plot_path is not None
                else []
            ),
            artifact_entry(
                plot_manifest_file,
                sample_dir,
                kind="json",
                label="Plot manifest",
                stage="analysis",
            ),
            artifact_entry(
                table_manifest_file,
                sample_dir,
                kind="json",
                label="Table manifest",
                stage="analysis",
            ),
        ]
        artifacts.extend(plot_artifacts)

        inputs_payload = {
            "sequences.parquet": {
                "path": str(seq_path.relative_to(sample_dir)),
                "sha256": sha256_path(seq_path),
            },
            "elites.parquet": {
                "path": str(elites_path.relative_to(sample_dir)),
                "sha256": sha256_path(elites_path),
            },
            "config_used.yaml": {
                "path": str(config_used_path(sample_dir).relative_to(sample_dir)),
                "sha256": sha256_path(config_used_path(sample_dir)),
            },
        }
        if trace_file.exists():
            inputs_payload["trace.nc"] = {
                "path": str(trace_file.relative_to(sample_dir)),
                "sha256": sha256_path(trace_file),
            }
        summary_payload = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run": run_name,
            "run_dir": str(sample_dir.resolve()),
            "analysis_dir": str(analysis_root.resolve()),
            "tf_names": tf_names,
            "diagnostics": diagnostics_summary,
            "analysis_layout_version": ANALYSIS_LAYOUT_VERSION,
            "analysis_config": analysis_cfg.model_dump(),
            "cruncher_version": _get_version(),
            "git_commit": _get_git_commit(config_path),
            "analysis_used": str(analysis_used_file.relative_to(sample_dir)),
            "config_used": str(config_used_path(sample_dir).relative_to(sample_dir)),
            "plot_manifest": str(plot_manifest_file.relative_to(sample_dir)),
            "table_manifest": str(table_manifest_file.relative_to(sample_dir)),
            "inputs": inputs_payload,
            "signature": analysis_signature,
            "signature_inputs": signature_payload,
            "artifacts": [item["path"] for item in artifacts],
        }
        summary_file = summary_path(analysis_root)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary_payload, indent=2))
        artifacts.append(
            artifact_entry(
                summary_file,
                sample_dir,
                kind="json",
                label="Analysis summary",
                stage="analysis",
            )
        )

        append_artifacts(manifest, artifacts)
        analysis_root.mkdir(parents=True, exist_ok=True)

        from dnadesign.cruncher.app.run_service import (
            update_run_index_from_manifest,
        )
        from dnadesign.cruncher.artifacts.manifest import write_manifest

        write_manifest(sample_dir, manifest)
        update_run_index_from_manifest(
            config_path,
            sample_dir,
            manifest,
            catalog_root=cfg.motif_store.catalog_root,
        )
        logger.info("Analysis artifacts recorded (%s).", analysis_id)
        analysis_runs.append(analysis_root)
    return analysis_runs
