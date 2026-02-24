"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/mmr_sweep_service.py

Service entrypoints to replay MMR sweep diagnostics from a sample run directory.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

from dnadesign.cruncher.analysis.layout import analysis_root, analysis_table_path
from dnadesign.cruncher.analysis.mmr_sweep import run_mmr_sweep
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.app.analyze.metadata import load_pwms_from_config
from dnadesign.cruncher.artifacts.layout import elites_path, elites_yaml_path, sequences_path
from dnadesign.cruncher.artifacts.manifest import load_manifest


def _resolve_tf_names(used_cfg: dict[str, object], pwms: dict[str, object]) -> list[str]:
    active = used_cfg.get("active_regulator_set") if isinstance(used_cfg, dict) else None
    if isinstance(active, dict):
        tfs = active.get("tfs")
        if isinstance(tfs, list) and tfs:
            return [str(tf) for tf in tfs if str(tf)]
    return sorted(pwms.keys())


def _load_elites_meta(run_dir: Path) -> dict[str, object]:
    path = elites_yaml_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing elites metadata YAML: {path}")
    try:
        payload = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid elites metadata YAML at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Elites metadata must contain a YAML mapping: {path}")
    return payload


def _resolve_objective_payload(manifest: dict[str, object], used_cfg: dict[str, object]) -> dict[str, object]:
    objective = manifest.get("objective")
    if objective is None:
        objective = {}
    if not isinstance(objective, dict):
        raise ValueError("Run manifest field 'objective' must be an object when provided.")

    total_sweeps_raw = manifest.get("total_sweeps")
    if total_sweeps_raw is None:
        sample_payload = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
        if not isinstance(sample_payload, dict):
            raise ValueError("config_used.yaml missing sample settings required to infer total_sweeps.")
        budget_payload = sample_payload.get("budget")
        if not isinstance(budget_payload, dict):
            raise ValueError("config_used.yaml sample.budget missing while inferring total_sweeps.")
        tune = budget_payload.get("tune")
        draws = budget_payload.get("draws")
        if not isinstance(tune, int) or not isinstance(draws, int):
            raise ValueError("config_used.yaml sample.budget tune/draws must be integers.")
        total_sweeps_raw = int(tune) + int(draws)
    try:
        total_sweeps = int(total_sweeps_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Run manifest field 'total_sweeps' must be an integer.") from exc
    if total_sweeps < 1:
        raise ValueError("Run manifest field 'total_sweeps' must be >= 1.")
    return {**objective, "total_sweeps": total_sweeps}


def _resolve_scoring(sample_payload: dict[str, object] | None) -> tuple[float, float | None]:
    default_pseudocounts = 0.10
    default_clip = None
    if not isinstance(sample_payload, dict):
        return default_pseudocounts, default_clip
    objective_payload = sample_payload.get("objective")
    if not isinstance(objective_payload, dict):
        return default_pseudocounts, default_clip
    scoring_payload = objective_payload.get("scoring")
    if not isinstance(scoring_payload, dict):
        return default_pseudocounts, default_clip

    pseudocounts_raw = scoring_payload.get("pwm_pseudocounts")
    if pseudocounts_raw is not None:
        default_pseudocounts = float(pseudocounts_raw)
    clip_raw = scoring_payload.get("log_odds_clip")
    if clip_raw is not None:
        default_clip = float(clip_raw)
    return default_pseudocounts, default_clip


def _resolve_baseline_pool_size(
    sample_payload: dict[str, object] | None,
    elites_meta: dict[str, object],
) -> str | int | None:
    if isinstance(sample_payload, dict):
        elites_payload = sample_payload.get("elites")
        if isinstance(elites_payload, dict):
            select_payload = elites_payload.get("select")
            if isinstance(select_payload, dict):
                pool_size = select_payload.get("pool_size")
                if pool_size is not None:
                    return pool_size
    return elites_meta.get("pool_size")


def compute_mmr_sweep_for_run(
    run_dir: Path,
    *,
    pool_size_values: Sequence[str | int],
    diversity_values: Sequence[float],
) -> pd.DataFrame:
    sequences_file = sequences_path(run_dir)
    elites_file = elites_path(run_dir)
    if not sequences_file.exists():
        raise FileNotFoundError(f"Missing sequences parquet for MMR sweep replay: {sequences_file}")
    if not elites_file.exists():
        raise FileNotFoundError(f"Missing elites parquet for MMR sweep replay: {elites_file}")

    sequences_df = read_parquet(sequences_file)
    elites_df = read_parquet(elites_file)
    manifest = load_manifest(run_dir)
    pwms, used_cfg = load_pwms_from_config(run_dir)
    tf_names = _resolve_tf_names(used_cfg, pwms)
    if not tf_names:
        raise ValueError("MMR sweep replay requires at least one TF in config_used.yaml.")

    sample_payload = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
    objective_payload = _resolve_objective_payload(manifest, used_cfg)
    pseudocounts, log_odds_clip = _resolve_scoring(sample_payload if isinstance(sample_payload, dict) else None)
    elites_meta = _load_elites_meta(run_dir)
    baseline_pool_size = _resolve_baseline_pool_size(
        sample_payload if isinstance(sample_payload, dict) else None,
        elites_meta,
    )

    baseline_diversity_raw = elites_meta.get("selection_diversity")
    try:
        baseline_diversity = float(baseline_diversity_raw) if baseline_diversity_raw is not None else None
    except (TypeError, ValueError, OverflowError):
        baseline_diversity = None

    bidirectional = bool(objective_payload.get("bidirectional", True))
    elite_k_raw = manifest.get("top_k")
    elite_k = int(elite_k_raw) if elite_k_raw is not None else int(len(elites_df))
    if elite_k < 1:
        raise ValueError("Run manifest top_k must be >= 1 for MMR sweep replay.")

    return run_mmr_sweep(
        sequences_df=sequences_df,
        elites_df=elites_df,
        tf_names=tf_names,
        pwms=pwms,
        objective_config=objective_payload,
        bidirectional=bidirectional,
        elite_k=elite_k,
        pwm_pseudocounts=float(pseudocounts),
        log_odds_clip=log_odds_clip,
        pool_size_values=pool_size_values,
        diversity_values=diversity_values,
        baseline_pool_size=baseline_pool_size,
        baseline_diversity=baseline_diversity,
    )


def run_mmr_sweep_for_run(
    run_dir: Path,
    *,
    pool_size_values: Sequence[str | int],
    diversity_values: Sequence[float],
    out_path: Path | None = None,
) -> Path:
    sweep_df = compute_mmr_sweep_for_run(
        run_dir,
        pool_size_values=pool_size_values,
        diversity_values=diversity_values,
    )
    if out_path is None:
        out_path = analysis_table_path(analysis_root(run_dir), "elites_mmr_sweep", "parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        write_parquet(sweep_df, out_path)
    else:
        sweep_df.to_csv(out_path, index=False)
    return out_path
