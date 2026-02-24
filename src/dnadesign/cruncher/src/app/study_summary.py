"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/study_summary.py

Aggregate Study trial outputs into summary tables, manifests, and plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.cruncher.analysis.layout import analysis_root, analysis_table_path
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.study.layout import (
    spec_frozen_path,
    study_manifest_path,
    study_manifests_dir,
    study_meta_dir,
    study_plot_path,
    study_status_path,
    study_table_path,
)
from dnadesign.cruncher.study.manifest import (
    StudyManifestV1,
    StudyTrialRun,
    load_study_manifest,
    load_study_status,
    utc_now_iso,
    write_study_status,
)
from dnadesign.cruncher.study.metrics import extract_elite_metrics
from dnadesign.cruncher.study.schema_models import StudySpec
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

__all__ = ["StudySummaryResult", "summarize_study_run"]


@dataclass(frozen=True)
class StudySummaryResult:
    study_run_dir: Path
    n_missing_total: int
    n_missing_non_success: int
    n_missing_run_dirs: int
    n_missing_metric_artifacts: int
    n_missing_mmr_tables: int
    exit_code_policy: str


@dataclass(frozen=True)
class _TrialMetricCollection:
    rows: list[dict[str, object]]
    summarized_keys: set[tuple[str, int, int]]
    n_missing_run_dirs: int
    n_missing_metric_artifacts: int


@dataclass(frozen=True)
class _SummaryMissingCounts:
    n_missing_total: int
    n_missing_non_success: int


@dataclass(frozen=True)
class _BasePlotValues:
    sequence_length: int
    diversity: float


def _write_study_manifests(
    study_run_dir: Path,
    *,
    table_entries: list[dict[str, object]],
    plot_entries: list[dict[str, object]],
    study_id: str,
    study_name: str,
) -> None:
    manifest_dir = study_manifests_dir(study_run_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    table_manifest_file = manifest_dir / "table_manifest.json"
    plot_manifest_file = manifest_dir / "plot_manifest.json"
    table_manifest = {"study_id": study_id, "study_name": study_name, "tables": table_entries}
    plot_manifest = {"study_id": study_id, "study_name": study_name, "plots": plot_entries}
    atomic_write_json(table_manifest_file, table_manifest, allow_nan=False)
    atomic_write_json(plot_manifest_file, plot_manifest, allow_nan=False)
    atomic_write_json(
        manifest_dir / "manifest.json",
        {
            "study_id": study_id,
            "study_name": study_name,
            "table_manifest": table_manifest_file.name,
            "plot_manifest": plot_manifest_file.name,
        },
        allow_nan=False,
    )


def _frozen_study_spec(study_run_dir: Path) -> StudySpec:
    spec_file = spec_frozen_path(study_run_dir)
    if not spec_file.exists():
        raise FileNotFoundError(f"Missing frozen study spec: {spec_file}")
    payload = yaml.safe_load(spec_file.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen study spec must be a mapping: {spec_file}")
    study_payload = payload.get("study")
    if not isinstance(study_payload, dict):
        raise ValueError(f"Frozen study spec missing 'study' mapping: {spec_file}")
    if "execution" not in study_payload:
        raise ValueError(f"Frozen study spec missing required key: execution ({spec_file})")
    if "replays" not in study_payload:
        raise ValueError(f"Frozen study spec missing required key: replays ({spec_file})")
    try:
        return StudySpec.model_validate(study_payload)
    except Exception as exc:
        raise ValueError(f"Frozen study spec failed schema validation: {spec_file}") from exc


def _series_label(row: StudyTrialRun) -> str:
    seq_len = row.factor_columns.get("param__sample__sequence_length")
    label_parts = [f"Regulator set {int(row.target_set_index)}", f"Trial {row.trial_id}"]
    if seq_len is not None:
        label_parts.append(f"Length {seq_len}")
    return ", ".join(label_parts)


def _resolve_base_plot_values(spec: StudySpec) -> _BasePlotValues:
    cfg = load_config(Path(spec.base_config))
    if cfg.sample is None:
        raise ValueError("Study summarize requires sample section in base_config.")
    return _BasePlotValues(
        sequence_length=int(cfg.sample.sequence_length),
        diversity=float(cfg.sample.elites.select.diversity),
    )


def _aggregate_metric(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[metric].agg(["count", "mean", "std"]).reset_index()
    grouped = grouped.rename(
        columns={
            "count": "n_replicates",
            "mean": f"{metric}_mean",
            "std": f"{metric}_std",
        }
    )
    std_values = pd.to_numeric(grouped[f"{metric}_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    replicate_counts = pd.to_numeric(grouped["n_replicates"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem_values = np.where(replicate_counts > 0.0, std_values / np.sqrt(replicate_counts), 0.0)
    grouped[f"{metric}_sem"] = sem_values.astype(float)
    return grouped


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _resolve_successful_runs(
    manifest: StudyManifestV1,
    *,
    allow_partial: bool,
) -> tuple[list[StudyTrialRun], int]:
    successful_runs = [item for item in manifest.trial_runs if item.status == "success"]
    n_non_success_runs = len(manifest.trial_runs) - len(successful_runs)
    if not allow_partial and len(successful_runs) != len(manifest.trial_runs):
        raise ValueError("study summarize requires all trial runs to be successful unless --allow-partial is set.")
    if not successful_runs:
        raise ValueError("study summarize found no successful trial runs.")
    return successful_runs, n_non_success_runs


def _serialize_trial_rows(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    serialized_rows: list[dict[str, object]] = []
    for row in trial_rows:
        payload = dict(row)
        factors = payload.get("factors")
        factor_columns = payload.get("factor_columns")
        if not isinstance(factors, dict):
            raise ValueError("Study summarize expected trial.factors to be a mapping.")
        if not isinstance(factor_columns, dict):
            raise ValueError("Study summarize expected trial.factor_columns to be a mapping.")
        payload["factors"] = json.dumps(factors, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["factor_columns"] = json.dumps(
            factor_columns,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        serialized_rows.append(payload)
    return serialized_rows


def _collect_trial_metrics(
    *,
    successful_runs: list[StudyTrialRun],
    allow_partial: bool,
) -> _TrialMetricCollection:
    metric_rows: list[dict[str, object]] = []
    summarized_keys: set[tuple[str, int, int]] = set()
    n_missing_run_dirs = 0
    n_missing_metric_artifacts = 0
    for trial_run in successful_runs:
        if trial_run.run_dir is None:
            if allow_partial:
                n_missing_run_dirs += 1
                continue
            raise ValueError(f"Successful trial has no run_dir recorded: {trial_run.trial_id}/seed_{trial_run.seed}")
        run_dir = Path(trial_run.run_dir)
        if not run_dir.exists():
            if allow_partial:
                n_missing_run_dirs += 1
                continue
            raise FileNotFoundError(f"Study trial run directory is missing: {run_dir}")
        try:
            metrics = extract_elite_metrics(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            if allow_partial:
                n_missing_metric_artifacts += 1
                continue
            raise ValueError(
                "Study summarize failed while computing trial metrics for "
                f"{trial_run.trial_id}/seed_{trial_run.seed}/set_{trial_run.target_set_index}: {exc}"
            ) from exc
        row: dict[str, object] = {
            "trial_id": trial_run.trial_id,
            "seed": trial_run.seed,
            "target_set_index": trial_run.target_set_index,
            "target_tfs": ",".join(trial_run.target_tfs),
            "run_dir": str(run_dir),
            "series_label": _series_label(trial_run),
            **metrics,
            **trial_run.factor_columns,
        }
        metric_rows.append(row)
        summarized_keys.add((trial_run.trial_id, int(trial_run.seed), int(trial_run.target_set_index)))
    if not metric_rows:
        raise ValueError("No trial metrics could be computed for summarize.")
    return _TrialMetricCollection(
        rows=metric_rows,
        summarized_keys=summarized_keys,
        n_missing_run_dirs=n_missing_run_dirs,
        n_missing_metric_artifacts=n_missing_metric_artifacts,
    )


def _sequence_length_by_trial(trial_metrics_df: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    return {
        (str(row.trial_id), int(row.seed), int(row.target_set_index)): float(row.sequence_length)
        for row in trial_metrics_df[["trial_id", "seed", "target_set_index", "sequence_length"]].itertuples(index=False)
    }


def _collect_mmr_tables(
    *,
    successful_runs: list[StudyTrialRun],
    summarized_keys: set[tuple[str, int, int]],
    sequence_length_by_run: dict[tuple[str, int, int], float],
    require_mmr: bool,
    allow_partial: bool,
) -> tuple[list[pd.DataFrame], int]:
    mmr_rows: list[pd.DataFrame] = []
    n_missing_mmr_tables = 0
    for trial_run in successful_runs:
        trial_key = (trial_run.trial_id, int(trial_run.seed), int(trial_run.target_set_index))
        if trial_key not in summarized_keys or trial_run.run_dir is None:
            continue
        run_dir = Path(trial_run.run_dir)
        mmr_table = analysis_table_path(analysis_root(run_dir), "elites_mmr_sweep", "parquet")
        if not mmr_table.exists():
            if not require_mmr:
                continue
            if allow_partial:
                n_missing_mmr_tables += 1
                continue
            raise FileNotFoundError(
                f"Missing MMR sweep table for successful trial: {mmr_table}. "
                "Re-run `cruncher study run` with replay enabled or use --allow-partial."
            )
        run_df = read_parquet(mmr_table)
        run_df["trial_id"] = trial_run.trial_id
        run_df["seed"] = trial_run.seed
        run_df["target_set_index"] = trial_run.target_set_index
        run_df["series_label"] = _series_label(trial_run)
        for key, value in trial_run.factor_columns.items():
            run_df[key] = value
        seq_len = sequence_length_by_run.get(trial_key)
        if seq_len is not None and "median_nn_full_distance" in run_df.columns:
            run_df["median_nn_full_bp"] = pd.to_numeric(run_df["median_nn_full_distance"], errors="coerce") * seq_len
        mmr_rows.append(run_df)
    return mmr_rows, n_missing_mmr_tables


def _resolve_missing_counts(
    *,
    allow_partial: bool,
    n_non_success_runs: int,
    n_missing_run_dirs: int,
    n_missing_metric_artifacts: int,
    n_missing_mmr_tables: int,
) -> _SummaryMissingCounts:
    if not allow_partial:
        return _SummaryMissingCounts(n_missing_total=0, n_missing_non_success=0)
    n_missing_non_success = int(n_non_success_runs)
    n_missing_total = int(
        n_missing_non_success + n_missing_run_dirs + n_missing_metric_artifacts + n_missing_mmr_tables
    )
    return _SummaryMissingCounts(
        n_missing_total=n_missing_total,
        n_missing_non_success=n_missing_non_success,
    )


def _aggregate_trial_metrics_for_summary(
    *,
    trial_metrics_df: pd.DataFrame,
    n_missing_total: int,
    n_missing_non_success: int,
    n_missing_run_dirs: int,
    n_missing_metric_artifacts: int,
    n_missing_mmr_tables: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    override_cols = sorted(column for column in trial_metrics_df.columns if column.startswith("param__"))
    group_cols = ["trial_id", "target_set_index", "target_tfs", "series_label", *override_cols]
    metric_cols = [
        "mean_score",
        "median_score",
        "best_score",
        "median_nn_full_bp",
        "min_pairwise_full_bp",
        "mean_pairwise_full_bp",
        "median_nn_full_distance",
        "sequence_length",
    ]
    agg_df = trial_metrics_df[group_cols].drop_duplicates().reset_index(drop=True)
    for metric_idx, metric in enumerate(metric_cols):
        metric_group = _aggregate_metric(trial_metrics_df, group_cols, metric)
        if metric_idx > 0 and "n_replicates" in metric_group.columns:
            metric_group = metric_group.drop(columns=["n_replicates"])
        agg_df = agg_df.merge(metric_group, on=group_cols, how="left")
    agg_df["n_missing_total"] = int(n_missing_total)
    agg_df["n_missing_non_success"] = int(n_missing_non_success)
    agg_df["n_missing_run_dirs"] = int(n_missing_run_dirs)
    agg_df["n_missing_metric_artifacts"] = int(n_missing_metric_artifacts)
    agg_df["n_missing_mmr_tables"] = int(n_missing_mmr_tables)

    length_df = agg_df.copy()
    length_df = length_df.rename(columns={"sequence_length_mean": "sequence_length"})
    if "target_set_index" in length_df.columns:
        length_df["series_label"] = length_df["target_set_index"].map(lambda item: f"Regulator set {int(item)}")
    else:
        length_df["series_label"] = "All targets"
    factor_column = "param__sample__sequence_length"
    if factor_column in length_df.columns:
        configured_lengths = pd.to_numeric(length_df[factor_column], errors="coerce")
        if configured_lengths.isna().any():
            invalid_values = sorted(
                set(str(value) for value in length_df.loc[configured_lengths.isna(), factor_column])
            )
            raise ValueError(
                f"Study summarize expected numeric trial factor values for {factor_column}: {invalid_values}"
            )
        configured_raw = configured_lengths.to_numpy(dtype=float)
        configured_int = np.rint(configured_raw)
        if not np.allclose(configured_raw, configured_int, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                "Study summarize expected integer trial factor values for "
                f"{factor_column}, found {configured_raw.tolist()}"
            )
        length_df["sequence_length"] = configured_int.astype(int)
    return agg_df, length_df


def _build_summary_table_entries(
    *,
    trial_runs_path: Path,
    trial_metrics_path: Path,
    trial_metrics_agg_path: Path,
    length_agg_path: Path,
    n_missing_total: int,
) -> list[dict[str, object]]:
    return [
        {
            "key": "trial_runs",
            "path": trial_runs_path.name,
            "exists": trial_runs_path.exists(),
            "n_missing_total": int(n_missing_total),
        },
        {
            "key": "trial_metrics",
            "path": trial_metrics_path.name,
            "exists": trial_metrics_path.exists(),
            "n_missing_total": int(n_missing_total),
        },
        {
            "key": "trial_metrics_agg",
            "path": trial_metrics_agg_path.name,
            "exists": trial_metrics_agg_path.exists(),
            "n_missing_total": int(n_missing_total),
        },
        {
            "key": "length_tradeoff_agg",
            "path": length_agg_path.name,
            "exists": length_agg_path.exists(),
            "n_missing_total": int(n_missing_total),
        },
    ]


def _write_mmr_tradeoff_outputs(
    *,
    study_run_dir: Path,
    mmr_rows: list[pd.DataFrame],
    require_mmr: bool,
    allow_partial: bool,
    n_missing_total: int,
    n_missing_non_success: int,
    n_missing_run_dirs: int,
    n_missing_metric_artifacts: int,
    n_missing_mmr_tables: int,
    table_entries: list[dict[str, object]],
    plot_mmr_diversity_tradeoff_fn: Callable[[pd.DataFrame, Path], None],
    base_diversity: float,
) -> Path | None:
    mmr_tradeoff_path = study_table_path(study_run_dir, "mmr_tradeoff_agg", "parquet")
    mmr_plot_output_path = study_plot_path(study_run_dir, "mmr_diversity_tradeoff", "pdf")
    if not mmr_rows:
        _remove_if_exists(mmr_tradeoff_path)
        _remove_if_exists(mmr_plot_output_path)
        if require_mmr and not allow_partial:
            raise FileNotFoundError(
                "Study summarize expected MMR replay tables but none were found. "
                "Run `cruncher study run --resume --spec <path>` to regenerate replay outputs."
            )
        return None

    mmr_df = pd.concat(mmr_rows, ignore_index=True)
    mmr_group_cols = ["target_set_index", "trial_id", "series_label", "diversity", "pool_size_input"]
    mmr_agg = mmr_df[mmr_group_cols].drop_duplicates().reset_index(drop=True)
    mmr_metric_cols = ["median_joint_score_selected", "median_nn_full_bp", "median_nn_full_distance"]
    for metric_idx, metric in enumerate(mmr_metric_cols):
        if metric in mmr_df.columns:
            mmr_metric = _aggregate_metric(mmr_df, mmr_group_cols, metric)
            if metric_idx > 0 and "n_replicates" in mmr_metric.columns:
                mmr_metric = mmr_metric.drop(columns=["n_replicates"])
            mmr_agg = mmr_agg.merge(mmr_metric, on=mmr_group_cols, how="left")
    mmr_agg["n_missing_total"] = int(n_missing_total)
    mmr_agg["n_missing_non_success"] = int(n_missing_non_success)
    mmr_agg["n_missing_run_dirs"] = int(n_missing_run_dirs)
    mmr_agg["n_missing_metric_artifacts"] = int(n_missing_metric_artifacts)
    mmr_agg["n_missing_mmr_tables"] = int(n_missing_mmr_tables)
    write_parquet(mmr_agg, mmr_tradeoff_path)
    table_entries.append(
        {
            "key": "mmr_tradeoff_agg",
            "path": mmr_tradeoff_path.name,
            "exists": mmr_tradeoff_path.exists(),
            "n_missing_total": int(n_missing_total),
        }
    )

    mmr_plot_df = mmr_agg.rename(
        columns={
            "median_joint_score_selected_mean": "score_mean",
            "median_joint_score_selected_sem": "score_sem",
        }
    )
    if "median_nn_full_bp_mean" in mmr_plot_df.columns:
        mmr_plot_df["diversity_metric_mean"] = mmr_plot_df["median_nn_full_bp_mean"]
        mmr_plot_df["diversity_metric_sem"] = mmr_plot_df["median_nn_full_bp_sem"]
        mmr_plot_df["diversity_metric_label"] = "Median NN full-seq Hamming (bp)"
    else:
        mmr_plot_df["diversity_metric_mean"] = mmr_plot_df["median_nn_full_distance_mean"]
        mmr_plot_df["diversity_metric_sem"] = mmr_plot_df["median_nn_full_distance_sem"]
        mmr_plot_df["diversity_metric_label"] = "Median NN full-seq distance"
    diversity_values = pd.to_numeric(mmr_plot_df["diversity"], errors="coerce")
    mmr_plot_df["is_base_value"] = np.isclose(diversity_values, float(base_diversity), rtol=0.0, atol=1e-9)
    plot_mmr_diversity_tradeoff_fn(mmr_plot_df, mmr_plot_output_path)
    return mmr_plot_output_path


def _write_length_tradeoff_plot(
    study_run_dir: Path,
    length_df: pd.DataFrame,
    *,
    plot_sequence_length_tradeoff_fn: Callable[[pd.DataFrame, Path], None],
    base_sequence_length: int,
) -> Path | None:
    length_plot_df = length_df.rename(
        columns={
            "median_score_mean": "score_mean",
            "median_score_sem": "score_sem",
            "median_nn_full_bp_mean": "diversity_metric_mean",
            "median_nn_full_bp_sem": "diversity_metric_sem",
        }
    )
    if "diversity_metric_mean" not in length_plot_df.columns:
        length_plot_df["diversity_metric_mean"] = length_plot_df["median_nn_full_distance_mean"]
        length_plot_df["diversity_metric_sem"] = length_plot_df["median_nn_full_distance_sem"]
        length_plot_df["diversity_metric_label"] = "Median NN full-seq distance"
    else:
        length_plot_df["diversity_metric_label"] = "Median NN full-seq Hamming (bp)"
    length_values = pd.to_numeric(length_plot_df["sequence_length"], errors="coerce")
    length_plot_df["is_base_value"] = np.isclose(length_values, float(base_sequence_length), rtol=0.0, atol=1e-9)
    length_plot_path = study_plot_path(study_run_dir, "sequence_length_tradeoff", "pdf")
    unique_lengths = pd.to_numeric(length_plot_df["sequence_length"], errors="coerce").dropna().astype(float).to_numpy()
    if len(np.unique(unique_lengths)) < 2:
        _remove_if_exists(length_plot_path)
        return None
    plot_sequence_length_tradeoff_fn(length_plot_df, length_plot_path)
    return length_plot_path


def summarize_study_run(study_run_dir: Path, *, allow_partial: bool = False) -> StudySummaryResult:
    manifest_file = study_manifest_path(study_run_dir)
    status_file = study_status_path(study_run_dir)
    manifest = load_study_manifest(manifest_file)
    status = load_study_status(status_file)

    frozen_spec = _frozen_study_spec(study_run_dir)
    base_plot_values = _resolve_base_plot_values(frozen_spec)
    require_mmr = bool(frozen_spec.replays.mmr_sweep.enabled)
    exit_code_policy = str(frozen_spec.execution.exit_code_policy)
    successful_runs, n_non_success_runs = _resolve_successful_runs(manifest, allow_partial=allow_partial)

    trial_rows = _serialize_trial_rows([item.model_dump(mode="json") for item in manifest.trial_runs])
    trial_runs_df = pd.DataFrame(trial_rows)
    trial_runs_path = study_table_path(study_run_dir, "trial_runs", "parquet")
    write_parquet(trial_runs_df, trial_runs_path)

    metric_collection = _collect_trial_metrics(successful_runs=successful_runs, allow_partial=allow_partial)
    trial_metrics_df = pd.DataFrame(metric_collection.rows)
    trial_metrics_path = study_table_path(study_run_dir, "trial_metrics", "parquet")
    write_parquet(trial_metrics_df, trial_metrics_path)
    sequence_length_by_run = _sequence_length_by_trial(trial_metrics_df)
    mmr_rows, n_missing_mmr_tables = _collect_mmr_tables(
        successful_runs=successful_runs,
        summarized_keys=metric_collection.summarized_keys,
        sequence_length_by_run=sequence_length_by_run,
        require_mmr=require_mmr,
        allow_partial=allow_partial,
    )
    missing_counts = _resolve_missing_counts(
        allow_partial=allow_partial,
        n_non_success_runs=n_non_success_runs,
        n_missing_run_dirs=metric_collection.n_missing_run_dirs,
        n_missing_metric_artifacts=metric_collection.n_missing_metric_artifacts,
        n_missing_mmr_tables=n_missing_mmr_tables,
    )
    agg_df, length_df = _aggregate_trial_metrics_for_summary(
        trial_metrics_df=trial_metrics_df,
        n_missing_total=missing_counts.n_missing_total,
        n_missing_non_success=missing_counts.n_missing_non_success,
        n_missing_run_dirs=metric_collection.n_missing_run_dirs,
        n_missing_metric_artifacts=metric_collection.n_missing_metric_artifacts,
        n_missing_mmr_tables=n_missing_mmr_tables,
    )
    trial_metrics_agg_path = study_table_path(study_run_dir, "trial_metrics_agg", "parquet")
    write_parquet(agg_df, trial_metrics_agg_path)
    length_agg_path = study_table_path(study_run_dir, "length_tradeoff_agg", "parquet")
    write_parquet(length_df, length_agg_path)

    table_entries = _build_summary_table_entries(
        trial_runs_path=trial_runs_path,
        trial_metrics_path=trial_metrics_path,
        trial_metrics_agg_path=trial_metrics_agg_path,
        length_agg_path=length_agg_path,
        n_missing_total=missing_counts.n_missing_total,
    )

    ensure_mpl_cache(study_meta_dir(study_run_dir))
    from dnadesign.cruncher.study import plots as study_plots

    mmr_plot_path = _write_mmr_tradeoff_outputs(
        study_run_dir=study_run_dir,
        mmr_rows=mmr_rows,
        require_mmr=require_mmr,
        allow_partial=allow_partial,
        n_missing_total=missing_counts.n_missing_total,
        n_missing_non_success=missing_counts.n_missing_non_success,
        n_missing_run_dirs=metric_collection.n_missing_run_dirs,
        n_missing_metric_artifacts=metric_collection.n_missing_metric_artifacts,
        n_missing_mmr_tables=n_missing_mmr_tables,
        table_entries=table_entries,
        plot_mmr_diversity_tradeoff_fn=study_plots.plot_mmr_diversity_tradeoff,
        base_diversity=base_plot_values.diversity,
    )
    length_plot_path = _write_length_tradeoff_plot(
        study_run_dir,
        length_df,
        plot_sequence_length_tradeoff_fn=study_plots.plot_sequence_length_tradeoff,
        base_sequence_length=base_plot_values.sequence_length,
    )
    plot_entries: list[dict[str, object]] = []
    if length_plot_path is not None:
        plot_entries.append(
            {"key": "sequence_length_tradeoff", "path": length_plot_path.name, "exists": length_plot_path.exists()}
        )
    if mmr_plot_path is not None:
        plot_entries.append(
            {"key": "mmr_diversity_tradeoff", "path": mmr_plot_path.name, "exists": mmr_plot_path.exists()}
        )

    _write_study_manifests(
        study_run_dir,
        table_entries=table_entries,
        plot_entries=plot_entries,
        study_id=manifest.study_id,
        study_name=manifest.study_name,
    )

    if allow_partial and missing_counts.n_missing_total > 0:
        warning = (
            "summarize used partial data: "
            f"n_missing_total={missing_counts.n_missing_total} "
            f"(non_success={missing_counts.n_missing_non_success}, "
            f"missing_run_dirs={metric_collection.n_missing_run_dirs}, "
            f"missing_metric_artifacts={metric_collection.n_missing_metric_artifacts}, "
            f"missing_mmr_tables={n_missing_mmr_tables})"
        )
        if warning not in status.warnings:
            status.warnings.append(warning)
    status.updated_at = utc_now_iso()
    write_study_status(status_file, status)
    return StudySummaryResult(
        study_run_dir=study_run_dir,
        n_missing_total=int(missing_counts.n_missing_total),
        n_missing_non_success=int(missing_counts.n_missing_non_success),
        n_missing_run_dirs=int(metric_collection.n_missing_run_dirs),
        n_missing_metric_artifacts=int(metric_collection.n_missing_metric_artifacts),
        n_missing_mmr_tables=int(n_missing_mmr_tables),
        exit_code_policy=exit_code_policy,
    )
