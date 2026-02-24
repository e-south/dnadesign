"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/portfolio_studies.py

Helpers for portfolio study-spec resolution and study-derived summary extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.app.portfolio_preflight import _resolve_source_label
from dnadesign.cruncher.portfolio.schema_models import PortfolioSource, PortfolioSpec
from dnadesign.cruncher.study.identity import resolve_deterministic_study_run_dir
from dnadesign.cruncher.study.layout import study_manifest_path, study_status_path, study_table_path
from dnadesign.cruncher.study.manifest import load_study_manifest, load_study_status


def run_study(*args, **kwargs):
    from dnadesign.cruncher.app.study_workflow import run_study as _run_study

    return _run_study(*args, **kwargs)


RunStudyFn = Callable[..., Path]


def _ensure_required_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _resolve_source_study_spec_path(source: PortfolioSource, *, study_spec: Path, label: str) -> Path:
    source_workspace = source.workspace.resolve()
    candidate = study_spec if study_spec.is_absolute() else (source_workspace / study_spec)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(source_workspace)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside source workspace: id={source.id!r} study_spec={resolved} "
            f"workspace={source_workspace}"
        ) from exc
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found for source id={source.id!r}: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file for source id={source.id!r}: {resolved}")
    return resolved


def _ensure_study_run_completed(study_spec_path: Path, *, run_study_fn: RunStudyFn) -> Path:
    study_run_dir = resolve_deterministic_study_run_dir(study_spec_path)
    if not study_run_dir.exists():
        run_study_fn(
            study_spec_path,
            resume=False,
            force_overwrite=False,
            progress_bar=False,
            quiet_logs=True,
        )
        return resolve_deterministic_study_run_dir(study_spec_path)

    manifest_file = study_manifest_path(study_run_dir)
    status_file = study_status_path(study_run_dir)
    if not manifest_file.exists() or not status_file.exists():
        raise FileNotFoundError(
            "Study run directory is missing required metadata files for portfolio aggregation: "
            f"study_run_dir={study_run_dir}"
        )
    status = load_study_status(status_file)
    if status.status in {"completed", "completed_with_errors"}:
        return study_run_dir

    run_study_fn(
        study_spec_path,
        resume=True,
        force_overwrite=False,
        progress_bar=False,
        quiet_logs=True,
    )
    refreshed = load_study_status(status_file)
    if refreshed.status not in {"completed", "completed_with_errors"}:
        raise ValueError(
            "Study run did not complete after resume and cannot be aggregated by portfolio: "
            f"study_run_dir={study_run_dir} status={refreshed.status!r}"
        )
    return study_run_dir


def _ensure_required_source_studies(spec: PortfolioSpec, *, run_study_fn: RunStudyFn) -> dict[tuple[str, str], Path]:
    return _ensure_required_source_studies_for_sources(spec, spec.sources, run_study_fn=run_study_fn)


def _ensure_required_source_studies_for_sources(
    spec: PortfolioSpec,
    sources: list[PortfolioSource],
    *,
    run_study_fn: RunStudyFn,
) -> dict[tuple[str, str], Path]:
    ensured: dict[tuple[str, str], Path] = {}
    if not spec.studies.ensure_specs:
        return ensured

    for source in sources:
        for relative_spec in spec.studies.ensure_specs:
            resolved_spec = _resolve_source_study_spec_path(
                source,
                study_spec=relative_spec,
                label="portfolio.studies.ensure_specs entry",
            )
            run_dir = _ensure_study_run_completed(resolved_spec, run_study_fn=run_study_fn)
            ensured[(str(source.id), str(resolved_spec))] = run_dir
    return ensured


def _load_source_study_summary(source: PortfolioSource, *, run_study_fn: RunStudyFn) -> dict[str, object] | None:
    study_spec = getattr(source, "study_spec", None)
    if study_spec is None:
        return None
    if not isinstance(study_spec, Path):
        raise ValueError(f"Portfolio source study_spec must resolve to a path: id={source.id!r}")

    resolved_spec = _resolve_source_study_spec_path(
        source,
        study_spec=study_spec,
        label="portfolio.sources[].study_spec",
    )
    study_run_dir = _ensure_study_run_completed(resolved_spec, run_study_fn=run_study_fn)
    manifest = load_study_manifest(study_manifest_path(study_run_dir))
    status = load_study_status(study_status_path(study_run_dir))
    if status.status not in {"completed", "completed_with_errors"}:
        raise ValueError(
            "Portfolio source study status must be completed for aggregation: "
            f"id={source.id!r} study={manifest.study_name!r} status={status.status!r}"
        )

    agg_path = study_table_path(study_run_dir, "trial_metrics_agg", "parquet")
    if not agg_path.exists():
        raise FileNotFoundError(
            "Portfolio source study summary table missing: "
            f"id={source.id!r} study={manifest.study_name!r} path={agg_path}"
        )
    study_df = read_parquet(agg_path)
    if study_df.empty:
        raise ValueError(
            "Portfolio source study summary table is empty: "
            f"id={source.id!r} study={manifest.study_name!r} path={agg_path}"
        )

    _ensure_required_columns(
        study_df,
        ["trial_id", "median_score_mean", "best_score_mean"],
        context=f"Portfolio source study summary ({agg_path})",
    )
    median_scores = pd.to_numeric(study_df["median_score_mean"], errors="coerce")
    if median_scores.isna().all():
        raise ValueError(
            "Portfolio source study summary has no numeric median_score_mean values: "
            f"id={source.id!r} study={manifest.study_name!r}"
        )
    best_scores = pd.to_numeric(study_df["best_score_mean"], errors="coerce")
    nn_bp = pd.to_numeric(study_df.get("median_nn_full_bp_mean"), errors="coerce")

    return {
        "source_id": str(source.id),
        "source_label": _resolve_source_label(source),
        "workspace_name": source.workspace.name,
        "workspace_path": str(source.workspace),
        "study_name": manifest.study_name,
        "study_id": manifest.study_id,
        "study_status": status.status,
        "study_run_dir": str(study_run_dir),
        "n_trials_agg_rows": int(len(study_df)),
        "median_score_mean_max": float(median_scores.max()),
        "best_score_mean_max": float(best_scores.max()) if not best_scores.isna().all() else None,
        "median_nn_full_bp_mean_max": float(nn_bp.max()) if not nn_bp.isna().all() else None,
    }


def _load_source_sequence_length_rows(
    source: PortfolioSource,
    *,
    study_spec: Path,
    top_n_lengths: int,
    ensured_study_runs: dict[tuple[str, str], Path],
    run_study_fn: RunStudyFn,
) -> list[dict[str, object]]:
    resolved_spec = _resolve_source_study_spec_path(
        source,
        study_spec=study_spec,
        label="portfolio.studies.sequence_length_table.study_spec",
    )
    run_key = (str(source.id), str(resolved_spec))
    study_run_dir = ensured_study_runs.get(run_key)
    if study_run_dir is None:
        study_run_dir = _ensure_study_run_completed(resolved_spec, run_study_fn=run_study_fn)

    manifest = load_study_manifest(study_manifest_path(study_run_dir))
    status = load_study_status(study_status_path(study_run_dir))
    if status.status not in {"completed", "completed_with_errors"}:
        raise ValueError(
            "Portfolio sequence-length study run must be completed: "
            f"id={source.id!r} study={manifest.study_name!r} status={status.status!r}"
        )

    length_path = study_table_path(study_run_dir, "length_tradeoff_agg", "parquet")
    if not length_path.exists():
        raise FileNotFoundError(
            "Portfolio sequence-length study table missing: "
            f"id={source.id!r} study={manifest.study_name!r} path={length_path}"
        )

    length_df = read_parquet(length_path)
    if length_df.empty:
        raise ValueError(
            "Portfolio sequence-length study table is empty: "
            f"id={source.id!r} study={manifest.study_name!r} path={length_path}"
        )
    _ensure_required_columns(
        length_df,
        ["trial_id", "sequence_length", "median_score_mean", "best_score_mean"],
        context=f"Portfolio sequence-length summary ({length_path})",
    )

    length_df = length_df.copy()
    length_df["sequence_length"] = pd.to_numeric(length_df["sequence_length"], errors="coerce")
    length_df["median_score_mean"] = pd.to_numeric(length_df["median_score_mean"], errors="coerce")
    length_df["best_score_mean"] = pd.to_numeric(length_df["best_score_mean"], errors="coerce")
    if length_df["sequence_length"].isna().any():
        raise ValueError(
            "Portfolio sequence-length summary contains non-numeric sequence_length values: "
            f"id={source.id!r} study={manifest.study_name!r}"
        )
    if length_df["median_score_mean"].isna().all():
        raise ValueError(
            "Portfolio sequence-length summary has no numeric median_score_mean values: "
            f"id={source.id!r} study={manifest.study_name!r}"
        )

    length_df = length_df.sort_values(["sequence_length", "median_score_mean"], ascending=[True, False])
    best_per_length = length_df.groupby("sequence_length", as_index=False).head(1)
    selected = best_per_length.head(int(top_n_lengths))

    rows: list[dict[str, object]] = []
    for item in selected.to_dict(orient="records"):
        seq_len = float(item["sequence_length"])
        rows.append(
            {
                "source_id": str(source.id),
                "source_label": _resolve_source_label(source),
                "workspace_name": source.workspace.name,
                "workspace_path": str(source.workspace),
                "study_name": manifest.study_name,
                "study_id": manifest.study_id,
                "study_run_dir": str(study_run_dir),
                "trial_id": str(item["trial_id"]),
                "sequence_length": int(round(seq_len)),
                "median_score_mean": float(item["median_score_mean"]),
                "best_score_mean": float(item["best_score_mean"]) if pd.notna(item["best_score_mean"]) else None,
                "median_nn_full_bp_mean": float(item["median_nn_full_bp_mean"])
                if pd.notna(item.get("median_nn_full_bp_mean"))
                else None,
            }
        )
    return rows


__all__ = [
    "_ensure_required_source_studies",
    "_ensure_required_source_studies_for_sources",
    "_ensure_study_run_completed",
    "_load_source_sequence_length_rows",
    "_load_source_study_summary",
    "_resolve_source_study_spec_path",
]
